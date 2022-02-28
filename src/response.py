"""
Nonlinear time-history analysis of a predefined building object
using a series of ground motion records
of varying levels of intensity to create a series of
peak response quantities for these levels of intensity,
suitable for subsequent use in PELICUN.

"""

import sys
sys.path.append("../OpenSeesPy_Building_Modeler")

import numpy as np
from scipy import integrate
import modeler
import solver
import time
import pickle
import sys
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--gm_dir')
parser.add_argument('--gm_dt')
parser.add_argument('--analysis_dt')
parser.add_argument('--gm_number')
parser.add_argument('--output_dir')

args = parser.parse_args()

ground_motion_dir = args.gm_dir  # 'ground_motions/test_case/parsed'
ground_motion_dt = float(args.gm_dt)  # 0.005
analysis_dt = float(args.analysis_dt)  # 0.05
gm_number = int(args.gm_number.replace('gm', ''))
output_folder = args.output_dir  # 'response/test_case'

# # debug
# ground_motion_dir = 'analysis/hazard_level_8/ground_motions/parsed'
# ground_motion_dt = 0.005
# analysis_dt = 0.01
# gm_number = 13
# output_folder = 'analysis/hazard_level_8/response/gm13'

# ~~~~~~~~~~ #
# parameters #
# ~~~~~~~~~~ #

# fundamental period of the building
t_1 = 0.945

# ~~~~~~~~~~~~~~~~~~~~ #
# function definitions #
# ~~~~~~~~~~~~~~~~~~~~ #


def get_duration(time_history_path, dt):
    """
    Get the duration of a fixed-step time-history
    stored in a text file, given its path and
    the time increment.
    """
    values = np.genfromtxt(time_history_path)
    num_points = len(values)
    return float(num_points) * dt


def retrieve_displacement_th(floor, drct, anal_obj):
    # retrieve the number of successful steps
    n_steps = anal_obj.n_steps_success
    # get the displacement time-history
    nid = anal_obj.building.list_of_parent_nodes()[floor].uniq_id
    d = []
    for i in range(n_steps):
        d.append(anal_obj.node_displacements[nid][i][drct])
    d = np.array(d)
    t = np.array(nlth.time_vector)
    return np.column_stack((t, d))


def retrieve_acceleration_th(floor, drct, anal_obj):
    # retrieve the number of successful steps
    n_steps = anal_obj.n_steps_success
    # get the acceleration time-history
    nid = anal_obj.building.list_of_parent_nodes()[floor].uniq_id
    d = []
    for i in range(n_steps):
        d.append(anal_obj.node_accelerations[nid][i][drct])
    d = np.array(d)
    t = np.array(nlth.time_vector)
    return np.column_stack((t, d))


def retrieve_velocity_th(floor, drct, anal_obj):
    # retrieve the number of successful steps
    n_steps = anal_obj.n_steps_success
    # get the acceleration time-history
    nid = anal_obj.building.list_of_parent_nodes()[floor].uniq_id
    d = []
    for i in range(n_steps):
        d.append(anal_obj.node_velocities[nid][i][drct])
    d = np.array(d)
    t = np.array(nlth.time_vector)
    return np.column_stack((t, d))


# ~~~~~~~~ #
# analysis #
# ~~~~~~~~ #

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

b = modeler.Building()

hi = np.array([15.00, 13.00, 13.00]) * 12.00  # in

b.add_level("base", 0.00, "fixed")
b.add_level("1", hi[0])
b.add_level("2", hi[0]+hi[1])
b.add_level("3", hi[0]+hi[1]+hi[2])

sections = dict(
    gravity_cols=dict(
        level_1="W14X90",
        level_2="W14X90",
        level_3="W14X90"),
    gravity_beams_perimeter=dict(
        level_1="W21X55",
        level_2="W21X55",
        level_3="W21X55"),
    gravity_beams_interior_32=dict(
        level_1="W12X152",
        level_2="W12X152",
        level_3="W12X152"),
    gravity_beams_interior_25=dict(
        level_1="W10X100",
        level_2="W10X100",
        level_3="W10X100"),
    secondary_beams="W14X30",
    lateral_cols=dict(
        level_1="W14X426",
        level_2="W14X426",
        level_3="W14X342"),
    lateral_beams=dict(
        level_1="W24X192",
        level_2="W24X192",
        level_3="W24X94")
    )

b.set_active_material('steel02-fy50')

# define sections
wsections = set()
for lvl_tag in ['level_1', 'level_2', 'level_3']:
    wsections.add(sections['gravity_beams_perimeter'][lvl_tag])
    wsections.add(sections['gravity_beams_interior_32'][lvl_tag])
    wsections.add(sections['gravity_beams_interior_25'][lvl_tag])
    wsections.add(sections['lateral_cols'][lvl_tag])
    wsections.add(sections['lateral_beams'][lvl_tag])
    wsections.add(sections['gravity_cols'][lvl_tag])
wsections.add(sections['secondary_beams'])

for sec in wsections:
    b.add_sections_from_json(
        "../OpenSeesPy_Building_Modeler/section_data/sections.json",
        'W',
        [sec])

nsub = 15  # element subdivision
pinned_ends = {'type': 'pinned', 'dist': 0.005}
fixedpinned_ends = {'type': 'fixed-pinned', 'dist': 0.005}
elastic_modeling_type = {'type': 'elastic'}
grav_col_ends = fixedpinned_ends
lat_bm_ends = {'type': 'steel_W_IMK', 'dist': 0.05,
               'Lb/ry': 60., 'L/H': 0.50, 'RBS_factor': 0.60,
               'composite action': True,
               'doubler plate thickness': 0.00}
lat_bm_modeling = {'type': 'elastic'}
lat_col_ends = {'type': 'steel_W_PZ_IMK', 'dist': 0.05,
                'Lb/ry': 60., 'L/H': 1.0, 'pgpye': 0.05,
                'doubler plate thickness': 0.00}
lat_col_modeling_type = {'type': 'elastic'}
lat_bm_modeling_type = {'type': 'elastic'}
grav_bm_ends = {'type': 'steel W shear tab', 'dist': 0.005,
                'composite action': True}
col_gtransf = 'Corotational'


#
# define structural members
#

# generate a dictionary containing coordinates given gridline tag names
# (here we won't use the native gridline objects,
#  since the geometry is very simple)
point = {}
x_grd_tags = ['A', 'B', 'C', 'D', 'E', 'F']
y_grd_tags = ['5', '4', '3', '2', '1']
x_grd_locs = np.array([0.00, 32.5, 57.5, 82.5, 107.5, 140.00]) * 12.00  # (in)
y_grd_locs = np.array([0.00, 25.00, 50.00, 75.00, 100.00]) * 12.00  # (in)

for i in range(len(x_grd_tags)):
    point[x_grd_tags[i]] = {}
    for j in range(len(y_grd_tags)):
        point[x_grd_tags[i]][y_grd_tags[j]] = \
            np.array([x_grd_locs[i], y_grd_locs[j]])

for level_counter in range(3):
    level_tag = 'level_'+str(level_counter+1)
    # define gravity columns
    b.set_active_angle(0.00)
    b.set_active_placement('centroid')
    b.set_active_levels([str(level_counter+1)])
    b.set_active_section(sections['gravity_cols'][level_tag])
    for tag in ['A', 'F']:
        pt = point[tag]['1']
        col = b.add_column_at_point(
            pt[0], pt[1], n_sub=1, ends=grav_col_ends,
            model_as=elastic_modeling_type, geomTransf=col_gtransf)
    for tag1 in ['B', 'C', 'D', 'E']:
        for tag2 in ['2', '3', '4']:
            pt = point[tag1][tag2]
            col = b.add_column_at_point(
                pt[0], pt[1], n_sub=1, ends=grav_col_ends,
                model_as=elastic_modeling_type, geomTransf=col_gtransf)

    # define X-dir frame columns
    b.set_active_section(sections['lateral_cols'][level_tag])
    b.set_active_angle(np.pi/2.00)
    for tag1 in ['B', 'C', 'D', 'E']:
        for tag2 in ['1', '5']:
            pt = point[tag1][tag2]
            b.add_column_at_point(
                pt[0], pt[1], n_sub=nsub, ends=lat_col_ends,
                model_as=lat_col_modeling_type, geomTransf=col_gtransf)
    # deffine Y-dir frame columns
    b.set_active_angle(0.00)
    for tag1 in ['A', 'F']:
        for tag2 in ['5', '4', '3', '2']:
            pt = point[tag1][tag2]
            b.add_column_at_point(
                pt[0], pt[1], n_sub=nsub, ends=lat_col_ends,
                model_as=lat_col_modeling_type, geomTransf=col_gtransf)
    # define X-dir frame beams
    b.set_active_section(sections['lateral_beams'][level_tag])
    b.set_active_placement('top_center')
    for tag1 in ['1', '5']:
        tag2_start = ['B', 'C', 'D']
        tag2_end = ['C', 'D', 'E']
        for j in range(len(tag2_start)):
            b.add_beam_at_points(
                point[tag2_start[j]][tag1],
                point[tag2_end[j]][tag1],
                ends=lat_bm_ends,
                model_as=lat_bm_modeling_type, n_sub=nsub,
                snap_i='bottom_center',
                snap_j='top_center')
    # define Y-dir frame beams
    for tag1 in ['A', 'F']:
        tag2_start = ['2', '3', '4']
        tag2_end = ['3', '4', '5']
        for j in range(len(tag2_start)):
            b.add_beam_at_points(
                point[tag1][tag2_start[j]],
                point[tag1][tag2_end[j]],
                ends=lat_bm_ends,
                model_as=lat_bm_modeling_type, n_sub=nsub,
                snap_i='bottom_center',
                snap_j='top_center')
    # define perimeter gravity beams
    b.set_active_section(sections['gravity_beams_perimeter'][level_tag])
    for tag1 in ['A', 'F']:
        tag2_start = ['1']
        tag2_end = ['2']
        for j in range(len(tag2_start)):
            b.add_beam_at_points(
                point[tag1][tag2_start[j]],
                point[tag1][tag2_end[j]],
                ends=grav_bm_ends)
    for tag1 in ['1', '5']:
        tag2_start = ['A', 'E']
        tag2_end = ['B', 'F']
        for j in range(len(tag2_start)):
            b.add_beam_at_points(
                point[tag2_start[j]][tag1],
                point[tag2_end[j]][tag1],
                ends=grav_bm_ends)
    # define interior gravity beams
    for tag1 in ['B', 'C', 'D', 'E']:
        b.set_active_section(
            sections['gravity_beams_interior_25'][level_tag])
        tag2_start = ['1', '2', '3', '4']
        tag2_end = ['2', '3', '4', '5']
        for j in range(len(tag2_start)):
            b.add_beam_at_points(
                point[tag1][tag2_start[j]],
                point[tag1][tag2_end[j]],
                ends=grav_bm_ends)
    for tag1 in ['2', '3', '4']:
        tag2_start = ['A', 'B', 'C', 'D', 'E']
        tag2_end = ['B', 'C', 'D', 'E', 'F']
        for j in range(len(tag2_start)):
            if tag2_start[j] in ['B', 'E']:
                b.set_active_section(
                    sections['gravity_beams_interior_32'][level_tag])
            else:
                b.set_active_section(
                    sections['gravity_beams_interior_25'][level_tag])
            b.add_beam_at_points(
                point[tag2_start[j]][tag1],
                point[tag2_end[j]][tag1],
                ends=grav_bm_ends)
    # define secondary beams
    b.set_active_section(sections['secondary_beams'])
    for tag1 in ['A', 'B', 'C', 'D', 'E']:
        tag2_start = ['1', '2', '3', '4']
        tag2_end = ['2', '3', '4', '5']
        if tag1 in ['A', 'E']:
            shifts = 32.5/4. * 12.  # in
            num = 3
        else:
            shifts = 25.0/3. * 12  # in
            num = 2
        shift = 0.00
        for i in range(num):
            shift += shifts
            for j in range(len(tag2_start)):
                b.add_beam_at_points(
                    point[tag1][tag2_start[j]] + np.array([shift, 0.00]),
                    point[tag1][tag2_end[j]] + np.array([shift, 0.00]),
                    offset_i=np.array([0., 0., 0.]),
                    offset_j=np.array([0., 0., 0.]),
                    ends=pinned_ends)

#
# define surface loads
#


b.set_active_levels(['1', '2'])
b.assign_surface_DL((75.+15.+20.+0.25*80.)/(12.**2))

b.set_active_levels(['3'])
b.assign_surface_DL((75.+15.+80.+0.25*20)/(12.**2))


# cladding - 1st story
b.select_perimeter_beams_story('1')
# 10 is the load in lb/ft2, we multiply it by the height
# the tributary area of the 1st story cladding support is
# half the height of the 1st story and half the height of the second
# we get lb/ft, so we divide by 12 to convert this to lb/in
# which is what OpenSeesPy_Building_Modeler uses.
b.selection.add_UDL(np.array((0.00, 0.00,
                              -((10./12.**2) * (hi[0] + hi[1]) / 2.00))))

# cladding - 2nd story
b.selection.clear()
b.select_perimeter_beams_story('2')
b.selection.add_UDL(np.array((0.00, 0.00,
                              -((10./12.**2) * (hi[1] + hi[2]) / 2.00))))

# cladding - roof
b.selection.clear()
b.select_perimeter_beams_story('3')
b.selection.add_UDL(np.array((0.00, 0.00,
                              -((10./12.**2) * hi[2] / 2.00))))
b.selection.clear()

b.preprocess(assume_floor_slabs=True, self_weight=True,
             steel_panel_zones=True, elevate_column_splices=0.25)




# retrieve some info used in the for loops
num_levels = len(b.levels.level_list) - 1
level_heights = []
for level in b.levels.level_list:
    level_heights.append(level.elevation)
level_heights = np.diff(level_heights)


# define analysis object
nlth = solver.NLTHAnalysis(b)

# get the corresponding ground motion duration
gm_X_filepath = ground_motion_dir + '/' + str(gm_number) + 'x.txt'
gm_Y_filepath = ground_motion_dir + '/' + str(gm_number) + 'y.txt'
gm_Z_filepath = ground_motion_dir + '/' + str(gm_number) + 'z.txt'

dx = get_duration(gm_X_filepath, ground_motion_dt)
dy = get_duration(gm_Y_filepath, ground_motion_dt)
dz = get_duration(gm_Z_filepath, ground_motion_dt)
duration = np.min(np.array((dx, dy, dz)))  # note: actually should be =


# run the nlth analysis
metadata = nlth.run(analysis_dt,
                    ground_motion_dir + '/' + str(gm_number) + 'x.txt',
                    ground_motion_dir + '/' + str(gm_number) + 'y.txt',
                    ground_motion_dir + '/' + str(gm_number) + 'z.txt',
                    ground_motion_dt,
                    finish_time=0.00,
                    damping_ratio=0.03,
                    num_modes=3,
                    printing=False,
                    data_retention='lightweight')


if not metadata['analysis_finished_successfully']:
    print('Analysis failed.')
    # print(args)
    sys.exit()

# ~~~~~~~~~~~~~~~~ #
# collect response #
# ~~~~~~~~~~~~~~~~ #


# ground acceleration, velocity and displacement
# interpolation functions

ag = {}  # g units
ag[0] = np.genfromtxt(gm_X_filepath)
ag[1] = np.genfromtxt(gm_Y_filepath)
n_pts = len(ag[0])
t = np.linspace(0.00, ground_motion_dt*n_pts, n_pts)
vg = {}  # in/s units
vg[0] = integrate.cumulative_trapezoid(
    ag[0]*modeler.common.G_CONST, t, initial=0)
vg[1] = integrate.cumulative_trapezoid(
    ag[1]*modeler.common.G_CONST, t, initial=0)
dg = {}  # in units
dg[0] = integrate.cumulative_trapezoid(vg[0], t, initial=0)
dg[1] = integrate.cumulative_trapezoid(vg[1], t, initial=0)

fag = {}
fag[0] = interp1d(t, ag[0], bounds_error=False, fill_value=0.00)
fag[1] = interp1d(t, ag[1], bounds_error=False, fill_value=0.00)
fvg = {}
fvg[0] = interp1d(t, vg[0], bounds_error=False, fill_value=0.00)
fvg[1] = interp1d(t, vg[1], bounds_error=False, fill_value=0.00)
fdg = {}
fdg[0] = interp1d(t, dg[0], bounds_error=False, fill_value=0.00)
fdg[1] = interp1d(t, dg[1], bounds_error=False, fill_value=0.00)


if not os.path.exists(output_folder):
    os.mkdir(output_folder)

time_vec = np.array(nlth.time_vector)
num_levels = 3

for direction in range(2):
    # store response time-histories
    # ground acceleration
    np.savetxt(f'{output_folder}/FA-0-{direction+1}.csv',
               fag[direction](time_vec))
    # ground velocity
    np.savetxt(f'{output_folder}/FV-0-{direction+1}.csv',
               fvg[direction](time_vec))
    for lvl in range(num_levels):
        # story drifts
        if lvl == 0:
            u = retrieve_displacement_th(lvl, direction, nlth)
            dr = u[:, 1] / level_heights[lvl]
        else:
            uprev = retrieve_displacement_th(lvl-1, direction, nlth)
            u = retrieve_displacement_th(lvl, direction, nlth)
            dr = (u[:, 1] - uprev[:, 1]) / level_heights[lvl]
        # story accelerations
        a1 = retrieve_acceleration_th(lvl, direction, nlth)
        # story velocities
        vel = retrieve_velocity_th(lvl, direction, nlth)

        np.savetxt(f'{output_folder}/ID-{lvl+1}-{direction+1}.csv', dr)
        np.savetxt(f'{output_folder}/FA-{lvl+1}-{direction+1}.csv',
                   a1[:, 1]/modeler.common.G_CONST + fag[direction](a1[:, 0]))
        np.savetxt(f'{output_folder}/FV-{lvl+1}-{direction+1}.csv',
                   vel[:, 1] + fvg[direction](vel[:, 0]))

    # global building drift
    bdr = retrieve_displacement_th(num_levels-1, direction, nlth)
    bdr[:, 1] /= np.sum(level_heights)
    np.savetxt(f'{output_folder}/BD-{direction+1}.csv', dr)

