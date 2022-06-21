import sys
sys.path.append("../OpenSees_Model_Builder/src")

import numpy as np
from scipy import integrate
import model
import solver
import time
import pickle
import sys
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# ~~~~~~~~~~~~~~~~~~~~~~ #
# set up argument parser #
# ~~~~~~~~~~~~~~~~~~~~~~ #

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

# ground_motion_dir = 'analysis/office3/hazard_level_8/ground_motions/parsed'
# ground_motion_dt = 0.005
# analysis_dt = 0.001
# gm_number = 1
# output_folder = '/tmp/TEST'


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

# ~~~~~~~~ #
# analysis #
# ~~~~~~~~ #

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

b = model.Model()

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
    gravity_beams_a=dict(
        level_1="W16X31",
        level_2="W16X31",
        level_3="W16X31"),
    gravity_beams_b=dict(
        level_1="W21X44",
        level_2="W21X44",
        level_3="W21X44"),
    gravity_beams_c=dict(
        level_1="W24X62",
        level_2="W24X62",
        level_3="W24X62"),
    gravity_beams_d=dict(
        level_1="W21X44",
        level_2="W21X44",
        level_3="W21X48"),
    gravity_beams_e=dict(
        level_1="W16X31",
        level_2="W16X31",
        level_3="W16X31"),
    lateral_cols=dict(
        exterior=dict(
            level_1="W24X131",
            level_2="W24X131",
            level_3="W24X131"),
        interior=dict(
            level_1="W24X192",
            level_2="W24X192",
            level_3="W24X146")),
    lateral_beams=dict(
        level_1="W30X116",
        level_2="W30X116",
        level_3="W24X62")
    )

doubler_plate_thicknesses = dict(
    exterior=dict(
        level_1=0.326704,
        level_2=0.326704,
        level_3=0.00
    ),
    interior=dict(
        level_1=0.964907,
        level_2=0.964907,
        level_3=0.297623
    )
)

# define materials
b.set_active_material('steel02-fy50')

# define sections
wsections = set()
for lvl_tag in ['level_1', 'level_2', 'level_3']:
    wsections.add(sections['gravity_beams_a'][lvl_tag])
    wsections.add(sections['gravity_beams_b'][lvl_tag])
    wsections.add(sections['gravity_beams_c'][lvl_tag])
    wsections.add(sections['gravity_beams_d'][lvl_tag])
    wsections.add(sections['gravity_beams_e'][lvl_tag])
    wsections.add(sections['lateral_beams'][lvl_tag])
    wsections.add(sections['gravity_cols'][lvl_tag])
for function in ['exterior', 'interior']:
    for lvl_tag in ['level_1', 'level_2', 'level_3']:
        wsections.add(sections['lateral_cols'][function][lvl_tag])


for sec in wsections:
    b.add_sections_from_json(
        "../OpenSees_Model_Builder/section_data/sections.json",
        'W',
        [sec])

#
# define structural members
#
pinned_ends = {'type': 'pinned', 'end_dist': 0.005}
fixedpinned_ends = {'type': 'fixed-pinned', 'end_dist': 0.005,
                    'doubler plate thickness': 0.00}
elastic_modeling_type = {'type': 'elastic'}
col_gtransf = 'Corotational'
nsub = 1  # element subdivision
grav_col_ends = fixedpinned_ends

# generate a dictionary containing coordinates given gridline tag names
# (here we won't use the native gridline objects,
#  since the geometry is very simple)
point = {}
x_grd_tags = ['A', 'B', 'C', 'D', 'E', 'F']
y_grd_tags = ['5', '4', '3', '2', '1']
x_grd_locs = np.array([0.00, 32.5, 57.5, 82.5, 107.5, 140.00]) * 12.00  # (in)
y_grd_locs = np.array([0.00, 25.00, 50.00, 75.00, 100.00]) * 12.00  # (in)

lat_bm_ends_a = {'type': 'steel_W_IMK', 'end_dist': (6.00+10.0)/(25.*12.-5.),
                 'Lb/ry': 60., 'L/H': 0.50, 'RBS_factor': 0.60,
                 'composite action': True,
                 'doubler plate thickness': 0.00}
lat_bm_ends_b = {'type': 'steel_W_IMK', 'end_dist': (4.00+10.00)/(25.*12.-5.),
                 'Lb/ry': 60., 'L/H': 0.50, 'RBS_factor': 0.60,
                 'composite action': True,
                 'doubler plate thickness': 0.00}
lat_bm_modeling = {'type': 'elastic'}
lat_col_ends = {'type': 'steel_W_PZ_IMK', 'end_dist': 0.05,
                'Lb/ry': 60., 'L/H': 1.0, 'pgpye': 0.05,
                'doubler plate thickness': 0.00}
lat_col_modeling_type = {'type': 'elastic'}
lat_bm_modeling_type = {'type': 'elastic'}
grav_bm_ends = {'type': 'steel W shear tab', 'end_dist': 0.005,
                'composite action': True}

for i in range(len(x_grd_tags)):
    point[x_grd_tags[i]] = {}
    for j in range(len(y_grd_tags)):
        point[x_grd_tags[i]][y_grd_tags[j]] = \
            np.array([x_grd_locs[i], y_grd_locs[j]])

for level_counter in range(3):
    if level_counter in [0, 1]:
        lat_bm_ends = lat_bm_ends_a
    else:
        lat_bm_ends = lat_bm_ends_b
    level_tag = 'level_'+str(level_counter+1)
    # define gravity columns
    b.set_active_angle(0.00)
    b.set_active_placement('centroid')
    b.set_active_levels([str(level_counter+1)])
    b.set_active_section(sections['gravity_cols'][level_tag])
    for tag in ['A', 'F']:
        pt = point[tag]['1']
        col = b.add_column_at_point(
            pt, n_sub=1, ends=grav_col_ends,
            model_as=elastic_modeling_type, geom_transf=col_gtransf)
    for tag1 in ['B', 'C', 'D', 'E']:
        for tag2 in ['2', '3', '4']:
            pt = point[tag1][tag2]
            col = b.add_column_at_point(
                pt, n_sub=1, ends=grav_col_ends,
                model_as=elastic_modeling_type, geom_transf=col_gtransf)

    # define X-dir frame columns
    b.set_active_angle(np.pi/2.00)
    b.set_active_section(sections['lateral_cols']['exterior'][level_tag])
    for tag1 in ['B', 'E']:
        for tag2 in ['1', '5']:
            pt = point[tag1][tag2]
            dbth = doubler_plate_thicknesses['exterior'][level_tag]
            lat_col_ends['doubler plate thickness'] = dbth
            b.add_column_at_point(
                pt, n_sub=nsub, ends=lat_col_ends,
                model_as=lat_col_modeling_type, geom_transf=col_gtransf)
    b.set_active_section(sections['lateral_cols']['interior'][level_tag])
    for tag1 in ['C', 'D']:
        for tag2 in ['1', '5']:
            pt = point[tag1][tag2]
            dbth = doubler_plate_thicknesses['interior'][level_tag]
            lat_col_ends['doubler plate thickness'] = dbth
            b.add_column_at_point(
                pt, n_sub=nsub, ends=lat_col_ends,
                model_as=lat_col_modeling_type, geom_transf=col_gtransf)
    # deffine Y-dir frame columns
    b.set_active_angle(0.00)
    b.set_active_section(sections['lateral_cols']['exterior'][level_tag])
    for tag1 in ['A', 'F']:
        for tag2 in ['5', '2']:
            pt = point[tag1][tag2]
            b.add_column_at_point(
                pt, n_sub=nsub, ends=lat_col_ends,
                model_as=lat_col_modeling_type, geom_transf=col_gtransf)
    b.set_active_section(sections['lateral_cols']['interior'][level_tag])
    for tag1 in ['A', 'F']:
        for tag2 in ['4', '3']:
            pt = point[tag1][tag2]
            b.add_column_at_point(
                pt, n_sub=nsub, ends=lat_col_ends,
                model_as=lat_col_modeling_type, geom_transf=col_gtransf)
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

    # define gravity beams of designation A
    b.set_active_section(sections['gravity_beams_a'][level_tag])
    for tag1 in ['A', 'F']:
        tag2_start = ['1']
        tag2_end = ['2']
        for j in range(len(tag2_start)):
            b.add_beam_at_points(
                point[tag1][tag2_start[j]],
                point[tag1][tag2_end[j]],
                ends=grav_bm_ends,
                snap_i='bottom_center',
                snap_j='top_center')
    for tag1 in ['B', 'C', 'D', 'E']:
        tag2_start = ['1', '2', '3', '4']
        tag2_end = ['2', '3', '4', '5']
        for j in range(len(tag2_start)):
            if tag2_start[j] == '1':
                si = 'center_left'
                sj = 'top_center'
            elif tag2_end[j] == '5':
                si = 'bottom_center'
                sj = 'center_right'
            else:
                si = 'bottom_center'
                sj = 'top_center'
            b.add_beam_at_points(
                point[tag1][tag2_start[j]],
                point[tag1][tag2_end[j]],
                ends=grav_bm_ends,
                snap_i=si,
                snap_j=sj)

    # define gravity beams of designation B
    b.set_active_section(sections['gravity_beams_b'][level_tag])
    b.add_beam_at_points(
        point['A']['1'],
        point['B']['1'],
        snap_i='center_right',
        snap_j='top_center',
        ends=grav_bm_ends)
    b.add_beam_at_points(
        point['E']['1'],
        point['F']['1'],
        snap_i='bottom_center',
        snap_j='center_left',
        ends=grav_bm_ends)
    b.add_beam_at_points(
        point['A']['5'],
        point['B']['5'],
        snap_i='center_right',
        snap_j='top_center',
        ends=grav_bm_ends)
    b.add_beam_at_points(
        point['E']['5'],
        point['F']['5'],
        snap_i='bottom_center',
        snap_j='center_left',
        ends=grav_bm_ends)

    # define gravity beams of designation C
    b.set_active_section(sections['gravity_beams_c'][level_tag])
    for tag1 in ['2', '3', '4']:
        b.add_beam_at_points(
            point['A'][tag1],
            point['B'][tag1],
            snap_i='center_right',
            snap_j='center_left',
            ends=grav_bm_ends)
        b.add_beam_at_points(
            point['E'][tag1],
            point['F'][tag1],
            snap_i='center_right',
            snap_j='center_left',
            ends=grav_bm_ends)

    # define gravity beams of designation D
    b.set_active_section(sections['gravity_beams_d'][level_tag])
    for tag1 in ['2', '3', '4']:
        tag2_start = ['B', 'C', 'D']
        tag2_end = ['C', 'D', 'E']
        for j in range(len(tag2_start)):
            b.add_beam_at_points(
                point[tag2_start[j]][tag1],
                point[tag2_end[j]][tag1],
                snap_i='center_right',
                snap_j='center_left',
                ends=grav_bm_ends)

    # define gravity beams of designation e
    b.set_active_section(sections['gravity_beams_e'][level_tag])
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
b.assign_surface_load((63.+15.+15.)/(12.**2))
b.assign_surface_load_massless((0.25 * 50)/(12.**2))

b.set_active_levels(['3'])
b.assign_surface_load((63.+15.+80.*0.26786)/(12.**2))
b.assign_surface_load_massless((0.25 * 20)/(12.**2))

# cladding - 1st story
b.select_perimeter_beams_story('1')
# 15 is the load in lb/ft2, we multiply it by the height
# the tributary area of the 1st story cladding support is
# half the height of the 1st story and half the height of the second
# we get lb/ft, so we divide by 12 to convert this to lb/in
# which is what OpenSeesPy_Building_Modeler uses.
b.selection.add_UDL(np.array((0.00, 0.00,
                              -((15./12.**2) * (hi[0] + hi[1]) / 2.00))))

# cladding - 2nd story
b.selection.clear()
b.select_perimeter_beams_story('2')
b.selection.add_UDL(np.array((0.00, 0.00,
                              -((15./12.**2) * (hi[1] + hi[2]) / 2.00))))

# cladding - roof
b.selection.clear()
b.select_perimeter_beams_story('3')
b.selection.add_UDL(np.array((0.00, 0.00,
                              -((15./12.**2) * hi[2] / 2.00))))
b.selection.clear()

b.preprocess(assume_floor_slabs=True, self_weight=True,
             steel_panel_zones=True, elevate_column_splices=0.25)

# b.plot_building_geometry(extrude_frames=True)
# b.plot_building_geometry(extrude_frames=False, frame_axes=False)


# modal_analysis = solver.ModalAnalysis(b, num_modes=6)
# modal_analysis.run()

# modal_analysis.deformed_shape(step=0, scaling=0.00, extrude_frames=True)



# retrieve some info used in the for loops
num_levels = len(b.levels.registry) - 1
level_heights = []
for level in b.levels.registry.values():
    level_heights.append(level.elevation)
level_heights = np.diff(level_heights)


base_node = list(b.levels.registry['base'].nodes_primary.registry.values())[0].uid
lvl1_node = b.levels.registry['1'].parent_node.uid
lvl2_node = b.levels.registry['2'].parent_node.uid
lvl3_node = b.levels.registry['3'].parent_node.uid

# define analysis object
nlth = solver.NLTHAnalysis(
    b,
    output_directory=f'{output_folder}',
    disk_storage=False,
    store_fiber=False,
    store_forces=False,
    store_reactions=False,
    store_release_force_defo=False,
    specific_nodes=[base_node, lvl1_node,
                    lvl2_node, lvl3_node])

# get the corresponding ground motion duration
gm_X_filepath = ground_motion_dir + '/' + str(gm_number) + 'x.txt'
gm_Y_filepath = ground_motion_dir + '/' + str(gm_number) + 'y.txt'
gm_Z_filepath = ground_motion_dir + '/' + str(gm_number) + 'z.txt'

dx = get_duration(gm_X_filepath, ground_motion_dt)
dy = get_duration(gm_Y_filepath, ground_motion_dt)
dz = get_duration(gm_Z_filepath, ground_motion_dt)
duration = np.min(np.array((dx, dy, dz)))  # note: actually should be =


damping = {'type': 'rayleigh',
           'ratio': 0.03,
           'periods': [0.82, 0.12]}
# damping = {'type': 'modal',
#            'num_modes': 50,
#            'ratio': 0.03}

# run the nlth analysis
metadata = nlth.run(analysis_dt,
                    ground_motion_dir + '/' + str(gm_number) + 'x.txt',
                    ground_motion_dir + '/' + str(gm_number) + 'y.txt',
                    ground_motion_dir + '/' + str(gm_number) + 'z.txt',
                    ground_motion_dt,
                    finish_time=0.00,
                    damping=damping,
                    printing=False)

if not metadata['analysis_finished_successfully']:
    print(f'Analysis failed due to convergence issues. '
          f'| {ground_motion_dir}: {gm_number}')
    sys.exit()

# # reopen shelves
# nlth = solver.NLTHAnalysis(
#     b, output_directory=f'{output_folder}',
#     disk_storage=True,
#     store_fiber=False,
#     store_forces=False,
#     store_reactions=False)
# nlth.read_results()

time_vec = np.array(nlth.time_vector)
resp_a = {}
resp_v = {}
resp_u = {}
resp_a[0] = nlth.retrieve_node_abs_acceleration(base_node)
resp_a[1] = nlth.retrieve_node_abs_acceleration(lvl1_node)
resp_a[2] = nlth.retrieve_node_abs_acceleration(lvl2_node)
resp_a[3] = nlth.retrieve_node_abs_acceleration(lvl3_node)
resp_v[0] = nlth.retrieve_node_abs_velocity(base_node)
resp_v[1] = nlth.retrieve_node_abs_velocity(lvl1_node)
resp_v[2] = nlth.retrieve_node_abs_velocity(lvl2_node)
resp_v[3] = nlth.retrieve_node_abs_velocity(lvl3_node)
# resp_u[0] = nlth.retrieve_node_displacement(base_node)  # always 0 - fixed
resp_u[1] = nlth.retrieve_node_displacement(lvl1_node)
resp_u[2] = nlth.retrieve_node_displacement(lvl2_node)
resp_u[3] = nlth.retrieve_node_displacement(lvl3_node)


# ~~~~~~~~~~~~~~~~ #
# collect response #
# ~~~~~~~~~~~~~~~~ #


# ground acceleration, velocity and displacement
# interpolation functions

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

time_vec = np.array(nlth.time_vector)
np.savetxt(f'{output_folder}/time.csv',
           time_vec)
num_levels = len(level_heights)

for direction in range(2):
    # store response time-histories
    # ground acceleration
    np.savetxt(f'{output_folder}/FA-0-{direction+1}.csv',
               resp_a[0][:, direction])
    # ground velocity
    np.savetxt(f'{output_folder}/FV-0-{direction+1}.csv',
               resp_v[0][:, direction])
    for lvl in range(num_levels):
        # story drifts
        if lvl == 0:
            u = resp_u[1][:, direction]
            dr = u / level_heights[lvl]
        else:
            uprev = resp_u[lvl][:, direction]
            u = resp_u[lvl + 1][:, direction]
            dr = (u - uprev) / level_heights[lvl]
        # story accelerations
        a = resp_a[lvl + 1][:, direction]
        # story velocities
        v = resp_v[lvl + 1][:, direction]

        np.savetxt(f'{output_folder}/ID-{lvl+1}-{direction+1}.csv', dr)
        np.savetxt(f'{output_folder}/FA-{lvl+1}-{direction+1}.csv', a)
        np.savetxt(f'{output_folder}/FV-{lvl+1}-{direction+1}.csv', v)

    # global building drift
    bdr = resp_u[3][:, direction]
    bdr /= np.sum(level_heights)
    np.savetxt(f'{output_folder}/BD-{direction+1}.csv', bdr)
