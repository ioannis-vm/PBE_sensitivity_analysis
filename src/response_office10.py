"""
Nonlinear time-history analysis of a predefined building object
using a series of ground motion records
of varying levels of intensity to create a series of
peak response quantities for these levels of intensity,
suitable for subsequent use in PELICUN.

"""

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

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

# parser = argparse.ArgumentParser()
# parser.add_argument('--gm_dir')
# parser.add_argument('--gm_dt')
# parser.add_argument('--analysis_dt')
# parser.add_argument('--gm_number')
# parser.add_argument('--output_dir')

# args = parser.parse_args()

# ground_motion_dir = args.gm_dir  # 'ground_motions/test_case/parsed'
# ground_motion_dt = float(args.gm_dt)  # 0.005
# analysis_dt = float(args.analysis_dt)  # 0.05
# gm_number = int(args.gm_number.replace('gm', ''))
# output_folder = args.output_dir  # 'response/test_case'

# debug
ground_motion_dir = 'analysis/hazard_level_8/ground_motions/parsed'
ground_motion_dt = 0.005
analysis_dt = 0.001
gm_number = 1
output_folder = 'analysis/hazard_level_8/response/gm1'


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

hi = np.array([15.00]+[13.00]*9) * 12.00  # in

h_lvl = np.zeros(10)
h_lvl[0] = hi[0]
for i in range(1, 10):
    h_lvl[i] = h_lvl[i-1] + hi[i]

# Add levels
b.add_level("base", 0.00, "fixed")
for i in range(10):
    b.add_level(str(i+1), h_lvl[i])
sections = dict(
    gravity_cols=dict(
        level_1="W14X90",
        level_2="W14X90",
        level_3="W14X90",
        level_4="W14X90",
        level_5="W14X90",
        level_6="W14X90",
        level_7="W14X90",
        level_8="W14X90",
        level_9="W14X90",
        level_10="W14X90"),
    gravity_beams_perimeter=dict(
        level_1="W21X55",
        level_2="W21X55",
        level_3="W21X55",
        level_4="W21X55",
        level_5="W21X55",
        level_6="W21X55",
        level_7="W21X55",
        level_8="W21X55",
        level_9="W21X55",
        level_10="W21X55"),
    gravity_beams_interior_32=dict(
        level_1="W12X152",
        level_2="W12X152",
        level_3="W12X152",
        level_4="W12X152",
        level_5="W12X152",
        level_6="W12X152",
        level_7="W12X152",
        level_8="W12X152",
        level_9="W12X152",
        level_10="W12X152"),
    gravity_beams_interior_25=dict(
        level_1="W10X100",
        level_2="W10X100",
        level_3="W10X100",
        level_4="W10X100",
        level_5="W10X100",
        level_6="W10X100",
        level_7="W10X100",
        level_8="W10X100",
        level_9="W10X100",
        level_10="W10X100"),
    secondary_beams="W14X30",
    lateral_cols=dict(
        level_1="W14X665",
        level_2="W14X665",
        level_3="W14X665",
        level_4="W14X605",
        level_5="W14X550",
        level_6="W14X550",
        level_7="W14X500",
        level_8="W14X455",
        level_9="W14X398",
        level_10="W14X283"),
    lateral_beams=dict(
        level_1="W27X307",
        level_2="W27X307",
        level_3="W27X307",
        level_4="W27X258",
        level_5="W27X258",
        level_6="W24X250",
        level_7="W24X250",
        level_8="W24X207",
        level_9="W24X162",
        level_10="W18X130")
    )



# define materials
b.set_active_material('steel02-fy50')

# define sections
wsections = set()
hsssections = set()
for lvl_tag in [f'level_{i+1}' for i in range(10)]:
    wsections.add(sections['gravity_beams_perimeter'][lvl_tag])
    wsections.add(sections['gravity_beams_interior_32'][lvl_tag])
    wsections.add(sections['gravity_beams_interior_25'][lvl_tag])
    wsections.add(sections['lateral_cols'][lvl_tag])
    wsections.add(sections['lateral_beams'][lvl_tag])
    wsections.add(sections['gravity_cols'][lvl_tag])
wsections.add(sections['secondary_beams'])


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

lat_bm_ends = {'type': 'steel_W_IMK', 'end_dist': 0.05,
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

for level_counter in range(10):
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
    b.set_active_section(sections['lateral_cols'][level_tag])
    b.set_active_angle(np.pi/2.00)
    for tag1 in ['B', 'C', 'D', 'E']:
        for tag2 in ['1', '5']:
            pt = point[tag1][tag2]
            b.add_column_at_point(
                pt, n_sub=nsub, ends=lat_col_ends,
                model_as=lat_col_modeling_type, geom_transf=col_gtransf)
    # deffine Y-dir frame columns
    b.set_active_angle(0.00)
    for tag1 in ['A', 'F']:
        for tag2 in ['5', '4', '3', '2']:
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
    # define perimeter gravity beams
    b.set_active_section(sections['gravity_beams_perimeter'][level_tag])
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
    b.add_beam_at_points(
        point['A']['1'],
        point['B']['1'],
        snap_j='top_center',
        ends=grav_bm_ends)
    b.add_beam_at_points(
        point['E']['1'],
        point['F']['1'],
        snap_i='bottom_center',
        ends=grav_bm_ends)
    b.add_beam_at_points(
        point['A']['5'],
        point['B']['5'],
        snap_j='top_center',
        ends=grav_bm_ends)
    b.add_beam_at_points(
        point['E']['5'],
        point['F']['5'],
        snap_i='bottom_center',
        ends=grav_bm_ends)
    # define interior gravity beams
    for tag1 in ['B', 'C', 'D', 'E']:
        b.set_active_section(
            sections['gravity_beams_interior_25'][level_tag])
        tag2_start = ['2', '3']
        tag2_end = ['3', '4']
        for j in range(len(tag2_start)):
            b.add_beam_at_points(
                point[tag1][tag2_start[j]],
                point[tag1][tag2_end[j]],
                snap_i='bottom_center',
                snap_j='top_center',
                ends=grav_bm_ends)
        tag2_start = ['1']
        tag2_end = ['2']
        for j in range(len(tag2_start)):
            b.add_beam_at_points(
                point[tag1][tag2_start[j]],
                point[tag1][tag2_end[j]],
                snap_j='top_center',
                ends=grav_bm_ends)
        tag2_start = ['4']
        tag2_end = ['5']
        for j in range(len(tag2_start)):
            b.add_beam_at_points(
                point[tag1][tag2_start[j]],
                point[tag1][tag2_end[j]],
                snap_i='bottom_center',
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
                    offset_i=np.array([0., 0., -10.]),
                    offset_j=np.array([0., 0., -10.]),
                    ends=pinned_ends)

































































# # generate a dictionary containing coordinates given gridline tag names
# # (here we won't use the native gridline objects,
# #  since the geometry is very simple)
# point = {}
# x_grd_tags = ['A', 'B', 'C', 'D', 'E', 'F']
# y_grd_tags = ['5', '4', '3', '2', '1']
# x_grd_locs = np.array([0.00, 32.5, 57.5, 82.5, 107.5, 140.00]) * 12.00  # (in)
# y_grd_locs = np.array([0.00, 25.00, 50.00, 75.00, 100.00]) * 12.00  # (in)


# for i in range(len(x_grd_tags)):
#     point[x_grd_tags[i]] = {}
#     for j in range(len(y_grd_tags)):
#         point[x_grd_tags[i]][y_grd_tags[j]] = \
#             np.array([x_grd_locs[i], y_grd_locs[j]])

# for level_counter in range(3):
#     level_tag = 'level_'+str(level_counter+1)
#     # define gravity columns
#     b.set_active_angle(0.00)
#     b.set_active_placement('centroid')
#     b.set_active_levels([str(level_counter+1)])
#     b.set_active_section(sections['gravity_cols'][level_tag])
#     for tag in ['A', 'F']:
#         pt = point[tag]['1']
#         col = b.add_column_at_point(
#             pt, n_sub=1, ends=grav_col_ends,
#             model_as=elastic_modeling_type, geom_transf=col_gtransf)
#     for tag1 in ['B', 'C', 'D', 'E']:
#         for tag2 in ['2', '3', '4']:
#             pt = point[tag1][tag2]
#             col = b.add_column_at_point(
#                 pt, n_sub=1, ends=grav_col_ends,
#                 model_as=elastic_modeling_type, geom_transf=col_gtransf)

#     # define X-dir frame columns
#     b.set_active_section(sections['lateral_cols'][level_tag])
#     b.set_active_angle(np.pi/2.00)
#     for tag1 in ['B', 'C', 'D', 'E']:
#         for tag2 in ['1', '5']:
#             pt = point[tag1][tag2]
#             b.add_column_at_point(
#                 pt, n_sub=nsub, ends=lat_col_ends,
#                 model_as=lat_col_modeling_type, geom_transf=col_gtransf)
#     # deffine Y-dir frame columns
#     b.set_active_angle(0.00)
#     for tag1 in ['A', 'F']:
#         for tag2 in ['5', '4', '3', '2']:
#             pt = point[tag1][tag2]
#             b.add_column_at_point(
#                 pt, n_sub=nsub, ends=lat_col_ends,
#                 model_as=lat_col_modeling_type, geom_transf=col_gtransf)
#     # define X-dir frame beams
#     b.set_active_section(sections['lateral_beams'][level_tag])
#     b.set_active_placement('top_center')
#     for tag1 in ['1', '5']:
#         tag2_start = ['B', 'C', 'D']
#         tag2_end = ['C', 'D', 'E']
#         for j in range(len(tag2_start)):
#             b.add_beam_at_points(
#                 point[tag2_start[j]][tag1],
#                 point[tag2_end[j]][tag1],
#                 ends=lat_bm_ends,
#                 model_as=lat_bm_modeling_type, n_sub=nsub,
#                 snap_i='bottom_center',
#                 snap_j='top_center')
#     # define Y-dir frame beams
#     for tag1 in ['A', 'F']:
#         tag2_start = ['2', '3', '4']
#         tag2_end = ['3', '4', '5']
#         for j in range(len(tag2_start)):
#             b.add_beam_at_points(
#                 point[tag1][tag2_start[j]],
#                 point[tag1][tag2_end[j]],
#                 ends=lat_bm_ends,
#                 model_as=lat_bm_modeling_type, n_sub=nsub,
#                 snap_i='bottom_center',
#                 snap_j='top_center')
#     # define perimeter gravity beams
#     b.set_active_section(sections['gravity_beams_perimeter'][level_tag])
#     for tag1 in ['A', 'F']:
#         tag2_start = ['1']
#         tag2_end = ['2']
#         for j in range(len(tag2_start)):
#             b.add_beam_at_points(
#                 point[tag1][tag2_start[j]],
#                 point[tag1][tag2_end[j]],
#                 ends=grav_bm_ends,
#                 snap_i='bottom_center',
#                 snap_j='top_center')
#     b.add_beam_at_points(
#         point['A']['1'],
#         point['B']['1'],
#         snap_j='top_center',
#         ends=grav_bm_ends)
#     b.add_beam_at_points(
#         point['E']['1'],
#         point['F']['1'],
#         snap_i='bottom_center',
#         ends=grav_bm_ends)
#     b.add_beam_at_points(
#         point['A']['5'],
#         point['B']['5'],
#         snap_j='top_center',
#         ends=grav_bm_ends)
#     b.add_beam_at_points(
#         point['E']['5'],
#         point['F']['5'],
#         snap_i='bottom_center',
#         ends=grav_bm_ends)
#     # define interior gravity beams
#     for tag1 in ['B', 'C', 'D', 'E']:
#         b.set_active_section(
#             sections['gravity_beams_interior_25'][level_tag])
#         tag2_start = ['2', '3']
#         tag2_end = ['3', '4']
#         for j in range(len(tag2_start)):
#             b.add_beam_at_points(
#                 point[tag1][tag2_start[j]],
#                 point[tag1][tag2_end[j]],
#                 snap_i='bottom_center',
#                 snap_j='top_center',
#                 ends=grav_bm_ends)
#         tag2_start = ['1']
#         tag2_end = ['2']
#         for j in range(len(tag2_start)):
#             b.add_beam_at_points(
#                 point[tag1][tag2_start[j]],
#                 point[tag1][tag2_end[j]],
#                 snap_j='top_center',
#                 ends=grav_bm_ends)
#         tag2_start = ['4']
#         tag2_end = ['5']
#         for j in range(len(tag2_start)):
#             b.add_beam_at_points(
#                 point[tag1][tag2_start[j]],
#                 point[tag1][tag2_end[j]],
#                 snap_i='bottom_center',
#                 ends=grav_bm_ends)
#     for tag1 in ['2', '3', '4']:
#         tag2_start = ['A', 'B', 'C', 'D', 'E']
#         tag2_end = ['B', 'C', 'D', 'E', 'F']
#         for j in range(len(tag2_start)):
#             if tag2_start[j] in ['B', 'E']:
#                 b.set_active_section(
#                     sections['gravity_beams_interior_32'][level_tag])
#             else:
#                 b.set_active_section(
#                     sections['gravity_beams_interior_25'][level_tag])
#             b.add_beam_at_points(
#                 point[tag2_start[j]][tag1],
#                 point[tag2_end[j]][tag1],
#                 ends=grav_bm_ends)
#     # define secondary beams
#     b.set_active_section(sections['secondary_beams'])
#     for tag1 in ['A', 'B', 'C', 'D', 'E']:
#         tag2_start = ['1', '2', '3', '4']
#         tag2_end = ['2', '3', '4', '5']
#         if tag1 in ['A', 'E']:
#             shifts = 32.5/4. * 12.  # in
#             num = 3
#         else:
#             shifts = 25.0/3. * 12  # in
#             num = 2
#         shift = 0.00
#         for i in range(num):
#             shift += shifts
#             for j in range(len(tag2_start)):
#                 b.add_beam_at_points(
#                     point[tag1][tag2_start[j]] + np.array([shift, 0.00]),
#                     point[tag1][tag2_end[j]] + np.array([shift, 0.00]),
#                     offset_i=np.array([0., 0., -10.]),
#                     offset_j=np.array([0., 0., -10.]),
#                     ends=pinned_ends)





#
# define surface loads
#


b.set_active_levels([f'{i+1}' for i in range(9)])
b.assign_surface_DL((75.+15.+20.+0.25*80.)/(12.**2))

b.set_active_levels(['10'])
b.assign_surface_DL((75.+15.+80.+0.25*20)/(12.**2))


# cladding - 1st story
b.select_perimeter_beams_story('1')
# 10 is the load in lb/ft2, we multiply it by the height
# the tributary area of the 1st story cladding support is
# half the height of the 1st story and half the height of the second
# we get lb/ft, so we divide by 12 to convert this to lb/in
b.selection.add_UDL(np.array((0.00, 0.00,
                              -((10./12.**2) * (hi[0] + hi[1]) / 2.00))))

# cladding - 2nd story, up to 9th story
for j in ['2', '3', '4', '5', '6', '7', '8', '9']:
    b.selection.clear()
    b.select_perimeter_beams_story(j)
    b.selection.add_UDL(np.array((0.00, 0.00,
                                  -((10./12.**2) * (hi[1] + hi[2]) / 2.00))))

# cladding - roof
b.selection.clear()
b.select_perimeter_beams_story('10')
b.selection.add_UDL(np.array((0.00, 0.00,
                              -((10./12.**2) * hi[2] / 2.00))))
b.selection.clear()


b.preprocess(assume_floor_slabs=True, self_weight=True,
             steel_panel_zones=True, elevate_column_splices=0.25)

b.plot_building_geometry(extrude_frames=True)
b.plot_building_geometry(extrude_frames=False, frame_axes=False)


# num_modes = 6
# modal_analysis = solver.ModalAnalysis(b, num_modes=num_modes)
# modal_analysis.run()
# ts = modal_analysis.periods




# retrieve some info used in the for loops
num_levels = len(b.levels.registry) - 1
level_heights = []
for level in b.levels.registry.values():
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


damping = {'type': 'rayleigh',
           'ratio': 0.03,
           'periods': [2.20, 0.40]}
# damping = {'type': 'modal',
#            'num_modes': 50,
#            'ratio': 0.03}

# run the nlth analysis
metadata = nlth.run(analysis_dt,
                    ground_motion_dir + '/' + str(gm_number) + 'x.txt',
                    ground_motion_dir + '/' + str(gm_number) + 'y.txt',
                    ground_motion_dir + '/' + str(gm_number) + 'z.txt',
                    ground_motion_dt,
                    finish_time=4.00,
                    damping=damping,
                    printing=True)


# plot_ground_motion(ground_motion_dir + '/' + str(gm_number) + 'x.txt', ground_motion_dt)


if not metadata['analysis_finished_successfully']:
    print('Analysis failed.')
    # print(args)
    sys.exit()


base_node = list(b.levels.registry['base'].nodes_primary.registry.values())[0].uid
lvl1_node = b.levels.registry['1'].parent_node.uid
lvl2_node = b.levels.registry['2'].parent_node.uid
lvl3_node = b.levels.registry['3'].parent_node.uid

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

