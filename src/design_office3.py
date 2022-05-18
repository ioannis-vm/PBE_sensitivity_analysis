import sys
sys.path.append("../OpenSees_Model_Builder/src")

import numpy as np
import model
import solver
import time
import pickle
from scipy.interpolate import interp1d
import argparse
import pandas as pd

# ~~~~~~~~~~~~~~~~~~~~ #
# function definitions #
# ~~~~~~~~~~~~~~~~~~~~ #

def k(T):
    if T <= 0.5:
        res = 1.0
    elif T >= 2.5:
        res = 2.0
    else:
        x = np.array([0.5, 2.5])
        y = np.array([1., 2.])
        f = interp1d(x, y)
        res = f(np.array([T]))[0]
    return res

def Tmax(ct, exponent, height, Sd1):
    def cu(Sd1):
        if Sd1 <= 0.1:
            cu = 1.7
        elif Sd1 >= 0.4:
            cu = 1.4
        else:
            x = np.array([0.1, 0.15, 0.2, 0.3, 0.4])
            y = np.array([1.7, 1.6, 1.5, 1.4, 1.4])
            f = interp1d(x, y)
            cu = f(np.array([Sd1]))[0]
        return cu

    Ta = ct * height**exponent
    return cu(Sd1) * Ta

def get_floor_displacements(building, fxx, direction):

    parent_nodes = building.list_of_parent_nodes()

    if direction == 0:
        for i, node in enumerate(parent_nodes):
            node.load = np.array([fxx[i]*1e3, 0.00, 0.00, 0.00, 0.00, 0.00])
    elif direction == 1:
        for i, node in enumerate(parent_nodes):
            node.load = np.array([0.00, fxx[i]*1e3, 0.00, 0.00, 0.00, 0.00])
    else:
        raise ValueError('Invalid Direction')

    linear_gravity_analysis = solver.LinearGravityAnalysis(b)
    linear_gravity_analysis.run()

    u1_el = linear_gravity_analysis.node_displacements[
        str(parent_nodes[0].uid)][0][direction]
    u2_el = linear_gravity_analysis.node_displacements[
        str(parent_nodes[1].uid)][0][direction]
    u3_el = linear_gravity_analysis.node_displacements[
        str(parent_nodes[2].uid)][0][direction]

    return np.array([u1_el, u2_el, u3_el])

def cs(T, Sds, Sd1, R, Ie):
    Tshort = Sd1/Sds
    if T < Tshort:
        res = Sds / R * Ie
    else:
        res = Sd1 / R * Ie / T
    return res


# ~~~~ #
# main #
# ~~~~ #


# Define a building
b = model.Model()

hi = np.array([15.00, 13.00, 13.00]) * 12.00  # in

# Add levels
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

# start from 0.00, determine requirement, provide requirement & check
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

elastic_modeling_type = {'type': 'elastic'}
lat_col_ends = {'type': 'steel_W_PZ_IMK', 'end_dist': 0.05,
                'Lb/ry': 60., 'L/H': 1.0, 'pgpye': 0.005,
                'doubler plate thickness': 0.00}
RBS_ends_a = {'type': 'RBS', 'end_dist': (6.00+20.0)/(25.*12.-5.),
              'rbs_length': 17.5, 'rbs_reduction': 0.60, 'rbs_n_sub': 10}
RBS_ends_b = {'type': 'RBS', 'end_dist': (4.00+20.00)/(25.*12.-5.),
              'rbs_length': 17.5, 'rbs_reduction': 0.60, 'rbs_n_sub': 10}
fiber_modeling_type = {'type': 'fiber', 'n_x': 10, 'n_y': 25}
pinned_ends = {'type': 'pinned', 'end_dist': 0.001}
fixedpinned_ends = {'type': 'fixed-pinned', 'end_dist': 0.005,
                    'doubler plate thickness': 0.00}
grav_col_ends = fixedpinned_ends
col_gtransf = 'Corotational'
gtransf = 'Corotational'
lat_col_modeling_type=elastic_modeling_type
lat_bm_modeling_type=elastic_modeling_type
grav_bm_ends=pinned_ends

nsub = 10  # element subdivision

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
    if level_counter in [0, 1]:
        lat_bm_ends = RBS_ends_a
    else:
        lat_bm_ends = RBS_ends_b
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


b.set_active_levels(['1', '2'])
b.assign_surface_load((63.+15.+15.)/(12.**2))

b.set_active_levels(['3'])
b.assign_surface_load((63.+15.+80.*0.26786)/(12.**2))


# cladding - 1st story
b.select_perimeter_beams_story('1')
# 10 is the load in lb/ft2, we multiply it by the height
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

cam = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0.00),
    eye=dict(x=0.00, y=2.25, z=0.00),
    projection={
        "type": "perspective"
    }
)

b.plot_building_geometry(
    extrude_frames=True,
    frame_axes=False,
    global_axes=False,
    tributary_areas=False,
    diaphragm_lines=True,
    camera=cam
)

b.plot_building_geometry(
    extrude_frames=False,
    frame_axes=False,
    global_axes=False,
    tributary_areas=False,
    diaphragm_lines=True,
    camera=cam
)

linear_gravity_analysis = solver.LinearGravityAnalysis(b)
linear_gravity_analysis.run()

# store column axial loads for design checks
col_uids = dict(
    exterior=dict(level_1='3907',
                  level_2='4489',
                  level_3='5071'),
    interior=dict(level_1='4007',
                  level_2='4589',
                  level_3='5171'))
col_puc = dict(exterior={}, interior={})
for key1 in col_uids.keys():
    for key2 in col_uids[key1].keys():
        col_puc[key1][key2] = linear_gravity_analysis.element_forces[
            col_uids[key1][key2]][0][8] / 1000.  # kips
# store beam gravity UDL for design checks
# kips/in
we = dict(
    level_1=-b.dct_line_elements[319].udl_total()[1] / 1000.,
    level_2=-b.dct_line_elements[1544].udl_total()[1] / 1000.,
    level_3=-b.dct_line_elements[2767].udl_total()[1] / 1000.
)


# strong column - weak beam
sh = 26.25  # in
ext_res = []
int_res = []
lvl_tags = list(sections['lateral_beams'].keys())
for place in ['exterior', 'interior']:
    for lvl_num in range(len(lvl_tags)-1):
        this_lvl = lvl_tags[lvl_num]
        level_above = lvl_tags[lvl_num + 1]
        beam_sec = b.sections.registry[sections['lateral_beams'][this_lvl]].properties
        # in3
        zc_below = b.sections.registry[sections['lateral_cols'][place][this_lvl]].properties['Zx']
        # in2
        ac_below = b.sections.registry[sections['lateral_cols'][place][this_lvl]].properties['A']
        # in3
        zc_above = b.sections.registry[sections['lateral_cols'][place][level_above]].properties['Zx']
        # in2
        ac_above = b.sections.registry[sections['lateral_cols'][place][level_above]].properties['A']
        # kip-in
        mc_below = zc_below * (50.00 - col_puc[place][this_lvl]/ac_below)
        # kip-in
        mc_above = zc_above * (50.00 - col_puc[place][this_lvl]/ac_above)
        # kips
        vc_star = (mc_below + mc_above) / ((hi[lvl_num] - beam_sec['d']))
        # kip-in
        sigma_mc_star = (mc_below + mc_above) + vc_star * (beam_sec['d'])/2.
        # in
        c_rbs = beam_sec['bf'] * (1. - 0.60) / 2.
        # in3
        z_rbs = beam_sec['Zx'] - 2. * c_rbs * beam_sec['tf'] * (beam_sec['d'] - beam_sec['tf'])
        # kip-in
        m_pr = 1.15 * 1.10 * 50 * z_rbs
        # kip
        v_e = (2 * m_pr) / (25.*12. - 2. * sh)
        # kip
        v_g = we[this_lvl] * (25.*12. - 2. * sh) / 2.
        dc = b.sections.registry[sections['lateral_cols'][place][this_lvl]].properties['d']
        if place == 'exterior':
            sigm_mb_star = 1.00 * (m_pr + v_e * (sh + dc/2.) + v_g*(sh + dc/2.))
            ext_res.append(sigma_mc_star / sigm_mb_star)
        if place == 'interior':
            sigm_mb_star = 2.00 * (m_pr + v_e * (sh + dc/2.))
            int_res.append(sigma_mc_star / sigm_mb_star)
scwb_check = pd.DataFrame({'exterior': ext_res, 'interior': int_res}, index=lvl_tags[:-1])
print()
print('Strong Column - Weak Beam Check')
print(scwb_check)
print()


# doubler plate requirement check
ext_res = []
int_res = []
ext_doubler_thickness = []
int_doubler_thickness = []
lvl_tags = list(sections['lateral_beams'].keys())
for place in ['exterior', 'interior']:
    for lvl_num in range(len(lvl_tags)):
        this_lvl = lvl_tags[lvl_num]
        beam_sec = b.sections.registry[sections['lateral_beams'][this_lvl]].properties
        col_sec = b.sections.registry[sections['lateral_cols'][place][this_lvl]].properties
        # assert(col_sec['d'] / col_sec['tw'] < 35.09), 'Error: Column not seismically compact'
        # in
        c_rbs = beam_sec['bf'] * (1. - 0.60) / 2.
        # in3
        z_rbs = beam_sec['Zx'] - 2. * c_rbs * beam_sec['tf'] * (beam_sec['d'] - beam_sec['tf'])
        # kip-in
        m_pr = 1.15 * 1.10 * 50 * z_rbs
        # kip
        ve = 2. * m_pr / (25.*12. - 2. * sh)
        # kip-in
        m_f = m_pr + ve * sh
        # kips
        r_n = 0.60 * 50 * col_sec['d'] * col_sec['tw'] * (1.00 + (3. * col_sec['bf'] * (col_sec['tf'])**2) / (col_sec['d'] * beam_sec['d'] * col_sec['tw']))
        if place == 'interior':
            r_u = 2 * m_f / (beam_sec['d'] - beam_sec['tf'])
            int_res.append(r_u/r_n)
            tdoub = (r_u-r_n) / (0.60 * 50. * col_sec['d'])
            tdoub = max(tdoub, 0.00)
            int_doubler_thickness.append(tdoub)
        else:
            r_u = m_f / (beam_sec['d'] - beam_sec['tf'])
            ext_res.append(r_u/r_n)
            tdoub = (r_u-r_n) / (0.60 * 50. * col_sec['d'])
            tdoub = max(tdoub, 0.00)
            ext_doubler_thickness.append(tdoub)

pz_check = pd.DataFrame({'exterior': ext_res, 'interior': int_res}, index=lvl_tags)
# print()
# print('Doubler Plate Requirement Check')
# print(pz_check)
# print()

doubler_thickness = pd.DataFrame({'exterior': ext_doubler_thickness, 'interior': int_doubler_thickness}, index=lvl_tags)
print()
print('Required Doubler Plate Thickness')
print(doubler_thickness)
print()





print()
print("~~~ Seismic Weight ~~~")
for m in b.level_masses():
    print('%.2f kips' % (m / 1.0e3 * 386.22))
print()

# b.plot_building_geometry(extrude_frames=True,
#                          offsets=True,
#                          gridlines=True,
#                          global_axes=False,
#                          diaphragm_lines=True,
#                          tributary_areas=True,
#                          just_selection=False,
#                          parent_nodes=True,
#                          frame_axes=False)

# b.plot_building_geometry(extrude_frames=False,
#                          offsets=True,
#                          gridlines=True,
#                          global_axes=False,
#                          diaphragm_lines=True,
#                          tributary_areas=True,
#                          just_selection=False,
#                          parent_nodes=True,
#                          frame_axes=False)


p_nodes = b.list_of_parent_nodes()

Cd = 5.5
R = 8.0
Ie = 1.0

Sds = 1.58
Sd1 = 1.38
Tshort = Sd1/Sds

# multi-period design spectrum
mlp_periods = np.array(
    (0.00, 0.01, 0.02, 0.03, 0.05,
     0.075, 0.1, 0.15, 0.2, 0.25,
     0.3, 0.4, 0.5, 0.75, 1.,
     1.5, 2., 3., 4., 5., 7.5, 10.))
mlp_des_spc = np.array(
    (0.66, 0.66, 0.66, 0.67, 0.74,
     0.90, 1.03, 1.22, 1.36, 1.48,
     1.62, 1.75, 1.73, 1.51, 1.32,
     0.98, 0.77, 0.51, 0.35, 0.26,
     0.14, 0.083))
design_spectrum_ifun = interp1d(mlp_periods, mlp_des_spc, kind='linear')


# import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# plt.figure(figsize=(2.8, 2.8))
# plt.plot(mlp_periods, mlp_des_spc, 'k')
# plt.xlabel('T (s)')
# plt.ylabel('Sa (g)')
# plt.xscale('log')
# plt.subplots_adjust(bottom=0.17, left=0.22, top=0.98, right=0.97)
# plt.savefig('/home/john_vm/google_drive_encr/UCB/research/projects/299_report/299_report/images/design_spec.pdf')
# plt.close()


# period estimation (Table 12.8-2)

ct = 0.028
exponent = 0.8

wi = np.zeros(3)
hi_add = np.zeros(3)
for i in range(3):
    wi[i] = p_nodes[i].mass[0]*386.22/1000
    hi_add[i] = p_nodes[i].coords[2]/12.

T_max = Tmax(ct, exponent, np.max(hi_add), Sd1)

print('T_max = %.2f s\n' % (T_max))

# Note: For drift checks we don't use Tmax. We use the calculated
# period instead.


#
# x direction
#

# run modal analysis
# (must be done here to get the period)

num_modes = 9
modal_analysis = solver.ModalAnalysis(b, num_modes=num_modes)
modal_analysis.run()
gammasX, mstarsX, mtot = modal_analysis.modal_participation_factors('x')
gammasY, mstarsY, mtot = modal_analysis.modal_participation_factors('y')
ts = modal_analysis.periods

print(pd.DataFrame({'periods': [f'{t:.2f}' for t in ts],
                    'mustarX': [f'{m:.3f}' for m in mstarsX],
                    'mustarY': [f'{m:.3f}' for m in mstarsY]}).to_latex())

# # visualization
# eye_vec = np.array((-2.25, 2.25, 1.50))/1.7
# cam = dict(
#     up=dict(x=0, y=0, z=1),
#     center=dict(x=0, y=0, z=-0.30),
#     eye=dict(x=eye_vec[0], y=eye_vec[1], z=eye_vec[2]),
#     projection={
#         "type": "perspective"
#     }
# )
# for k in range(0, 9):
#     metadata = modal_analysis.deformed_shape(
#         k, extrude_frames=True, camera=cam,
#         scaling=6000)

print('Modal Mass Participation, X: %.2f' % (np.sum(mstarsX)))
print('Modal Mass Participation, Y: %.2f' % (np.sum(mstarsY)))

print('T_1 = %.2f s\n' % (ts[1]))

vb_elf = np.sum(wi) * cs(ts[1], Sds, Sd1, R, Ie)
print('V_b_elf = %.2f kips \n' % (vb_elf))
print(f'Cs = {cs(ts[1], Sds, Sd1, R, Ie)}')
# cvx = np.reshape(wi, (-1)) * hi_add**k(ts[1]) / np.sum(wi * hi_add**k(ts[1]))

# fx = vb_elf * cvx

# #
# # ELF
# #

# u1_el, u2_el, u3_el = get_floor_displacements(b, fx, 0)

# u1 = Cd / Ie * u1_el
# u2 = Cd / Ie * u2_el
# u3 = Cd / Ie * u3_el

# dr1 = u1 / (15.*12.)
# dr2 = (u2 - u1) / (13.*12.)
# dr3 = (u3 - u2) / (13.*12.)

# print("Drift capacity ratios, X direction (ELF):")
# print("%.2f %.2f %.2f" % (dr1/0.02, dr2/0.02, dr3/0.02))

# print('Note: ELF always leads to a stiffer design.')
# print('      Values are provided for reference.')
# print('      Design is based on modal analysis.')


#
# modal
#
vb_modal = np.zeros(num_modes)
for i in range(num_modes):
    vb_modal[i] = (design_spectrum_ifun(ts[i]) / (R/Ie)) * mstarsX[i] * mtot * 386.22 / 1000.

print(f'V_b_modal_X = {np.sum(vb_modal):.2f} kips')
print(f'Total Seismic Weight: {mtot * 386.22 / 1000:2f} kips')

modal_q0 = np.zeros(num_modes)
modal_dr1 = np.zeros(num_modes)
modal_dr2 = np.zeros(num_modes)
modal_dr3 = np.zeros(num_modes)

for i in range(num_modes):
    modal_q0[i] = gammasX[i] * (design_spectrum_ifun(ts[i]) / (R/Ie)) / (2.*np.pi / ts[i])**2 * 386.22
    modal_dr1[i] = (modal_analysis.node_displacements[str(p_nodes[0].uid)][i][0]) * modal_q0[i]
    modal_dr2[i] = (modal_analysis.node_displacements[str(p_nodes[1].uid)][i][0] - modal_analysis.node_displacements[str(p_nodes[0].uid)][i][0]) * modal_q0[i]
    modal_dr3[i] = (modal_analysis.node_displacements[str(p_nodes[2].uid)][i][0] - modal_analysis.node_displacements[str(p_nodes[1].uid)][i][0]) * modal_q0[i]

dr1 = np.sqrt(np.sum(modal_dr1**2)) / (15.*12.) * Cd / Ie
dr2 = np.sqrt(np.sum(modal_dr2**2)) / (13.*12.) * Cd / Ie
dr3 = np.sqrt(np.sum(modal_dr3**2)) / (13.*12.) * Cd / Ie

print("Drift capacity ratios, X direction (MODAL):")
print("%.2f %.2f %.2f" % (dr1/0.02, dr2/0.02, dr3/0.02))


modal_q0 = np.zeros(num_modes)
modal_dr1 = np.zeros(num_modes)
modal_dr2 = np.zeros(num_modes)
modal_dr3 = np.zeros(num_modes)

for i in range(num_modes):
    modal_q0[i] = gammasY[i] * (design_spectrum_ifun(ts[i]) / (R/Ie)) / (2.*np.pi / ts[i])**2 * 386.22
    modal_dr1[i] = (modal_analysis.node_displacements[str(p_nodes[0].uid)][i][1]) * modal_q0[i]
    modal_dr2[i] = (modal_analysis.node_displacements[str(p_nodes[1].uid)][i][1] - modal_analysis.node_displacements[str(p_nodes[0].uid)][i][1]) * modal_q0[i]
    modal_dr3[i] = (modal_analysis.node_displacements[str(p_nodes[2].uid)][i][1] - modal_analysis.node_displacements[str(p_nodes[1].uid)][i][1]) * modal_q0[i]

dr1 = np.sqrt(np.sum(modal_dr1**2)) / (15.*12.) * Cd / Ie
dr2 = np.sqrt(np.sum(modal_dr2**2)) / (13.*12.) * Cd / Ie
dr3 = np.sqrt(np.sum(modal_dr3**2)) / (13.*12.) * Cd / Ie

print("Drift capacity ratios, Y direction (MODAL):")
print("%.2f %.2f %.2f" % (dr1/0.02, dr2/0.02, dr3/0.02))
