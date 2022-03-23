import sys
sys.path.append("../OpenSees_Model_Builder/src")

import numpy as np
import model
import solver
import time
import pickle
from scipy.interpolate import interp1d
import argparse

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

print("=== Defining Building ===")

# Define a building
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





# strong column - weak beam checks
zc1 = b.sections.registry[sections['lateral_cols']['level_1']].properties['Zx']
zc2 = b.sections.registry[sections['lateral_cols']['level_2']].properties['Zx']
zc3 = b.sections.registry[sections['lateral_cols']['level_3']].properties['Zx']
zc4 = b.sections.registry[sections['lateral_cols']['level_4']].properties['Zx']
zc5 = b.sections.registry[sections['lateral_cols']['level_5']].properties['Zx']
zc6 = b.sections.registry[sections['lateral_cols']['level_6']].properties['Zx']
zc7 = b.sections.registry[sections['lateral_cols']['level_7']].properties['Zx']
zc8 = b.sections.registry[sections['lateral_cols']['level_8']].properties['Zx']
zc9 = b.sections.registry[sections['lateral_cols']['level_9']].properties['Zx']
zc10 = b.sections.registry[sections['lateral_cols']['level_10']].properties['Zx']
b1_sec = b.sections.registry[sections['lateral_beams']['level_1']].properties
b2_sec = b.sections.registry[sections['lateral_beams']['level_2']].properties
b3_sec = b.sections.registry[sections['lateral_beams']['level_3']].properties
b4_sec = b.sections.registry[sections['lateral_beams']['level_4']].properties
b5_sec = b.sections.registry[sections['lateral_beams']['level_5']].properties
b6_sec = b.sections.registry[sections['lateral_beams']['level_6']].properties
b7_sec = b.sections.registry[sections['lateral_beams']['level_7']].properties
b8_sec = b.sections.registry[sections['lateral_beams']['level_8']].properties
b9_sec = b.sections.registry[sections['lateral_beams']['level_9']].properties
b10_sec = b.sections.registry[sections['lateral_beams']['level_10']].properties
c_1 = b1_sec['bf'] * (1. - 0.60) / 2.
c_2 = b2_sec['bf'] * (1. - 0.60) / 2.
c_3 = b3_sec['bf'] * (1. - 0.60) / 2.
c_4 = b4_sec['bf'] * (1. - 0.60) / 2.
c_5 = b5_sec['bf'] * (1. - 0.60) / 2.
c_6 = b6_sec['bf'] * (1. - 0.60) / 2.
c_7 = b7_sec['bf'] * (1. - 0.60) / 2.
c_8 = b8_sec['bf'] * (1. - 0.60) / 2.
c_9 = b9_sec['bf'] * (1. - 0.60) / 2.
c_10 = b10_sec['bf'] * (1. - 0.60) / 2.
zb1 = b1_sec['Zx'] - 2. * c_1 * b1_sec['tf'] * (b1_sec['d'] - b1_sec['tf'])
zb2 = b2_sec['Zx'] - 2. * c_2 * b2_sec['tf'] * (b2_sec['d'] - b2_sec['tf'])
zb3 = b3_sec['Zx'] - 2. * c_3 * b3_sec['tf'] * (b3_sec['d'] - b3_sec['tf'])
zb4 = b4_sec['Zx'] - 2. * c_4 * b4_sec['tf'] * (b4_sec['d'] - b4_sec['tf'])
zb5 = b5_sec['Zx'] - 2. * c_5 * b5_sec['tf'] * (b5_sec['d'] - b5_sec['tf'])
zb6 = b6_sec['Zx'] - 2. * c_6 * b6_sec['tf'] * (b6_sec['d'] - b6_sec['tf'])
zb7 = b7_sec['Zx'] - 2. * c_7 * b7_sec['tf'] * (b7_sec['d'] - b7_sec['tf'])
zb8 = b8_sec['Zx'] - 2. * c_8 * b8_sec['tf'] * (b8_sec['d'] - b8_sec['tf'])
zb9 = b9_sec['Zx'] - 2. * c_9 * b9_sec['tf'] * (b9_sec['d'] - b9_sec['tf'])
zb10 = b10_sec['Zx'] - 2. * c_10 * b10_sec['tf'] * (b10_sec['d'] - b10_sec['tf'])

scwbr1 = (zc1 + zc2) / (2. * zb1)
scwbr2 = (zc2 + zc3) / (2. * zb2)
scwbr3 = (zc3 + zc4) / (2. * zb3)
scwbr4 = (zc4 + zc5) / (2. * zb4)
scwbr5 = (zc5 + zc6) / (2. * zb5)
scwbr6 = (zc6 + zc7) / (2. * zb6)
scwbr7 = (zc7 + zc8) / (2. * zb7)
scwbr8 = (zc8 + zc9) / (2. * zb8)
scwbr9 = (zc9 + zc10) / (2. * zb9)

print("SCWB ratios: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n" % (
    scwbr1,
    scwbr2,
    scwbr3,
    scwbr4,
    scwbr5,
    scwbr6,
    scwbr7,
    scwbr8,
    scwbr9))


#
# define structural members
#

elastic_modeling_type = {'type': 'elastic'}
lat_col_ends = {'type': 'steel_W_PZ_IMK', 'end_dist': 0.05,
                'Lb/ry': 60., 'L/H': 1.0, 'pgpye': 0.005,
                'doubler plate thickness': 0.00}
RBS_ends = {'type': 'RBS', 'end_dist': (17.50+17.5)/(25.*12.),
            'rbs_length': 17.5, 'rbs_reduction': .60, 'rbs_n_sub': 1}
lat_bm_ends = RBS_ends
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

nsub = 1  # element subdivision

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


#
# define surface loads
#

print("=== Defining Loads ===")

# to validate design - only use dead loads (ASCE 7 sec. 12.7.2)

b.set_active_levels([f'{i+1}' for i in range(9)])
b.assign_surface_DL((75.+15.+20.)/(12.**2))

b.set_active_levels(['10'])
b.assign_surface_DL((75.+15.+80.)/(12.**2))


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

print('=== Preprocessing Building ===')


b.preprocess(assume_floor_slabs=True, self_weight=True,
             steel_panel_zones=True, elevate_column_splices=0.25)


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


print('=== Running Eigenvalue Analysis ===')

p_nodes = b.list_of_parent_nodes()

Cd = 5.5
R = 8.0
Ie = 1.0

# Sds = 1.206
# Sd1 = 1.1520
Sds = 1.58
Sd1 = 1.38
Tshort = Sd1/Sds

# period estimation (Table 12.8-2)

ct = 0.028
exponent = 0.8

wi = np.zeros(10)
hi_add = np.zeros(10)
for i in range(10):
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

num_modes = 6
modal_analysis = solver.ModalAnalysis(b, num_modes=num_modes)
modal_analysis.run()
gammasX, mstarsX = modal_analysis.modal_participation_factors('x')
# gammasY, mstarsY = modal_analysis.modal_participation_factors('y')
ts = modal_analysis.periods

print('T_1 = %.2f s\n' % (ts[1]))

vb_elf = np.sum(wi) * cs(ts[1], Sds, Sd1, R, Ie)
print('V_b_elf = %.2f kips \n' % (vb_elf))

cvx = np.reshape(wi, (-1)) * hi_add**k(ts[1]) / np.sum(wi * hi_add**k(ts[1]))

fx = vb_elf * cvx

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

modal_q0 = np.zeros(num_modes)
modal_dr1 = np.zeros(num_modes)
modal_dr2 = np.zeros(num_modes)
modal_dr3 = np.zeros(num_modes)
modal_dr4 = np.zeros(num_modes)
modal_dr5 = np.zeros(num_modes)
modal_dr6 = np.zeros(num_modes)
modal_dr7 = np.zeros(num_modes)
modal_dr8 = np.zeros(num_modes)
modal_dr9 = np.zeros(num_modes)
modal_dr10 = np.zeros(num_modes)

for i in range(num_modes):
    modal_q0[i] = gammasX[i] * cs(ts[i], Sds, Sd1, R, Ie) / (2.*np.pi / ts[i])**2 * 386.22
    modal_dr1[i] = (modal_analysis.node_displacements[str(p_nodes[0].uid)][i][0]) * modal_q0[i]
    modal_dr2[i] = (modal_analysis.node_displacements[str(p_nodes[1].uid)][i][0] - modal_analysis.node_displacements[str(p_nodes[0].uid)][i][0]) * modal_q0[i]
    modal_dr3[i] = (modal_analysis.node_displacements[str(p_nodes[2].uid)][i][0] - modal_analysis.node_displacements[str(p_nodes[1].uid)][i][0]) * modal_q0[i]
    modal_dr4[i] = (modal_analysis.node_displacements[str(p_nodes[3].uid)][i][0] - modal_analysis.node_displacements[str(p_nodes[2].uid)][i][0]) * modal_q0[i]
    modal_dr5[i] = (modal_analysis.node_displacements[str(p_nodes[4].uid)][i][0] - modal_analysis.node_displacements[str(p_nodes[3].uid)][i][0]) * modal_q0[i]
    modal_dr6[i] = (modal_analysis.node_displacements[str(p_nodes[5].uid)][i][0] - modal_analysis.node_displacements[str(p_nodes[4].uid)][i][0]) * modal_q0[i]
    modal_dr7[i] = (modal_analysis.node_displacements[str(p_nodes[6].uid)][i][0] - modal_analysis.node_displacements[str(p_nodes[5].uid)][i][0]) * modal_q0[i]
    modal_dr8[i] = (modal_analysis.node_displacements[str(p_nodes[7].uid)][i][0] - modal_analysis.node_displacements[str(p_nodes[6].uid)][i][0]) * modal_q0[i]
    modal_dr9[i] = (modal_analysis.node_displacements[str(p_nodes[8].uid)][i][0] - modal_analysis.node_displacements[str(p_nodes[7].uid)][i][0]) * modal_q0[i]
    modal_dr10[i] = (modal_analysis.node_displacements[str(p_nodes[9].uid)][i][0] - modal_analysis.node_displacements[str(p_nodes[8].uid)][i][0]) * modal_q0[i]

dr1 = np.sqrt(np.sum(modal_dr1**2)) / (15.*12.) * Cd / Ie
dr2 = np.sqrt(np.sum(modal_dr2**2)) / (13.*12.) * Cd / Ie
dr3 = np.sqrt(np.sum(modal_dr3**2)) / (13.*12.) * Cd / Ie
dr4 = np.sqrt(np.sum(modal_dr4**2)) / (13.*12.) * Cd / Ie
dr5 = np.sqrt(np.sum(modal_dr5**2)) / (13.*12.) * Cd / Ie
dr6 = np.sqrt(np.sum(modal_dr6**2)) / (13.*12.) * Cd / Ie
dr7 = np.sqrt(np.sum(modal_dr7**2)) / (13.*12.) * Cd / Ie
dr8 = np.sqrt(np.sum(modal_dr8**2)) / (13.*12.) * Cd / Ie
dr9 = np.sqrt(np.sum(modal_dr9**2)) / (13.*12.) * Cd / Ie
dr10 = np.sqrt(np.sum(modal_dr10**2)) / (13.*12.) * Cd / Ie

print("Drift capacity ratios, X direction (MODAL):")
print("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % (dr1/0.02, dr2/0.02, dr3/0.02, dr4/0.02, dr5/0.02, dr6/0.02, dr7/0.02, dr8/0.02, dr9/0.02, dr10/0.02))


# modal_q0 = np.zeros(num_modes)
# modal_dr1 = np.zeros(num_modes)
# modal_dr2 = np.zeros(num_modes)
# modal_dr3 = np.zeros(num_modes)

# for i in range(num_modes):
#     modal_q0[i] = gammasY[i] * cs(ts[i], Sds, Sd1, R, Ie) / (2.*np.pi / ts[i])**2 * 386.22
#     modal_dr1[i] = (modal_analysis.node_displacements[str(p_nodes[0].uid)][i][1]) * modal_q0[i]
#     modal_dr2[i] = (modal_analysis.node_displacements[str(p_nodes[1].uid)][i][1] - modal_analysis.node_displacements[str(p_nodes[0].uid)][i][1]) * modal_q0[i]
#     modal_dr3[i] = (modal_analysis.node_displacements[str(p_nodes[2].uid)][i][1] - modal_analysis.node_displacements[str(p_nodes[1].uid)][i][1]) * modal_q0[i]

# dr1 = np.sqrt(np.sum(modal_dr1**2)) / (15.*12.) * Cd / Ie
# dr2 = np.sqrt(np.sum(modal_dr2**2)) / (13.*12.) * Cd / Ie
# dr3 = np.sqrt(np.sum(modal_dr3**2)) / (13.*12.) * Cd / Ie

# print("Drift capacity ratios, Y direction (MODAL):")
# print("%.2f %.2f %.2f" % (dr1/0.02, dr2/0.02, dr3/0.02))
