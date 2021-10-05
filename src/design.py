import sys
sys.path.append("../OpenSeesPy_Building_Modeler")

import numpy as np
import modeler
import solver
import time
import pickle
from scipy.interpolate import interp1d
import argparse

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--output_path')
args = parser.parse_args()

output_path = args.output_path


# Define a building
b = modeler.Building()

# Add levels
b.add_level("base", 0.00, "fixed")
b.add_level("1", 13.00 * 12.00)
b.add_level("2", 13.00 * 12.00 * 2.00)
b.add_level("3", 13.00 * 12.00 * 3.00)

# assign section properties
c1grav = "HSS14X14X5/16"
c2grav = "HSS14X14X5/16"
c3grav = "HSS14X14X5/16"
b1grav = "W24X55"
b2grav = "W24X55"
b3grav = "W27X84"
bsecondary = "W16X26"

c1lat = "W27X178"
c2lat = "W27X178"
c3lat = "W27X114"
b1lat = "W27X114"
b2lat = "W27X102"
b3lat = "W27X84"

# define materials
b.materials.enable_Steel02()
b.set_active_material('steel')

# define sections
wsections = set()
wsections.add(b1grav)
wsections.add(b2grav)
wsections.add(b3grav)
wsections.add(bsecondary)
wsections.add(c1lat)
wsections.add(c2lat)
wsections.add(c3lat)
wsections.add(b1lat)
wsections.add(b2lat)
wsections.add(b3lat)
hsssections = set()
hsssections.add(c1grav)
hsssections.add(c2grav)
hsssections.add(c3grav)

for sec in wsections:
    b.add_sections_from_json(
        "../OpenSeesPy_Building_Modeler/section_data/sections.json",
        'W',
        [sec])

for sec in hsssections:
    b.add_sections_from_json(
        "../OpenSeesPy_Building_Modeler/section_data/sections.json",
        'HSS',
        [sec])

#
# define structural members
#

elastic_modeling_type = {'type': 'elastic'}
fiber_modeling_type = {'type': 'fiber', 'n_x': 25, 'n_y': 50}
# fiber_modeling_type = {'type': 'elastic'}  # (debug)

pinned_ends = {'type': 'pinned', 'dist': 0.01}
# pinned_ends = {'type': 'fixed'}

# gtransf = 'Linear'
gtransf = 'Corotational'

b.add_gridline('A', [0.00, 0.00], [0.00, 3.*25.*12.])
b.add_gridline('A1', [0.00, 0.00], [1.00, 0.00])
b.add_gridline('A2', [0.00, 25.00*12.], [1.00, 25.00*12.])
b.add_gridline('A3', [0.00, 50.00*12.], [1.00, 50.00*12.])
b.add_gridline('A4', [0.00, 75.00*12.], [1.00, 75.00*12.])
b.add_gridline('F', [140.00*12., 0.00], [140.00*12., 3.*25.*12.])
b.add_gridline('F1', [140.00*12., 0.00], [141.00*12., 0.00])
b.add_gridline('F2', [140.00*12., 25.00*12.], [141.00*12., 25.00*12.])
b.add_gridline('F3', [140.00*12., 50.00*12.], [141.00*12., 50.00*12.])
b.add_gridline('F4', [140.00*12., 75.00*12.], [141.00*12., 75.00*12.])

b.set_active_angle(0.00)
b.set_active_placement('centroid')
b.set_active_levels(['1'])
b.set_active_section(c1lat)
b.add_columns_from_grids(model_as=fiber_modeling_type, geomTransf=gtransf)
b.set_active_levels(['2'])
b.set_active_section(c2lat)
b.add_columns_from_grids(model_as=fiber_modeling_type, geomTransf=gtransf)
b.set_active_levels(['3'])
b.set_active_section(c3lat)
b.add_columns_from_grids(model_as=fiber_modeling_type, geomTransf=gtransf)

b.set_active_placement('top_center')
b.set_active_levels(['1'])
b.set_active_section(b1lat)
b.add_beams_from_grid_intersections(model_as=fiber_modeling_type)
b.set_active_levels(['2'])
b.set_active_section(b2lat)
b.add_beams_from_grid_intersections(model_as=fiber_modeling_type)
b.set_active_levels(['3'])
b.set_active_section(b3lat)
b.add_beams_from_grid_intersections(model_as=fiber_modeling_type)

b.clear_gridlines_all()
b.add_gridline('1', [32.5*12., 0.00], [107.5*12., 0.00])
b.add_gridline('1b', [32.5*12., 0.00], [32.5*12., 1.00])
b.add_gridline('1c', [57.5*12., 0.00], [57.5*12., 1.00])
b.add_gridline('1d', [82.5*12., 0.00], [82.5*12., 1.00])
b.add_gridline('1e', [107.5*12., 0.00], [107.5*12., 1.00])
b.add_gridline('5', [32.5*12., 100.00*12.], [107.5*12., 100.00*12.])
b.add_gridline('5b', [32.5*12., 100.00*12.], [32.5*12., 101.00*12.])
b.add_gridline('5c', [57.5*12., 100.00*12.], [57.5*12., 101.00*12.])
b.add_gridline('5d', [82.5*12., 100.00*12.], [82.5*12., 101.00*12.])
b.add_gridline('5e', [107.5*12., 100.00*12.], [107.5*12., 101.00*12.])

b.set_active_angle(np.pi/2.00)
b.set_active_placement('centroid')
b.set_active_levels(['1'])
b.set_active_section(c1lat)
b.add_columns_from_grids(model_as=fiber_modeling_type, geomTransf=gtransf)
b.set_active_levels(['2'])
b.set_active_section(c2lat)
b.add_columns_from_grids(model_as=fiber_modeling_type, geomTransf=gtransf)
b.set_active_levels(['3'])
b.set_active_section(c3lat)
b.add_columns_from_grids(model_as=fiber_modeling_type, geomTransf=gtransf)

b.set_active_angle(0.00)
b.set_active_placement('top_center')
b.set_active_levels(['1'])
b.set_active_section(b1lat)
b.add_beams_from_grid_intersections(model_as=fiber_modeling_type)
b.set_active_levels(['2'])
b.set_active_section(b2lat)
b.add_beams_from_grid_intersections(model_as=fiber_modeling_type)
b.set_active_levels(['3'])
b.set_active_section(b3lat)
b.add_beams_from_grid_intersections(model_as=fiber_modeling_type)

b.clear_gridlines_all()
b.add_gridline('A', [000.00*12., 000.00*12.], [000.00*12., 100.00*12.])
b.add_gridline('B', [032.50*12., 000.00*12.], [032.50*12., 100.00*12.])
b.add_gridline('C', [057.50*12., 000.00*12.], [057.50*12., 100.00*12.])
b.add_gridline('D', [082.50*12., 000.00*12.], [082.50*12., 100.00*12.])
b.add_gridline('E', [107.50*12., 000.00*12.], [107.50*12., 100.00*12.])
b.add_gridline('F', [140.00*12., 000.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])

b.set_active_placement('centroid')
b.set_active_levels(['1'])
b.set_active_section(c1grav)
b.add_columns_from_grids(model_as=fiber_modeling_type, geomTransf=gtransf)
b.set_active_levels(['2'])
b.set_active_section(c2grav)
b.add_columns_from_grids(model_as=fiber_modeling_type, geomTransf=gtransf)
b.set_active_levels(['3'])
b.set_active_section(c3grav)
b.add_columns_from_grids(model_as=fiber_modeling_type, geomTransf=gtransf)

b.set_active_placement('top_center')
b.set_active_levels(['1'])
b.set_active_section(b1grav)
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.set_active_levels(['2'])
b.set_active_section(b2grav)
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.set_active_levels(['3'])
b.set_active_section(b3grav)
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)

b.set_active_levels('all_above_base')
b.set_active_section(bsecondary)

b.clear_gridlines_all()
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('As1', [08.125*12., 000.00*12.], [08.125*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.clear_gridlines(['As1'])
b.add_gridline('As2', [016.25*12., 000.00*12.], [016.25*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.clear_gridlines(['As2'])
b.add_gridline('As3', [024.38*12., 000.00*12.], [024.38*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.clear_gridlines(['As3'])
b.add_gridline('Bs1', [040.83*12., 000.00*12.], [040.83*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.clear_gridlines(['Bs1'])
b.add_gridline('Bs2', [049.17*12., 000.00*12.], [049.17*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.clear_gridlines(['Bs2'])
b.add_gridline('Cs1', [065.83*12., 000.00*12.], [065.83*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.clear_gridlines(['Cs1'])
b.add_gridline('Cs2', [074.17*12., 000.00*12.], [074.17*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.clear_gridlines(['Cs2'])
b.add_gridline('Ds1', [090.83*12., 000.00*12.], [090.83*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.clear_gridlines(['Ds1'])
b.add_gridline('Ds2', [099.17*12., 000.00*12.], [099.17*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.clear_gridlines(['Ds2'])
b.add_gridline('Es1', [115.60*12., 000.00*12.], [115.60*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.clear_gridlines(['Es1'])
b.add_gridline('Es2', [123.80*12., 000.00*12.], [123.80*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.clear_gridlines(['Es2'])
b.add_gridline('Es3', [131.90*12., 000.00*12.], [131.90*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends=pinned_ends,
                                    model_as=elastic_modeling_type)
b.clear_gridlines_all()

#
# define surface loads
#

# TODO: Note that here I only define dead load, without any amplification

b.set_active_levels(['1', '2'])
b.assign_surface_DL(0.4514 + 0.1389)

b.set_active_levels(['3'])
b.assign_surface_DL(0.4514 + 0.6944)

b.select_perimeter_beams_all()
b.selection.add_UDL(np.array((0.00, 0.00, -10.83)))

b.preprocess(assume_floor_slabs=True, self_weight=True)


# b.plot_building_geometry(extrude_frames=True,
#                          offsets=True,
#                          gridlines=True,
#                          global_axes=False,
#                          diaphragm_lines=True,
#                          tributary_areas=True,
#                          just_selection=False,
#                          parent_nodes=True,
#                          frame_axes=False)



with open(output_path, 'wb') as f:
    pickle.dump(b, f)



# ~~~~~~~~~~~~~~~~ #
#  modal analysis  #
# ~~~~~~~~~~~~~~~~ #

# # performing a linear modal analysis
# modal_analysis = solver.ModalAnalysis(b, num_modes=1)

# modal_analysis.run()

# # retrieving textual results
# print(modal_analysis.periods)
# period = modal_analysis.periods[0]

# # visualizing results
# modal_analysis.deformed_shape(step=0, scaling=0.00, extrude_frames=True)


# ~~~~~~~~~~~~~~~~~~~~ #
#  ASCE ELF procedure  #
# ~~~~~~~~~~~~~~~~~~~~ #

# p_nodes = b.list_of_parent_nodes()

# Cd = 5.5
# R = 8.0
# Ie = 1.0

# Sds = 1.33
# Sd1 = 0.75
# Tshort = Sd1/Sds

# # period estimation (Table 12.8-2)
# def Tmax(ct, exponent, height, Sd1):
#     def cu(Sd1):
#         if Sd1 <= 0.1:
#             cu = 1.7
#         elif Sd1 >= 0.4:
#             cu = 1.4
#         else:
#             x = np.array([0.1, 0.15, 0.2, 0.3, 0.4])
#             y = np.array([1.7, 1.6, 1.5, 1.4, 1.4])
#             f = interp1d(x, y)
#             cu = f(np.array([Sd1]))[0]
#         return cu

#     Ta = ct * height**exponent
#     xx = np.linspace(0.00, 0.60, 1000)
#     yy = np.zeros(1000)
#     for i in range(1000):
#         yy[i] = cu(xx[i])
#     return cu(Sd1) * Ta

# ct = 0.028
# exponent = 0.8

# def cs(T, Sds, Sd1, R, Ie):
#     Tshort = Sd1/Sds
#     if T < Tshort:
#         res = Sds / R * Ie
#     else:
#         res = Sd1 / R * Ie / T
#     return res


# wi = np.zeros(3)
# hi = np.zeros(3)
# for i in range(3):
#     wi[i] = p_nodes[i].mass[0]*386/1000
#     hi[i] = 13. * (i+1.)

# T_max = Tmax(ct, exponent, np.max(hi), Sd1)

# print('T_max:', T_max)

# Vb = np.sum(wi) * cs(T_max, Sds, Sd1, R, Ie)

# def k(T):
#     if T <= 0.5:
#         res = 1.0
#     elif T >= 2.5:
#         res = 2.0
#     else:
#         x = np.array([0.5, 2.5])
#         y = np.array([1., 2.])
#         f = interp1d(x, y)
#         res = f(np.array([T]))[0]
#     return res

# cvx = wi * hi**k(T_max) / np.sum(wi * hi**k(T_max))

# fx = Vb * cvx

# ~~~~~~~~~~~~~~~~~ #
#  linear analysis  #
# ~~~~~~~~~~~~~~~~~ #

#
# x direction
#


# parent_nodes = b.list_of_parent_nodes()
# for i, node in enumerate(parent_nodes):
#     node.load = np.array([fx[i]*1e3, 0.00, 0.00, 0.00, 0.00, 0.00])

# linear_gravity_analysis = solver.LinearGravityAnalysis(b)
# linear_gravity_analysis.run()

# u1_el = linear_gravity_analysis.node_displacements[parent_nodes[0].uniq_id][0][0]
# u2_el = linear_gravity_analysis.node_displacements[parent_nodes[1].uniq_id][0][0]
# u3_el = linear_gravity_analysis.node_displacements[parent_nodes[2].uniq_id][0][0]

# u1 = Cd / Ie * u1_el
# u2 = Cd / Ie * u2_el
# u3 = Cd / Ie * u3_el

# dr1 = u1 / (13.*12.)
# dr2 = (u2 - u1) / (13.*12.)
# dr3 = (u3 - u2) / (13.*12.)

# print("Drift capacity ratios:")
# print(dr1/0.02, dr2/0.02, dr3/0.02)


# #
# # y direction
# #

# for i, node in enumerate(parent_nodes):
#     node.load = np.array([0.00, fx[i]*1e3, 0.00, 0.00, 0.00, 0.00])

# linear_gravity_analysis = solver.LinearGravityAnalysis(b)
# linear_gravity_analysis.run()

# u1_el = linear_gravity_analysis.node_displacements[parent_nodes[0].uniq_id][0][1]
# u2_el = linear_gravity_analysis.node_displacements[parent_nodes[1].uniq_id][0][1]
# u3_el = linear_gravity_analysis.node_displacements[parent_nodes[2].uniq_id][0][1]

# u1 = Cd / Ie * u1_el
# u2 = Cd / Ie * u2_el
# u3 = Cd / Ie * u3_el

# dr1 = u1 / (13.*12.)
# dr2 = (u2 - u1) / (13.*12.)
# dr3 = (u3 - u2) / (13.*12.)

# print("Drift capacity ratios:")
# print(dr1/0.02, dr2/0.02, dr3/0.02)
