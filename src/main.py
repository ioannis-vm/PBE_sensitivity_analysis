import sys
sys.path.append("../OpenSeesPy_Building_Modeler")

import numpy as np
import modeler
import solver

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
c1lat = "W24X55"
c2lat = "W24X55"
c3lat = "W24X55"
b1lat = "W24X55"
b2lat = "W24X55"
b3lat = "W24X55"

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
        "OpenSeesPy_Building_Modeler/section_data/sections.json",
        'W',
        [sec])

for sec in hsssections:
    b.add_sections_from_json(
        "OpenSeesPy_Building_Modeler/section_data/sections.json",
        'HSS',
        [sec])

#
# define structural members
#

elastic_modeling_type = {'type': 'elastic'}
fiber_modeling_type = {'type': 'fiber', 'n_x': 25, 'n_y': 50}


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
b.add_columns_from_grids(model_as=fiber_modeling_type)
b.set_active_levels(['2'])
b.set_active_section(c2lat)
b.add_columns_from_grids(model_as=fiber_modeling_type)
b.set_active_levels(['3'])
b.set_active_section(c3lat)
b.add_columns_from_grids(model_as=fiber_modeling_type)

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

b.clear_gridlines()
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
b.add_columns_from_grids(model_as=fiber_modeling_type)
b.set_active_levels(['2'])
b.set_active_section(c2lat)
b.add_columns_from_grids(model_as=fiber_modeling_type)
b.set_active_levels(['3'])
b.set_active_section(c3lat)
b.add_columns_from_grids(model_as=fiber_modeling_type)

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

b.clear_gridlines()
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
b.add_columns_from_grids(model_as=elastic_modeling_type)
b.set_active_levels(['2'])
b.set_active_section(c2grav)
b.add_columns_from_grids(model_as=elastic_modeling_type)
b.set_active_levels(['3'])
b.set_active_section(c3grav)
b.add_columns_from_grids(model_as=elastic_modeling_type)

b.set_active_placement('top_center')
b.set_active_levels(['1'])
b.set_active_section(b1grav)
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)
b.set_active_levels(['2'])
b.set_active_section(b2grav)
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)
b.set_active_levels(['3'])
b.set_active_section(b3grav)
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)

b.set_active_levels('all_above_base')
b.set_active_section(bsecondary)

b.clear_gridlines()
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('As1', [08.125*12., 000.00*12.], [08.125*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)

b.clear_gridlines()
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('As2', [016.25*12., 000.00*12.], [016.25*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)

b.clear_gridlines()
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('As3', [024.38*12., 000.00*12.], [024.38*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)

b.clear_gridlines()
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('Bs1', [040.83*12., 000.00*12.], [040.83*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)

b.clear_gridlines()
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('Bs2', [049.17*12., 000.00*12.], [049.17*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)

b.clear_gridlines()
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('Cs1', [065.83*12., 000.00*12.], [065.83*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)

b.clear_gridlines()
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('Cs2', [074.17*12., 000.00*12.], [074.17*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)

b.clear_gridlines()
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('Ds1', [090.83*12., 000.00*12.], [090.83*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)

b.clear_gridlines()
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('Ds2', [099.17*12., 000.00*12.], [099.17*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)

b.clear_gridlines()
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('Es1', [115.60*12., 000.00*12.], [115.60*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)

b.clear_gridlines()
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('Es2', [123.80*12., 000.00*12.], [123.80*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)

b.clear_gridlines()
b.add_gridline('1', [000.00*12., 000.00*12.], [140.00*12., 000.00*12.])
b.add_gridline('2', [000.00*12., 025.00*12.], [140.00*12., 025.00*12.])
b.add_gridline('3', [000.00*12., 050.00*12.], [140.00*12., 050.00*12.])
b.add_gridline('4', [000.00*12., 075.00*12.], [140.00*12., 075.00*12.])
b.add_gridline('5', [000.00*12., 100.00*12.], [140.00*12., 100.00*12.])
b.add_gridline('Es3', [131.90*12., 000.00*12.], [131.90*12., 100.00*12.])
b.add_beams_from_grid_intersections(ends={'type': 'pinned', 'dist': 0.01},
                                    model_as=elastic_modeling_type)

b.clear_gridlines()

#
# define surface loads
#

b.set_active_levels(['1', '2'])
b.assign_surface_DL(0.00)

b.set_active_levels(['3'])
b.assign_surface_DL(0.00)

b.select_perimeter_beams_all()
b.selection.add_UDL(np.array((0.00, 0.00, 0.00)))

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


# ~~~~~~~~~~~~~~~~~ #
#  linear analysis  #
# ~~~~~~~~~~~~~~~~~ #

# for node in b.list_of_parent_nodes():
#     node.load += np.array([0.00, 100000.00, 0.00, 0.00, 0.00, 0.00])

# # performing a linear gravity analysis.
# linear_gravity_analysis = solver.LinearGravityAnalysis(b)
# linear_gravity_analysis.run()

# # # retrieving aggregated textual results
# # reactions = linear_gravity_analysis.global_reactions(0)
# # print(reactions[0:3] / 1000)  # kip
# # print(reactions[3:6] / 1000 / 12)  # kip-ft

# # visualizing results
# linear_gravity_analysis.deformed_shape(extrude_frames=True)
# # linear_gravity_analysis.basic_forces()

# for node in b.list_of_parent_nodes():
#     print(node.coords)



# ~~~~~~~~~~~~~~~~ #
#  modal analysis  #
# ~~~~~~~~~~~~~~~~ #

# performing a linear modal analysis
modal_analysis = solver.ModalAnalysis(b, num_modes=1)

import pdb
pdb.set_trace()
modal_analysis.run()

# retrieving textual results
# print(modal_analysis.periods)

# visualizing results
# modal_analysis.deformed_shape(step=0, scaling=0.00, extrude_frames=True)
