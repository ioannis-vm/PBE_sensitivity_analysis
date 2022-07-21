from osmg.model import Model
from osmg.gen.beamcolumn_gen import BeamColumnGenerator
from osmg.gen.section_gen import SectionGenerator
from osmg import defaults
from osmg.preprocessing.self_weight_mass import self_weight
from osmg.preprocessing.self_weight_mass import self_mass
from osmg.ops.section import ElasticSection
from osmg.ops.element import elasticBeamColumn
from osmg.gen.zerolength_gen import release_56
from osmg.gen.zerolength_gen import release_5_imk_6
from osmg.gen.zerolength_gen import imk_6
from osmg.gen.zerolength_gen import gravity_shear_tab
from osmg.load_case import LoadCase
from osmg.preprocessing.tributary_area_analysis import PolygonLoad
import numpy as np


def generate_smrf_archetype(
        level_elevs,
        sections,
        doubler_plate_thicknesses,
        surf_loads,
        surf_loads_massless
):

    mdl = Model('test_model')
    mdl.settings.imperial_units = True
    mcg = BeamColumnGenerator(mdl)
    secg = SectionGenerator(mdl)

    num_levels = len(level_elevs)

    mdl.add_level(0, 0.00)
    for i, h in enumerate(level_elevs):
        mdl.add_level(i+1, h)

    defaults.load_default_steel(mdl)
    defaults.load_default_fix_release(mdl)
    steel_phys_mat = mdl.physical_materials.retrieve_by_attr('name', 'default steel')

    # define sections
    wsections = set()
    for lvl_tag in [f'level_{i+1}' for i in range(num_levels)]:
        wsections.add(sections['gravity_beams_a'][lvl_tag])
        wsections.add(sections['gravity_beams_b'][lvl_tag])
        wsections.add(sections['gravity_beams_c'][lvl_tag])
        wsections.add(sections['gravity_beams_d'][lvl_tag])
        wsections.add(sections['gravity_beams_e'][lvl_tag])
        wsections.add(sections['lateral_beams'][lvl_tag])
        wsections.add(sections['gravity_cols'][lvl_tag])
    for function in ['exterior', 'interior']:
        for lvl_tag in [f'level_{i+1}' for i in range(num_levels)]:
            wsections.add(sections['lateral_cols'][function][lvl_tag])

    section_type = ElasticSection
    element_type = elasticBeamColumn
    sec_collection = mdl.elastic_sections

    for sec in wsections:
        secg.load_AISC_from_database(
            'W',
            [sec],
            'default steel',
            'default steel',
            section_type
        )

    # generate a dictionary containing coordinates given gridline tag names
    point = {}
    x_grd_tags = ['A', 'B', 'C', 'D', 'E', 'F']
    y_grd_tags = ['5', '4', '3', '2', '1']
    x_grd_locs = np.array([0.00, 32.5, 57.5, 82.5, 107.5, 140.00]) * 12.00 + 10.00  # (in)
    y_grd_locs = np.array([0.00, 25.00, 50.00, 75.00, 100.00]) * 12.00 + 10.00  # (in)

    n_sub = 1

    for i in range(len(x_grd_tags)):
        point[x_grd_tags[i]] = {}
        for j in range(len(y_grd_tags)):
            point[x_grd_tags[i]][y_grd_tags[j]] = \
                np.array([x_grd_locs[i], y_grd_locs[j]])

    lat_col_n_sub = 2
    col_gtransf = 'Corotational'

    for level_counter in range(num_levels):
        level_tag = 'level_'+str(level_counter+1)
        mdl.levels.set_active([level_counter+1])

        # define gravity columns
        sec = sec_collection.retrieve_by_attr('name', sections['gravity_cols'][level_tag])
        for tag in ['A', 'F']:
            pt = point[tag]['1']
            mcg.add_vertical_active(
                pt[0], pt[1],
                np.zeros(3), np.zeros(3),
                col_gtransf,
                n_sub,
                sec,
                element_type,
                'centroid',
                0.00,
                method='generate_hinged_component_assembly',
                additional_args={
                    'zerolength_gen_i': None,
                    'zerolength_gen_args_i': {},
                    'zerolength_gen_j': release_56,
                    'zerolength_gen_args_j': {
                        'distance': 10.00,
                        'n_sub': 1
                    },
                }
            )
        for tag1 in ['B', 'C', 'D', 'E']:
            for tag2 in ['2', '3', '4']:
                pt = point[tag1][tag2]
                mcg.add_vertical_active(
                    pt[0], pt[1],
                    np.zeros(3), np.zeros(3),
                    col_gtransf,
                    n_sub,
                    sec,
                    element_type,
                    'centroid',
                    0.00,
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': None,
                        'zerolength_gen_args_i': {},
                        'zerolength_gen_j': release_56,
                        'zerolength_gen_args_j': {
                            'distance': 10.00,
                            'n_sub': 1
                        },
                    }
                )

        # define X-dir frame columns
        sec = sec_collection.retrieve_by_attr('name', sections['lateral_cols']['exterior'][level_tag])
        column_depth = sec.properties['d']
        beam_depth = sec_collection.retrieve_by_attr('name', sections['lateral_beams'][level_tag]).properties['d']
        for tag1 in ['B', 'E']:
            for tag2 in ['1', '5']:
                pt = point[tag1][tag2]
                mcg.add_pz_active(
                    pt[0], pt[1],
                    sec,
                    steel_phys_mat,
                    np.pi/2.00,
                    column_depth,
                    beam_depth,
                    doubler_plate_thicknesses['exterior'][level_tag],
                    0.02
                )
                mcg.add_vertical_active(
                    pt[0], pt[1],
                    np.zeros(3), np.zeros(3),
                    col_gtransf,
                    lat_col_n_sub,
                    sec,
                    element_type,
                    'centroid',
                    np.pi/2.00,
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': release_5_imk_6,
                        'zerolength_gen_args_i': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': release_5_imk_6,
                        'zerolength_gen_args_j': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )
        sec = sec_collection.retrieve_by_attr('name', sections['lateral_cols']['interior'][level_tag])
        column_depth = sec.properties['d']
        for tag1 in ['C', 'D']:
            for tag2 in ['1', '5']:
                pt = point[tag1][tag2]
                mcg.add_pz_active(
                    pt[0], pt[1],
                    sec,
                    steel_phys_mat,
                    np.pi/2.00,
                    column_depth,
                    beam_depth,
                    doubler_plate_thicknesses['interior'][level_tag],
                    0.02
                )
                mcg.add_vertical_active(
                    pt[0], pt[1],
                    np.zeros(3), np.zeros(3),
                    col_gtransf,
                    lat_col_n_sub,
                    sec,
                    element_type,
                    'centroid',
                    np.pi/2.00,
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': release_5_imk_6,
                        'zerolength_gen_args_i': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': release_5_imk_6,
                        'zerolength_gen_args_j': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )

        # deffine Y-dir frame columns
        sec = sec_collection.retrieve_by_attr('name', sections['lateral_cols']['exterior'][level_tag])
        column_depth = sec.properties['d']
        for tag1 in ['A', 'F']:
            for tag2 in ['5', '2']:
                pt = point[tag1][tag2]
                mcg.add_pz_active(
                    pt[0], pt[1],
                    sec,
                    steel_phys_mat,
                    0.00,
                    column_depth,
                    beam_depth,
                    doubler_plate_thicknesses['exterior'][level_tag],
                    0.02
                )
                mcg.add_vertical_active(
                    pt[0], pt[1],
                    np.zeros(3), np.zeros(3),
                    col_gtransf,
                    lat_col_n_sub,
                    sec,
                    element_type,
                    'centroid',
                    0.00,
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': release_5_imk_6,
                        'zerolength_gen_args_i': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': release_5_imk_6,
                        'zerolength_gen_args_j': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )
        sec = sec_collection.retrieve_by_attr('name', sections['lateral_cols']['interior'][level_tag])
        column_depth = sec.properties['d']
        for tag1 in ['A', 'F']:
            for tag2 in ['4', '3']:
                pt = point[tag1][tag2]
                mcg.add_pz_active(
                    pt[0], pt[1],
                    sec,
                    steel_phys_mat,
                    0.00,
                    column_depth,
                    beam_depth,
                    doubler_plate_thicknesses['interior'][level_tag],
                    0.02
                )
                mcg.add_vertical_active(
                    pt[0], pt[1],
                    np.zeros(3), np.zeros(3),
                    col_gtransf,
                    lat_col_n_sub,
                    sec,
                    element_type,
                    'centroid',
                    0.00,
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': release_5_imk_6,
                        'zerolength_gen_args_i': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': release_5_imk_6,
                        'zerolength_gen_args_j': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )
        # define X-dir frame beams
        sec = sec_collection.retrieve_by_attr('name', sections['lateral_beams'][level_tag])
        for tag1 in ['1', '5']:
            tag2_start = ['B', 'C', 'D']
            tag2_end = ['C', 'D', 'E']
            for j in range(len(tag2_start)):
                mcg.add_horizontal_active(
                    point[tag2_start[j]][tag1][0], point[tag2_start[j]][tag1][1],
                    point[tag2_end[j]][tag1][0], point[tag2_end[j]][tag1][1],
                    np.array((0., 0., 0.)),
                    np.array((0., 0., 0.)),
                    'middle_back',
                    'middle_front',
                    # 'centroid',
                    # 'centroid',
                    'Linear',
                    n_sub,
                    sec,
                    element_type,
                    'top_center',
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': imk_6,
                        'zerolength_gen_args_i': {
                            'lbry': 60.00,
                            'loverh': 0.50,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': imk_6,
                        'zerolength_gen_args_j': {
                            'lbry': 60.00,
                            'loverh': 0.50,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )
        # define Y-dir frame beams
        for tag1 in ['A', 'F']:
            tag2_start = ['2', '3', '4']
            tag2_end = ['3', '4', '5']
            for j in range(len(tag2_start)):
                mcg.add_horizontal_active(
                    point[tag1][tag2_start[j]][0], point[tag1][tag2_start[j]][1],
                    point[tag1][tag2_end[j]][0], point[tag1][tag2_end[j]][1],
                    np.array((0., 0., 0.)),
                    np.array((0., 0., 0.)),
                    'middle_back',
                    'middle_front',
                    # 'centroid',
                    # 'centroid',
                    'Linear',
                    n_sub,
                    sec,
                    element_type,
                    'top_center',
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': imk_6,
                        'zerolength_gen_args_i': {
                            'lbry': 60.00,
                            'loverh': 0.50,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': imk_6,
                        'zerolength_gen_args_j': {
                            'lbry': 60.00,
                            'loverh': 0.50,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )

        # define gravity beams of designation A
        sec = sec_collection.retrieve_by_attr('name', sections['gravity_beams_a'][level_tag])
        for tag1 in ['A', 'F']:
            tag2_start = ['1']
            tag2_end = ['2']
            for j in range(len(tag2_start)):
                mcg.add_horizontal_active(
                    point[tag1][tag2_start[j]][0], point[tag1][tag2_start[j]][1],
                    point[tag1][tag2_end[j]][0], point[tag1][tag2_end[j]][1],
                    np.array((0., 0., 0.)),
                    np.array((0., 0., 0.)),
                    'bottom_center',
                    'top_center',
                    'Linear',
                    n_sub,
                    sec,
                    element_type,
                    'top_center',
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': gravity_shear_tab,
                        'zerolength_gen_args_i': {
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': gravity_shear_tab,
                        'zerolength_gen_args_j': {
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )
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
                mcg.add_horizontal_active(
                    point[tag1][tag2_start[j]][0], point[tag1][tag2_start[j]][1],
                    point[tag1][tag2_end[j]][0], point[tag1][tag2_end[j]][1],
                    np.array((0., 0., 0.)),
                    np.array((0., 0., 0.)),
                    si,
                    sj,
                    'Linear',
                    n_sub,
                    sec,
                    element_type,
                    'top_center',
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': gravity_shear_tab,
                        'zerolength_gen_args_i': {
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': gravity_shear_tab,
                        'zerolength_gen_args_j': {
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )

        # define gravity beams of designation B
        sec = sec_collection.retrieve_by_attr('name', sections['gravity_beams_b'][level_tag])
        mcg.add_horizontal_active(
            point['A']['1'][0], point['A']['1'][1],
            point['B']['1'][0], point['B']['1'][1],
            np.array((0., 0., 0.)),
            np.array((0., 0., 0.)),
            'center_right',
            'middle_front',
            'Linear',
            n_sub,
            sec,
            element_type,
            'top_center',
            method='generate_hinged_component_assembly',
            additional_args={
                'zerolength_gen_i': gravity_shear_tab,
                'zerolength_gen_args_i': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                },
                'zerolength_gen_j': gravity_shear_tab,
                'zerolength_gen_args_j': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                }
            }
        )
        mcg.add_horizontal_active(
            point['E']['1'][0], point['E']['1'][1],
            point['F']['1'][0], point['F']['1'][1],
            np.array((0., 0., 0.)),
            np.array((0., 0., 0.)),
            'middle_back',
            'center_left',
            'Linear',
            n_sub,
            sec,
            element_type,
            'top_center',
            method='generate_hinged_component_assembly',
            additional_args={
                'zerolength_gen_i': gravity_shear_tab,
                'zerolength_gen_args_i': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                },
                'zerolength_gen_j': gravity_shear_tab,
                'zerolength_gen_args_j': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                }
            }
        )
        mcg.add_horizontal_active(
            point['A']['5'][0], point['A']['5'][1],
            point['B']['5'][0], point['B']['5'][1],
            np.array((0., 0., 0.)),
            np.array((0., 0., 0.)),
            'center_right',
            'middle_front',
            'Linear',
            n_sub,
            sec,
            element_type,
            'top_center',
            method='generate_hinged_component_assembly',
            additional_args={
                'zerolength_gen_i': gravity_shear_tab,
                'zerolength_gen_args_i': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                },
                'zerolength_gen_j': gravity_shear_tab,
                'zerolength_gen_args_j': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                }
            }
        )
        mcg.add_horizontal_active(
            point['E']['5'][0], point['E']['5'][1],
            point['F']['5'][0], point['F']['5'][1],
            np.array((0., 0., 0.)),
            np.array((0., 0., 0.)),
            'middle_back',
            'center_left',
            'Linear',
            n_sub,
            sec,
            element_type,
            'top_center',
            method='generate_hinged_component_assembly',
            additional_args={
                'zerolength_gen_i': gravity_shear_tab,
                'zerolength_gen_args_i': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                },
                'zerolength_gen_j': gravity_shear_tab,
                'zerolength_gen_args_j': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                }
            }
        )

        # define gravity beams of designation C
        sec = sec_collection.retrieve_by_attr('name', sections['gravity_beams_c'][level_tag])
        for tag1 in ['2', '3', '4']:
            mcg.add_horizontal_active(
                point['A'][tag1][0], point['A'][tag1][1],
                point['B'][tag1][0], point['B'][tag1][1],
                np.array((0., 0., 0.)),
                np.array((0., 0., 0.)),
                'center_right',
                'center_left',
                'Linear',
                n_sub,
                sec,
                element_type,
                'top_center',
                method='generate_hinged_component_assembly',
                additional_args={
                    'zerolength_gen_i': gravity_shear_tab,
                    'zerolength_gen_args_i': {
                        'consider_composite': True,
                        'section': sec,
                        'physical_material': steel_phys_mat,
                        'distance': 10.00,
                        'n_sub': 1
                    },
                    'zerolength_gen_j': gravity_shear_tab,
                    'zerolength_gen_args_j': {
                        'consider_composite': True,
                        'section': sec,
                        'physical_material': steel_phys_mat,
                        'distance': 10.00,
                        'n_sub': 1
                    }
                }
            )
            mcg.add_horizontal_active(
                point['E'][tag1][0], point['E'][tag1][1],
                point['F'][tag1][0], point['F'][tag1][1],
                np.array((0., 0., 0.)),
                np.array((0., 0., 0.)),
                'center_right',
                'center_left',
                'Linear',
                n_sub,
                sec,
                element_type,
                'top_center',
                method='generate_hinged_component_assembly',
                additional_args={
                    'zerolength_gen_i': gravity_shear_tab,
                    'zerolength_gen_args_i': {
                        'consider_composite': True,
                        'section': sec,
                        'physical_material': steel_phys_mat,
                        'distance': 10.00,
                        'n_sub': 1
                    },
                    'zerolength_gen_j': gravity_shear_tab,
                    'zerolength_gen_args_j': {
                        'consider_composite': True,
                        'section': sec,
                        'physical_material': steel_phys_mat,
                        'distance': 10.00,
                        'n_sub': 1
                    }
                }
            )

        # define gravity beams of designation D
        sec = sec_collection.retrieve_by_attr('name', sections['gravity_beams_d'][level_tag])
        for tag1 in ['2', '3', '4']:
            tag2_start = ['B', 'C', 'D']
            tag2_end = ['C', 'D', 'E']
            for j in range(len(tag2_start)):
                mcg.add_horizontal_active(
                    point[tag2_start[j]][tag1][0], point[tag2_start[j]][tag1][1],
                    point[tag2_end[j]][tag1][0], point[tag2_end[j]][tag1][1],
                    np.array((0., 0., 0.)),
                    np.array((0., 0., 0.)),
                    'center_right',
                    'center_left',
                    'Linear',
                    n_sub,
                    sec,
                    element_type,
                    'top_center',
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': gravity_shear_tab,
                        'zerolength_gen_args_i': {
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': gravity_shear_tab,
                        'zerolength_gen_args_j': {
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )

    # fix base
    for node in mdl.levels.registry[0].nodes.registry.values():
        node.restraint = [True]*6

    # ~~~~~~~~~~~~ #
    # assign loads #
    # ~~~~~~~~~~~~ #

    loadcase = LoadCase('1.2D+0.25L+-E', mdl)
    self_weight(mdl, loadcase, factor=1.20)
    self_mass(mdl, loadcase)

    # surface loads
    for key in range(1, 1+num_levels):
        loadcase.tributary_area_analysis.registry[key].polygon_loads.append(
            PolygonLoad('dead', surf_loads[key], None, None, False))
        loadcase.tributary_area_analysis.registry[key].polygon_loads.append(
            PolygonLoad('dead', surf_loads_massless[key], None, None, True))
        loadcase.tributary_area_analysis.registry[key].run(
            load_factor=1.20,
            massless_load_factor=0.25)

    # cladding loads
    def apply_cladding_load(coords, surf_load, surf_area, factor, massless=False):
        subset_model = mdl.initialize_empty_copy('subset_1')
        mdl.transfer_by_polygon_selection(subset_model, coords)
        # show(subset_model)
        elms = {}
        elm_lens = {}
        for comp in subset_model.list_of_components():
            if comp.component_purpose != 'steel_W_panel_zone':
                for elm in comp.list_of_elastic_beamcolumn_elements() + comp.list_of_disp_beamcolumn_elements():
                    elms[elm.uid] = elm
                    elm_lens[elm.uid] = elm.clear_length()
        len_tot = sum(elm_lens.values())
        load = surf_load * surf_area
        line_load = load/len_tot
        from osmg.common import G_CONST_IMPERIAL
        for key, elm in elms.items():
            half_mass = line_load * elm_lens[key] / G_CONST_IMPERIAL
            loadcase.line_element_udl.registry[key].add_glob(np.array((0.00, 0.00, -line_load*factor)))
            loadcase.node_mass.registry[elm.eleNodes[0].uid].add([half_mass]*3+[0.00]*3)

    apply_cladding_load(
        np.array(
            [
                [-10.00, -50.00],
                [+1690.00, -50.00],
                [+1690.00, +50.00],
                [-10.00, +50.00]
            ]
        ),
        15.00/12.00**2,
        140.00*(15.00+13.00+13.00)*12.00**2,
        1.2
    )
    apply_cladding_load(
        np.array(
            [
                [-10.00, +1200.00-50.00],
                [+1690.00, +1200.00-50.00],
                [+1690.00, +1200.00+50.00],
                [-10.00, +1200.00+50.00]
            ]
        ),
        15.00/12.00**2,
        140.00*(15.00+13.00+13.00)*12.00**2,
        1.2
    )
    apply_cladding_load(
        np.array(
            [
                [-50.00, -50.00],
                [-50.00, 1250.00],
                [+50.00, 1250.00],
                [+50.00, -50.00]
            ]
        ),
        15.00/12.00**2,
        100.00*(15.00+13.00+13.00)*12.00**2,
        1.2
    )
    apply_cladding_load(
        np.array(
            [
                [+1680.00-50.00, -50.00],
                [+1680.00-50.00, 1250.00],
                [+1680.00+50.00, 1250.00],
                [+1680.00+50.00, -50.00]
            ]
        ),
        15.00/12.00**2,
        100.00*(15.00+13.00+13.00)*12.00**2,
        1.2
    )
    loadcase.rigid_diaphragms([i for i in range(1, num_levels+1)], gather_mass=True)
    return mdl, loadcase


def smrf_3_of_II():

    heights = np.array(
        (15.00,
         13.00+15.00,
         13.00+13.00+15.00)) * 12.00

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
                level_1="W18X119",
                level_2="W18X119",
                level_3="W18X106"),
            interior=dict(
                level_1="W24X176",
                level_2="W24X176",
                level_3="W24X162")),
        lateral_beams=dict(
            level_1="W30X132",
            level_2="W30X124",
            level_3="W21X50")
        )

    doubler_plate_thicknesses = dict(
        exterior=dict(
            level_1=0.719992,
            level_2=0.628111,
            level_3=0.000000
        ),
        interior=dict(
            level_1=1.332163,
            level_2=1.193641,
            level_3=0.007416
        )
    )

    surf_loads = {
        1: (63.+15.+15.)/(12.**2) + 0.0184524,
        2: (63.+15.+15.)/(12.**2) + 0.0184524,
        3: (63.+15.+80.*0.26786)/(12.**2) + 0.0184524
    }
    surf_loads_massless = {
        1: 50.00/(12.**2),
        2: 50.00/(12.**2),
        3: 20.00/(12.**2)
    }

    mdl, loadcase = generate_smrf_archetype(
        level_elevs=heights,
        sections=sections,
        doubler_plate_thicknesses=doubler_plate_thicknesses,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless)

    return mdl, loadcase


def smrf_3_of_IV():

    heights = np.array(
        (15.00,
         13.00+15.00,
         13.00+13.00+15.00)) * 12.00

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
                level_1="W21X147",
                level_2="W21X147",
                level_3="W21X122"),
            interior=dict(
                level_1="W27X235",
                level_2="W27X217",
                level_3="W27X194")),
        lateral_beams=dict(
            level_1="W33X141",
            level_2="W33X141",
            level_3="W21X62")
        )

    doubler_plate_thicknesses = dict(
        exterior=dict(
            level_1=0.539838,
            level_2=0.539838,
            level_3=0.000000
        ),
        interior=dict(
            level_1=1.018482,
            level_2=1.134977,
            level_3=0.038376
        )
    )

    surf_loads = {
        1: (63.+15.+15.)/(12.**2) + 0.0184524,
        2: (63.+15.+15.)/(12.**2) + 0.0184524,
        3: (63.+15.+80.*0.26786)/(12.**2) + 0.0184524
    }
    surf_loads_massless = {
        1: 50.00/(12.**2),
        2: 50.00/(12.**2),
        3: 20.00/(12.**2)
    }

    mdl, loadcase = generate_smrf_archetype(
        level_elevs=heights,
        sections=sections,
        doubler_plate_thicknesses=doubler_plate_thicknesses,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless)

    return mdl, loadcase


def smrf_6_of_II():

    heights = np.array(
        (15.00,
         13.00+15.00,
         13.00*2.00+15.00,
         13.00*3.00+15.00,
         13.00*4.00+15.00,
         13.00*5.00+15.00)
    ) * 12.00

    sections = dict(
        gravity_cols=dict(
            level_1="W14X90",
            level_2="W14X90",
            level_3="W14X90",
            level_4="W14X90",
            level_5="W14X90",
            level_6="W14X90"),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31"),
        gravity_beams_b=dict(
            level_1="W21X44",
            level_2="W21X44",
            level_3="W21X44",
            level_4="W21X44",
            level_5="W21X44",
            level_6="W21X44"),
        gravity_beams_c=dict(
            level_1="W24X62",
            level_2="W24X62",
            level_3="W24X62",
            level_4="W24X62",
            level_5="W24X62",
            level_6="W24X62"),
        gravity_beams_d=dict(
            level_1="W21X44",
            level_2="W21X44",
            level_3="W21X44",
            level_4="W21X44",
            level_5="W21X44",
            level_6="W21X48"),
        gravity_beams_e=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31"),
        lateral_cols=dict(
            exterior=dict(
                level_1="W24X162",
                level_2="W24X162",
                level_3="W24X146",
                level_4="W24X131",
                level_5="W24X94",
                level_6="W24X84"),
            interior=dict(
                level_1="W30X211",
                level_2="W30X211",
                level_3="W30X211",
                level_4="W30X211",
                level_5="W30X148",
                level_6="W30X132")),
        lateral_beams=dict(
            level_1="W33X169",
            level_2="W33X169",
            level_3="W30X173",
            level_4="W30X148",
            level_5="W30X116",
            level_6="W21X50")
        )

    doubler_plate_thicknesses = dict(
        exterior=dict(
            level_1=1.079849,
            level_2=1.079849,
            level_3=1.138900,
            level_4=1.745616,
            level_5=1.845505,
            level_6=1.204157
        ),
        interior=dict(
            level_1=1.739763,
            level_2=1.739763,
            level_3=1.856444,
            level_4=2.501809,
            level_5=2.093302,
            level_6=1.154478
        )
    )

    surf_loads = {
        1: (63.+15.+15.)/(12.**2) + 0.0184524,
        2: (63.+15.+15.)/(12.**2) + 0.0184524,
        3: (63.+15.+15.)/(12.**2) + 0.0184524,
        4: (63.+15.+15.)/(12.**2) + 0.0184524,
        5: (63.+15.+15.)/(12.**2) + 0.0184524,
        6: (63.+15.+80.*0.26786)/(12.**2) + 0.0184524
    }
    surf_loads_massless = {
        1: 50.00/(12.**2),
        2: 50.00/(12.**2),
        3: 50.00/(12.**2),
        4: 50.00/(12.**2),
        5: 50.00/(12.**2),
        6: 20.00/(12.**2)
    }

    mdl, loadcase = generate_smrf_archetype(
        level_elevs=heights,
        sections=sections,
        doubler_plate_thicknesses=doubler_plate_thicknesses,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless)

    return mdl, loadcase


def smrf_6_of_IV():

    heights = np.array(
        (15.00,
         13.00+15.00,
         13.00*2.00+15.00,
         13.00*3.00+15.00,
         13.00*4.00+15.00,
         13.00*5.00+15.00)
    ) * 12.00

    sections = dict(
        gravity_cols=dict(
            level_1="W14X90",
            level_2="W14X90",
            level_3="W14X90",
            level_4="W14X90",
            level_5="W14X90",
            level_6="W14X90"),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31"),
        gravity_beams_b=dict(
            level_1="W21X44",
            level_2="W21X44",
            level_3="W21X44",
            level_4="W21X44",
            level_5="W21X44",
            level_6="W21X44"),
        gravity_beams_c=dict(
            level_1="W24X62",
            level_2="W24X62",
            level_3="W24X62",
            level_4="W24X62",
            level_5="W24X62",
            level_6="W24X62"),
        gravity_beams_d=dict(
            level_1="W21X44",
            level_2="W21X44",
            level_3="W21X44",
            level_4="W21X44",
            level_5="W21X44",
            level_6="W21X48"),
        gravity_beams_e=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31"),
        lateral_cols=dict(
            exterior=dict(
                level_1="W24X229",
                level_2="W24X229",
                level_3="W24X207",
                level_4="W24X176",
                level_5="W24X162",
                level_6="W24X76"),
            interior=dict(
                level_1="W30X326",
                level_2="W30X326",
                level_3="W30X326",
                level_4="W30X292",
                level_5="W30X235",
                level_6="W30X108")),
        lateral_beams=dict(
            level_1="W33X263",
            level_2="W33X263",
            level_3="W30X235",
            level_4="W30X211",
            level_5="W30X132",
            level_6="W30X124")
        )

    doubler_plate_thicknesses = dict(
        exterior=dict(
            level_1=1.037822,
            level_2=1.037822,
            level_3=0.935894,
            level_4=0.926930,
            level_5=0.313647,
            level_6=0.615541
        ),
        interior=dict(
            level_1=2.103138,
            level_2=2.103138,
            level_3=1.723039,
            level_4=1.602846,
            level_5=0.812174,
            level_6=1.155613
        )
    )

    surf_loads = {
        1: (63.+15.+15.)/(12.**2) + 0.0184524,
        2: (63.+15.+15.)/(12.**2) + 0.0184524,
        3: (63.+15.+15.)/(12.**2) + 0.0184524,
        4: (63.+15.+15.)/(12.**2) + 0.0184524,
        5: (63.+15.+15.)/(12.**2) + 0.0184524,
        6: (63.+15.+80.*0.26786)/(12.**2) + 0.0184524
    }
    surf_loads_massless = {
        1: 50.00/(12.**2),
        2: 50.00/(12.**2),
        3: 50.00/(12.**2),
        4: 50.00/(12.**2),
        5: 50.00/(12.**2),
        6: 20.00/(12.**2)
    }

    mdl, loadcase = generate_smrf_archetype(
        level_elevs=heights,
        sections=sections,
        doubler_plate_thicknesses=doubler_plate_thicknesses,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless)

    return mdl, loadcase


def smrf_9_of_II():

    heights = np.array(
        (15.00,
         13.00+15.00,
         13.00*2.00+15.00,
         13.00*3.00+15.00,
         13.00*4.00+15.00,
         13.00*5.00+15.00,
         13.00*6.00+15.00,
         13.00*7.00+15.00,
         13.00*8.00+15.00)
    ) * 12.00

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
            level_9="W14X90"),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31"),
        gravity_beams_b=dict(
            level_1="W21X44",
            level_2="W21X44",
            level_3="W21X44",
            level_4="W21X44",
            level_5="W21X44",
            level_6="W21X44",
            level_7="W21X44",
            level_8="W21X44",
            level_9="W21X44"),
        gravity_beams_c=dict(
            level_1="W24X62",
            level_2="W24X62",
            level_3="W24X62",
            level_4="W24X62",
            level_5="W24X62",
            level_6="W24X62",
            level_7="W24X62",
            level_8="W24X62",
            level_9="W24X62"),
        gravity_beams_d=dict(
            level_1="W21X44",
            level_2="W21X44",
            level_3="W21X44",
            level_4="W21X44",
            level_5="W21X44",
            level_6="W21X44",
            level_7="W21X44",
            level_8="W21X44",
            level_9="W21X48"),
        gravity_beams_e=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31"),
        lateral_cols=dict(
            exterior=dict(
                level_1="W24X279",
                level_2="W24X279",
                level_3="W24X279",
                level_4="W24X279",
                level_5="W24X279",
                level_6="W24X279",
                level_7="W24X162",
                level_8="W24X162",
                level_9="W24X131"),
            interior=dict(
                level_1="W36X361",
                level_2="W36X361",
                level_3="W36X361",
                level_4="W36X361",
                level_5="W36X361",
                level_6="W36X361",
                level_7="W36X256",
                level_8="W36X256",
                level_9="W36X182")),
        lateral_beams=dict(
            level_1="W33X318",
            level_2="W33X318",
            level_3="W33X291",
            level_4="W33X291",
            level_5="W30X292",
            level_6="W30X292",
            level_7="W27X217",
            level_8="W27X217",
            level_9="W21X44")
        )

    doubler_plate_thicknesses = dict(
        exterior=dict(
            level_1=1.146050,
            level_2=1.146050,
            level_3=0.956666,
            level_4=0.956666,
            level_5=0.917400,
            level_6=0.917400,
            level_7=1.050302,
            level_8=1.050302,
            level_9=0.000000
        ),
        interior=dict(
            level_1=2.229881,
            level_2=2.229881,
            level_3=1.965003,
            level_4=1.965003,
            level_5=1.919496,
            level_6=1.919496,
            level_7=1.392831,
            level_8=1.392831,
            level_9=0.000000
        )
    )

    surf_loads = {
        1: (63.+15.+15.)/(12.**2) + 0.0184524,
        2: (63.+15.+15.)/(12.**2) + 0.0184524,
        3: (63.+15.+15.)/(12.**2) + 0.0184524,
        4: (63.+15.+15.)/(12.**2) + 0.0184524,
        5: (63.+15.+15.)/(12.**2) + 0.0184524,
        6: (63.+15.+15.)/(12.**2) + 0.0184524,
        7: (63.+15.+15.)/(12.**2) + 0.0184524,
        8: (63.+15.+15.)/(12.**2) + 0.0184524,
        9: (63.+15.+80.*0.26786)/(12.**2) + 0.0184524
    }
    surf_loads_massless = {
        1: 50.00/(12.**2),
        2: 50.00/(12.**2),
        3: 50.00/(12.**2),
        4: 50.00/(12.**2),
        5: 50.00/(12.**2),
        6: 50.00/(12.**2),
        7: 50.00/(12.**2),
        8: 50.00/(12.**2),
        9: 20.00/(12.**2)
    }

    mdl, loadcase = generate_smrf_archetype(
        level_elevs=heights,
        sections=sections,
        doubler_plate_thicknesses=doubler_plate_thicknesses,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless)

    return mdl, loadcase
