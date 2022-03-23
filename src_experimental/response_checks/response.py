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


ground_motion_dir = 'src_experimental/response_checks/gm_input'
data_output_dir = 'src_experimental/response_checks/out'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# generate ground motion input #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# sine wave ground acceleration

gm_amplitude = 1.00  # g
gm_period = 0.945  # s
gm_tmax = 10.  # s
gm_dt = 0.005
gm_t_vec = np.arange(start=0.00, stop=gm_tmax+1e-5, step=gm_dt)
gm_a = gm_amplitude * np.sin(2. * np.pi / gm_period * gm_t_vec)

# # plot ground motion
# fig, ax = plt.subplots()
# ax.plot(gm_t_vec, gm_a)
# fig.show()

# write to a file
np.savetxt(f"{ground_motion_dir}/x.txt", gm_a)


# pulse

gm_amplitude = 0.20  # g
gm_period = 0.945  # s
gm_tmax = 8.  # s
gm_dt = 0.005
gm_t_vec = np.arange(start=0.00, stop=gm_tmax+1e-5, step=gm_dt)
gm_a = gm_amplitude * np.sin(2. * np.pi / gm_period * gm_t_vec)
idx = int(gm_period / gm_dt / 2.)
gm_a[idx::] = 0.00

# # plot ground motion
# fig, ax = plt.subplots()
# ax.plot(gm_t_vec, gm_a)
# fig.show()

# write to a file
np.savetxt(f"{ground_motion_dir}/x.txt", gm_a)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run the analysis             #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# ~~~~~~~~~~ #
# parameters #
# ~~~~~~~~~~ #

# fundamental period of the building
t_1 = 0.945

analysis_dt = 0.01  # s

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
    # nid = anal_obj.building.list_of__nodes()[floor].uniq_id
    nid = anal_obj.building.levels.level_list[floor+1].nodes_primary.node_list[0].uniq_id
    d = []
    for i in range(n_steps):
        d.append(anal_obj.node_displacements[nid][i][drct])
    d = np.array(d)
    t = np.array(anal_obj.time_vector)
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

# nsub = 15  # element subdivision
nsub = 1  # element subdivision
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























































# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# nlth analysis                #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# define analysis object
nlth = solver.NLTHAnalysis(b)

# get the corresponding ground motion duration
gm_X_filepath = f"{ground_motion_dir}/x.txt"
gm_Y_filepath = None
gm_Z_filepath = None

# run the nlth analysis
metadata = nlth.run(analysis_dt,
                    gm_X_filepath,
                    gm_Y_filepath,
                    gm_Z_filepath,
                    gm_dt,
                    finish_time=4.00,
                    damping_ratio=0.03,
                    num_modes=100,
                    printing=True,
                    data_retention='default')

if not metadata['analysis_finished_successfully']:
    print('Analysis failed.')


response_th_1 = retrieve_displacement_th(0, 0, nlth)
response_th_2 = retrieve_displacement_th(1, 0, nlth)
response_th_3 = retrieve_displacement_th(2, 0, nlth)

# # # save
# # np.savetxt(f'{data_output_dir}/th1-init3-noyield.txt', response_th_1)
# # np.savetxt(f'{data_output_dir}/th2-init3-noyield.txt', response_th_2)
# # np.savetxt(f'{data_output_dir}/th3-init3-noyield.txt', response_th_3)



fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
# plt.style.use('tableau-colorblind10')
fig.suptitle('Pulse excitation')
ax1.plot(gm_t_vec, gm_a)
ax1.set_ylabel('Ground Acceleration (g)')
ax2.plot(response_th_1[:, 0], response_th_1[:, 1], label='1st lvl')
ax2.plot(response_th_2[:, 0], response_th_2[:, 1], label='2nd lvl')
ax2.plot(response_th_3[:, 0], response_th_3[:, 1], label='3rd lvl')
ax2.legend()
ax2.set_ylabel('Floor Displacement (in)')
ax2.set_xlabel('Time (s)')
# plt.savefig(f'{data_output_dir}/pulse_collapse.pdf')
fig.show()

# resp = response_th_3[:, 1]
# local_max = []
# local_maxt = []
# for i in range(2, len(resp)):
#     if resp[i-2] < resp[i-1] and resp[i-1] > resp[i]:
#         local_max.append(resp[i-1])
#         local_maxt.append(response_th_3[i-1, 0])
# # plt.plot(response_th_1[:, 0], resp)
# # plt.scatter(local_maxt, local_max)
# # plt.show()

# zeta_est = 1./(2.*np.pi*(len(local_max)-1.))*np.log(local_max[0]/local_max[-1])
# print(zeta_est)

# # plot the deformed shape for any of the steps
# plot_metadata = nlth.deformed_shape(
#     step=metadata['successful steps'] - 1, scaling=1.00, extrude_frames=True)
# print(plot_metadata)



# response_th_comm3 = np.genfromtxt(f'{data_output_dir}/th3-comm3-noyield.txt')
# response_th_tang3 = np.genfromtxt(f'{data_output_dir}/th3-tang3-noyield.txt')
# response_th_init3 = np.genfromtxt(f'{data_output_dir}/th3-init3-noyield.txt')
# response_th_no3 = np.genfromtxt(f'{data_output_dir}/th3-no3-noyield.txt')
# response_th_ray3 = np.genfromtxt(f'{data_output_dir}/th3-ray3-noyield.txt')

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
# # plt.style.use('tableau-colorblind10')
# fig.suptitle('Pulse excitation')
# ax1.plot(gm_t_vec, gm_a)
# ax1.set_ylabel('Ground Acceleration (g)')
# ax2.plot(response_th_comm3[:, 0], response_th_comm3[:, 1], label='3rd lvl, commited')
# ax2.plot(response_th_init3[:, 0], response_th_init3[:, 1], label='3rd lvl, initial')
# ax2.plot(response_th_tang3[:, 0], response_th_tang3[:, 1], label='3rd lvl, tangent', linestyle='dashed')
# ax2.plot(response_th_ray3[:, 0], response_th_ray3[:, 1], label='3rd lvl, rayleigh')
# ax2.legend()
# ax2.set_ylabel('Floor Displacement (in)')
# ax2.set_xlabel('Time (s)')
# # plt.savefig(f'{data_output_dir}/pulse-noyield-damping.pdf')
# fig.show()





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # modal analysis               #
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #



# # performing a linear modal analysis
# modal_analysis = solver.ModalAnalysis(b, num_modes=3)
# modal_analysis.run()

# modal_analysis.deformed_shape(step=2, scaling=0.00, extrude_frames=False)

# # retrieving textual results
# print(modal_analysis.periods)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# pushover analysis            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# # performing a nonlinear pushover analysis
# pushover_analysis = solver.PushoverAnalysis(b)
# control_node = b.list_of_parent_nodes()[-1]  # top floor
# analysis_metadata = pushover_analysis.run(
#     "x",
#     np.array([80]),
#     control_node,
#     1./1., modeshape=np.array([0., 0.31, 0.69, 1.0]))

# # plot the deformed shape for any of the steps
# n_plot_steps = analysis_metadata['successful steps']
# # plot_metadata = pushover_analysis.deformed_shape(
# #     step=n_plot_steps-1, scaling=0.00, extrude_frames=True)
# # print(plot_metadata)
 
# # # plot pushover curve
# # pushover_analysis.plot_pushover_curve("x", control_node)

# deltas, vbs = pushover_analysis.table_pushover_curve('x', control_node)

# for i in range(len(vbs)):
#     print(deltas[i]/12./41.)

# y_axis = vbs / (0.9 * 6569.3 / 386.22) / 2.34e5
# x_axis = deltas / 7.626

# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='medium')
# plt.rc('ytick', labelsize='medium')
# plt.rc('text', usetex=True)

# plt.figure()
# plt.grid()
# plt.plot(x_axis, y_axis, color='k',
#          ls='dashed', label='IMK, composite all beams')
# plt.ylabel('Spectral Acceleration index')
# plt.xlabel('Ductility $\mu$')
# plt.legend()
# # plt.savefig("pushover_normalized.pdf")
# plt.show()





# seismic_weight = np.sum(b.level_masses() * 386.22 / 1.e3)  # (kips)
# total_height = (15. + 13.*2) * 12.00
# vbs /= seismic_weight * 1000.00
# deltas /= total_height

# cs = 0.15115
# omEga = 3.00


# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='medium')
# plt.rc('ytick', labelsize='medium')
# plt.rc('text', usetex=True)

# plt.figure()
# plt.grid()
# plt.axhline(y=cs, color='0.50', ls='dashed')
# plt.axhline(y=cs*omEga, color='0.50', ls='dashed')
# plt.plot(deltas, vbs, color='k',
#          ls='dashed', label='IMK, composite all beams')
# plt.ylabel('Vb / W')
# plt.xlabel('Roof Drift Ratio $\\Delta$/H')
# plt.legend()
# plt.show()





















































# # beams
# from components import LineElementSequence_Steel_W_PanelZone
# from components import LineElementSequence_Steel_W_PanelZone_IMK
# from components import LineElementSequence_W_grav_sear_tab
# from components import LineElementSequence_IMK
# imk_springs = []
# seqs = b.list_of_line_element_sequences()
# for seq in seqs:
#     if isinstance(seq, LineElementSequence_IMK):
#         imk_springs.append(seq)

# len(imk_springs)
# springs = []
# for seq in imk_springs:
#     springs.append(seq.end_segment_i.internal_elems[1])
#     springs.append(seq.end_segment_j.internal_elems[0])

# jmax = 0
# ult_max = 0.00

# for j, spring in enumerate(springs):
#     data = []
#     for i in range(nlth.n_steps_success):
#         data.append(
#             nlth.release_force_defo[spring.uniq_id][i]
#         )

#     data = np.array(data)
#     curr_max = np.max(np.abs(data[:, 1]))
#     if curr_max > ult_max:
#         ult_max = curr_max
#         jmax = j

# spring = springs[jmax]
# data = []
# for i in range(nlth.n_steps_success):
#     data.append(
#         nlth.release_force_defo[spring.uniq_id][i]
#     )

# data = np.array(data)

# curr_max = np.max(np.abs(data[:, 1]))

# mat_properties = spring.materials[-1].parameters

# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='medium')
# plt.rc('ytick', labelsize='medium')
# # plt.rc('text', usetex=True)
# plt.figure(figsize=(6, 6))
# plt.grid()
# # plt.axvline(x=mat_properties['theta_p+'], color='red', label='theta_p')
# # plt.axvline(x=-mat_properties['theta_p-'], color='red')
# # plt.axvline(x=mat_properties['theta_p+']+mat_properties['theta_pc+'],
# #             color='green', label='theta_pc + theta_p')
# # plt.axvline(x=-mat_properties['theta_p-']-mat_properties['theta_pc-'],
# #             color='green')
# # plt.axvline(x=mat_properties['theta_u'], color='purple', label='theta_u')
# # plt.axvline(x=-mat_properties['theta_u'], color='purple')
# # plt.axhline(y=mat_properties['my+']/1e3/12., color='cyan', label='my+')
# # plt.axhline(y=mat_properties['my-']/1e3/12., color='cyan')
# # plt.axhline(y=mat_properties['residual_plus'] / 1e3 / 12. *
# #             mat_properties['my+'], color='purple', label='residual moment')
# # plt.axhline(y=mat_properties['residual_minus'] / 1e3 / 12. *
# #             mat_properties['my-'], color='purple')
# plt.plot(data[:, 1], data[:, 0]/(1.e3 * 12.), color='blue', alpha=0.5)
# plt.ylabel('Moment (kip-ft)')
# plt.xlabel('Rotation (rad)')
# plt.title("RBS Beam (Modified IMK Model)")
# plt.show()
# # plt.savefig('beam.pdf')

























# # ~~~~~~~~~~~~~~~~~~ #
# # Panel zone springs #
# # ~~~~~~~~~~~~~~~~~~ #


# panel_zone_springs = []
# seqs = b.list_of_line_element_sequences()
# for seq in seqs:
#     if isinstance(seq, LineElementSequence_Steel_W_PanelZone):
#         panel_zone_springs.append(seq)
#     if isinstance(seq, LineElementSequence_Steel_W_PanelZone_IMK):
#         panel_zone_springs.append(seq)

# len(panel_zone_springs)
# springs = [spring.end_segment_i.internal_elems[8]
#            for spring in panel_zone_springs]

# spring = springs[14]

# data = []
# for i in range(nlth.n_steps_success):
#     data.append(
#         nlth.release_force_defo[spring.uniq_id][i]
#     )

# data = np.array(data)


# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='medium')
# plt.rc('ytick', labelsize='medium')
# # plt.rc('text', usetex=True)
# plt.figure(figsize=(6, 6))
# plt.grid()
# plt.plot(data[:, 1], data[:, 0], color='blue', alpha=0.5)
# plt.ylabel('Moment (lb-in)')
# plt.xlabel('Rotation (rad)')
# plt.title("Panel Zone (Hysteretic Model)")
# plt.show()
# # plt.savefig('panel_zone.pdf')











































# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# # gravity shear tab connection #
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# grav_springs = []
# seqs = nlth.building.list_of_line_element_sequences()
# for seq in seqs:
#     if isinstance(seq, LineElementSequence_W_grav_sear_tab):
#         grav_springs.append(seq)

# len(grav_springs)
# springs = []
# for seq in grav_springs:
#     springs.append(seq.end_segment_i.internal_elems[1])
#     springs.append(seq.end_segment_j.internal_elems[0])

# jmax = 0
# ult_max = 0.00

# for j, spring in enumerate(springs):
#     data = []
#     for i in range(nlth.n_steps_success):
#         data.append(
#             nlth.release_force_defo[spring.uniq_id][i]
#         )

#     data = np.array(data)
#     curr_max = np.max(np.abs(data[:, 0]))
#     if curr_max > ult_max:
#         ult_max = curr_max
#         jmax = j

# spring = springs[jmax]
# data = []
# for i in range(nlth.n_steps_success):
#     data.append(
#         nlth.release_force_defo[spring.uniq_id][i]
#     )

# data = np.array(data)

# curr_max = np.max(np.abs(data[:, 1]))

# mat_properties = spring.materials[-1].parameters

# backbone_x = [2*mat_properties['th_4_n'],
#               mat_properties['th_4_n'],
#               mat_properties['th_3_n'],
#               mat_properties['th_2_n'],
#               mat_properties['th_1_n'],
#               0.00,
#               mat_properties['th_1_p'],
#               mat_properties['th_2_p'],
#               mat_properties['th_3_p'],
#               mat_properties['th_4_p'],
#               2. * mat_properties['th_4_p']]
# backbone_y = [mat_properties['m4_n'],
#               mat_properties['m4_n'],
#               mat_properties['m3_n'],
#               mat_properties['m2_n'],
#               mat_properties['m1_n'],
#               0.00,
#               mat_properties['m1_p'],
#               mat_properties['m2_p'],
#               mat_properties['m3_p'],
#               mat_properties['m4_p'],
#               mat_properties['m4_p']]

# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='medium')
# plt.rc('ytick', labelsize='medium')
# plt.figure(figsize=(8, 4))
# plt.grid()
# plt.plot(data[:, 1], data[:, 0], color='blue',
#          ls='solid', alpha=0.5)
# plt.scatter(data[-1, 1], data[-1, 0])
# plt.plot(backbone_x, backbone_y, 'red', ls='dashed', label='backbone')
# # plt.axhline(y=sec_mp, label='Mult')
# # plt.axhline(y=-sec_mp)
# plt.ylabel('Moment $M$ (kip-in)')
# plt.xlabel('Rotation $\phi$ (rad)')
# plt.legend()
# plt.title("Gravity Shear Tab Conn. (Pinching Model)")
# plt.show()
# # plt.savefig('grav.pdf')
# plt.close()











































    
# # ~~~~~~~~~~~~~~~~ #
# # collect response #
# # ~~~~~~~~~~~~~~~~ #


# # ground acceleration, velocity and displacement
# # interpolation functions

# ag = {}  # g units
# ag[0] = np.genfromtxt(gm_X_filepath)
# ag[1] = np.genfromtxt(gm_Y_filepath)
# n_pts = len(ag[0])
# t = np.linspace(0.00, ground_motion_dt*n_pts, n_pts)
# vg = {}  # in/s units
# vg[0] = integrate.cumulative_trapezoid(
#     ag[0]*modeler.common.G_CONST, t, initial=0)
# vg[1] = integrate.cumulative_trapezoid(
#     ag[1]*modeler.common.G_CONST, t, initial=0)
# dg = {}  # in units
# dg[0] = integrate.cumulative_trapezoid(vg[0], t, initial=0)
# dg[1] = integrate.cumulative_trapezoid(vg[1], t, initial=0)

# fag = {}
# fag[0] = interp1d(t, ag[0], bounds_error=False, fill_value=0.00)
# fag[1] = interp1d(t, ag[1], bounds_error=False, fill_value=0.00)
# fvg = {}
# fvg[0] = interp1d(t, vg[0], bounds_error=False, fill_value=0.00)
# fvg[1] = interp1d(t, vg[1], bounds_error=False, fill_value=0.00)
# fdg = {}
# fdg[0] = interp1d(t, dg[0], bounds_error=False, fill_value=0.00)
# fdg[1] = interp1d(t, dg[1], bounds_error=False, fill_value=0.00)


# if not os.path.exists(output_folder):
#     os.mkdir(output_folder)

# time_vec = np.array(nlth.time_vector)
# np.savetxt(f'{output_folder}/time.csv', time_vec)

# for direction in range(2):
#     # store response time-histories
#     np.savetxt(f'{output_folder}/FA-0-{direction+1}.csv',
#                fag[direction](time_vec))
#     for lvl in range(num_levels):
#         # story drifts
#         if lvl == 0:
#             u = retrieve_displacement_th(lvl, direction, nlth)
#             dr = u[:, 1] / level_heights[lvl]
#         else:
#             uprev = retrieve_displacement_th(lvl-1, direction, nlth)
#             u = retrieve_displacement_th(lvl, direction, nlth)
#             dr = (u[:, 1] - uprev[:, 1]) / level_heights[lvl]
#         # story accelerations
#         a1 = retrieve_acceleration_th(lvl, direction, nlth)
#         # story velocities
#         vel = retrieve_velocity_th(lvl, direction, nlth)

#         np.savetxt(f'{output_folder}/ID-{lvl+1}-{direction+1}.csv', dr)
#         np.savetxt(f'{output_folder}/FA-{lvl+1}-{direction+1}.csv',
#                    a1[:, 1]/modeler.common.G_CONST + fag[direction](a1[:, 0]))
#         np.savetxt(f'{output_folder}/FV-{lvl+1}-{direction+1}.csv',
#                    vel[:, 1] + fvg[direction](vel[:, 0]))

#     # global building drift
#     bdr = retrieve_displacement_th(num_levels-1, direction, nlth)
#     bdr[:, 1] /= np.sum(level_heights)
#     np.savetxt(f'{output_folder}/BD-{direction+1}.csv', dr)

