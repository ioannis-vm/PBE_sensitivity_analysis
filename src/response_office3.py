import sys
sys.path.append("../OpenSees_Model_Generator/src")
sys.path.append("src")

import numpy as np
from scipy import integrate
import model
import solver
from archetypes import smrf_3_of_II
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
parser.add_argument('--archetype')
parser.add_argument('--gm_dir')
parser.add_argument('--gm_dt')
parser.add_argument('--analysis_dt')
parser.add_argument('--gm_number')
parser.add_argument('--output_dir')

args = parser.parse_args()
archetype = args.archetype
ground_motion_dir = args.gm_dir
ground_motion_dt = float(args.gm_dt)
analysis_dt = float(args.analysis_dt)
gm_number = int(args.gm_number.replace('gm', ''))
output_folder = args.output_dir

# archetype = 'smrf_3_of_II'
# ground_motion_dir = 'analysis/smrf_3_of_II/hazard_level_8/ground_motions'
# ground_motion_dt = 0.005
# analysis_dt = 0.001
# gm_number = 1
# output_folder = '/tmp/TEST'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# load archetype building here
if archetype == 'smrf_3_of_II':
    b = smrf_3_of_II()
else:
    raise ValueError(f'Unknown archetype code: {archetype}')


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


def get_duration(time_history_path, dt):
    """
    Get the duration of a fixed-step time-history
    stored in a text file, given its path and
    the time increment.
    """
    values = np.genfromtxt(time_history_path)
    num_points = len(values)
    return float(num_points) * dt

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


for thing in dir(nlth):
    print(thing)
    print(getattr(nlth, thing))
    print()
    print()

nlth.basic_forces
    
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
