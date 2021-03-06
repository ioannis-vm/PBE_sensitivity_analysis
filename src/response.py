import sys
sys.path.append("src")

import numpy as np
from scipy import integrate
from archetypes import smrf_3_of_II
from archetypes import smrf_6_of_II
from archetypes import smrf_9_of_II
from archetypes import smrf_3_of_IV
from archetypes import smrf_6_of_IV
from osmg import solver
from util import read_study_param
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

# archetype = 'smrf_9_of_II'
# ground_motion_dir = 'analysis/smrf_3_of_II/hazard_level_16/ground_motions'
# ground_motion_dt = 0.005
# analysis_dt = 0.001
# gm_number = 1
# output_folder = '/tmp/TEST'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# load archetype building here
if archetype == 'smrf_3_of_II':
    mdl, loadcase = smrf_3_of_II()
elif archetype == 'smrf_6_of_II':
    mdl, loadcase = smrf_6_of_II()
elif archetype == 'smrf_9_of_II':
    mdl, loadcase = smrf_9_of_II()
elif archetype == 'smrf_3_of_IV':
    mdl, loadcase = smrf_3_of_IV()
elif archetype == 'smrf_6_of_IV':
    mdl, loadcase = smrf_6_of_IV()
else:
    raise ValueError(f'Unknown archetype code: {archetype}')



# from osmg.gen.querry import LoadCaseQuerry
# querry = LoadCaseQuerry(mdl, loadcase)
# querry.level_masses()*386.22 / 2.00

# from osmg.graphics.preprocessing_3D import show
# show(mdl, loadcase, extrude=True)



t_bar = float(read_study_param(f'data/{archetype}/period'))

# retrieve some info used in the for loops
num_levels = len(mdl.levels.registry) - 1
level_heights = []
for level in mdl.levels.registry.values():
    level_heights.append(level.elevation)
level_heights = np.diff(level_heights)

lvl_nodes = []
base_node = list(mdl.levels.registry[0].nodes.registry.values())[0].uid
lvl_nodes.append(base_node)
for i in range(num_levels):
    lvl_nodes.append(loadcase.parent_nodes[i+1].uid)

# define analysis object
nlth = solver.NLTHAnalysis(
    mdl, {loadcase.name: loadcase},
    output_directory=f'{output_folder}')
nlth.settings.store_fiber = False
nlth.settings.store_forces = False
nlth.settings.store_reactions = False
nlth.settings.store_release_force_defo = False
nlth.settings.specific_nodes = lvl_nodes

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
           'periods': [t_bar, t_bar/10]}
# damping = {'type': 'modal',
#            'num_modes': 50,
#            'ratio': 0.03}

# nlth.plot_ground_motion(ground_motion_dir + '/' + str(gm_number) + 'x.txt', 0.005)

# run the nlth analysis
metadata = nlth.run(analysis_dt,
                    ground_motion_dir + '/' + str(gm_number) + 'x.txt',
                    ground_motion_dir + '/' + str(gm_number) + 'y.txt',
                    ground_motion_dir + '/' + str(gm_number) + 'z.txt',
                    ground_motion_dt,
                    finish_time=0.00,
                    damping=damping,
                    print_progress=False)

if not metadata['analysis_finished_successfully']:
    print(f'Analysis failed due to convergence issues. '
          f'| {ground_motion_dir}: {gm_number}')
    sys.exit()

# ~~~~~~~~~~~~~~~~ #
# collect response #
# ~~~~~~~~~~~~~~~~ #

time_vec = np.array(nlth.time_vector)
resp_a = {}
resp_v = {}
resp_u = {}
for i in range(num_levels+1):
    resp_a[i] = nlth.retrieve_node_abs_acceleration(lvl_nodes[i], loadcase.name)
    resp_v[i] = nlth.retrieve_node_abs_velocity(lvl_nodes[i], loadcase.name)
    if i > 0:
        resp_u[i] = nlth.retrieve_node_displacement(lvl_nodes[i], loadcase.name)


time_vec = np.array(nlth.time_vector)
np.savetxt(f'{output_folder}/time.csv',
           time_vec)

for direction in range(2):
    # store response time-histories
    # ground acceleration
    np.savetxt(f'{output_folder}/FA-0-{direction+1}.csv',
               resp_a[0].iloc[:, direction])
    # ground velocity
    np.savetxt(f'{output_folder}/FV-0-{direction+1}.csv',
               resp_v[0].iloc[:, direction])
    for lvl in range(num_levels):
        # story drifts
        if lvl == 0:
            u = resp_u[1].iloc[:, direction]
            dr = u / level_heights[lvl]
        else:
            uprev = resp_u[lvl].iloc[:, direction]
            u = resp_u[lvl + 1].iloc[:, direction]
            dr = (u - uprev) / level_heights[lvl]
        # story accelerations
        a = resp_a[lvl + 1].iloc[:, direction]
        # story velocities
        v = resp_v[lvl + 1].iloc[:, direction]

        np.savetxt(f'{output_folder}/ID-{lvl+1}-{direction+1}.csv', dr)
        np.savetxt(f'{output_folder}/FA-{lvl+1}-{direction+1}.csv', a)
        np.savetxt(f'{output_folder}/FV-{lvl+1}-{direction+1}.csv', v)

    # global building drift
    bdr = resp_u[num_levels].iloc[:, direction]
    bdr /= np.sum(level_heights)
    np.savetxt(f'{output_folder}/BD-{direction+1}.csv', bdr)
