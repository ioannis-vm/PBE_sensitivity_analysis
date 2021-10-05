"""
Nonlinear time-history analysis of a predefined building object
using a series of ground motion records
of varying levels of intensity to create a series of
peak response quantities for these levels of intensity,
suitable for subsequent use in PELICUN.

"""

import sys
sys.path.append("../../../OpenSeesPy_Building_Modeler")

import numpy as np
import modeler
import solver
import time
import pickle
import os
from tqdm import trange
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--building')
parser.add_argument('--gm_dir')
parser.add_argument('--gm_dt')
parser.add_argument('--analysis_dt')
parser.add_argument('--gm_number')
parser.add_argument('--output_dir')

args = parser.parse_args()

building_path = args.building  # 'tmp/building.pcl'
ground_motion_dir = args.gm_dir  # 'ground_motions/test_case/parsed'
ground_motion_dt = float(args.gm_dt)  # 0.005
analysis_dt = float(args.analysis_dt)  # 0.05
gm_number = int(args.gm_number)
output_folder = args.output_dir  # 'response/test_case'

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
    nid = anal_obj.building.list_of_parent_nodes()[floor].uniq_id
    d = []
    for i in range(n_steps):
        d.append(anal_obj.node_displacements[nid][i][drct])
    d = np.array(d)
    t = np.array(nlth.time_vector)
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

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# # initialize response dictionary
# edps = {}
# # evaluation id column
# c_tag = "%eval_id"
# edps[c_tag] = np.arange(1, num_analyses+1, 1)


# import the pre-processed building object
with open(building_path, 'rb') as f:
    b = pickle.load(f)

# retrieve some info used in the for loops
num_levels = len(b.levels.level_list) - 1
level_heights = []
for level in b.levels.level_list:
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

# run the nlth analysis
nlth.run(duration, analysis_dt,
         ground_motion_dir + '/' + str(gm_number) + 'x.txt',
         ground_motion_dir + '/' + str(gm_number) + 'y.txt',
         ground_motion_dir + '/' + str(gm_number) + 'z.txt',
         ground_motion_dt,
         damping_ratio=0.03,
         skip_steps=0,
         printing=True)

# ~~~~~~~~~~~~~~~~ #
# collect response #
# ~~~~~~~~~~~~~~~~ #
ag = {}
ag[0] = np.genfromtxt(gm_X_filepath)
ag[1] = np.genfromtxt(gm_Y_filepath)
n_pts = len(ag[0])
t = np.linspace(0.00, ground_motion_dt*n_pts, n_pts)
print()
print(t[-1])
print(np.array(nlth.time_vector)[-1])
print()
f = {}
f[0] = interp1d(t, ag[0], bounds_error=False, fill_value=0.00)
f[1] = interp1d(t, ag[1], bounds_error=False, fill_value=0.00)


for direction in range(2):
    # store response time-histories
    prepend = output_folder + '/'
    if not os.path.exists(prepend):
        os.mkdir(prepend)
    np.savetxt(prepend + "FA-0-" + str(direction+1) + '.csv',
               ag[direction])
    for lvl in range(num_levels):
        # story drifts
        if lvl == 0:
            u = retrieve_displacement_th(lvl, direction, nlth)
            dr = u[:, 1] / level_heights[lvl]
        else:
            uprev = retrieve_displacement_th(lvl-1, direction, nlth)
            u = retrieve_displacement_th(lvl, direction, nlth)
            dr = (u[:, 1] - uprev[:, 1]) / level_heights[lvl]
        # story accelerations
        a1 = retrieve_acceleration_th(lvl, direction, nlth)
        # story velocities
        vel = retrieve_velocity_th(lvl, direction, nlth)

        np.savetxt(prepend + "ID-" + str(lvl+1) + "-" + str(direction+1) +
                   '.csv',
                   dr)
        np.savetxt(prepend + "FA-" + str(lvl+1) + "-" + str(direction+1) +
                   '.csv',
                   a1[:, 1]/386. + f[direction](a1[:, 0]))
        np.savetxt(prepend + "FV-" + str(lvl+1) + "-" + str(direction+1) +
                   '.csv',
                   vel[:, 1])
