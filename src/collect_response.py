import sys
sys.path.append("../OpenSeesPy_Building_Modeler")

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

# ~~~~~~~~~~~~ #
#  parameters  #
# ~~~~~~~~~~~~ #

results_folder = 'tmp/nlth_results'
num_analyses = 13
num_levels = 3
level_heights = np.array([12.*13.]*3)

# ~~~~~~~~~~~~~~~~~~~~ #
# function definitions #
# ~~~~~~~~~~~~~~~~~~~~ #


def retrieve_displacement_th(floor, drct, anal_obj):
    # retrieve the number of successful steps
    n_steps = anal_obj.n_steps_success
    # get the displacement time-history
    nid = anal_obj.building.list_of_parent_nodes()[floor].uniq_id
    d = []
    for i in range(n_steps):
        d.append(anal_obj.node_displacements[nid][i][drct])
    d = np.array(d)
    t = np.linspace(0.00, n_steps*0.1, n_steps)
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
    t = np.linspace(0.00, n_steps*0.1, n_steps)
    return np.column_stack((t, d))


def add_entry(dictionary, tag, value):
    if tag not in dictionary.keys():
        dictionary[tag] = [value]
    else:
        dictionary[tag].append(value)


# ~~~~~~~~~~~~~~~~ #
# analysis_results #
# ~~~~~~~~~~~~~~~~ #


# initialize results container

edps = {}
# evaluation id column
c_tag = "%eval_id"
edps[c_tag] = np.arange(1, num_analyses+1, 1)


for analysis_no in trange(num_analyses):

    gm_X_filepath = 'ground_motions/test_case/parsed/' + \
        str(analysis_no+1) + 'x.txt'
    gm_Y_filepath = 'ground_motions/test_case/parsed/' + \
        str(analysis_no+1) + 'y.txt'
    # import the analysis result
    ag = {}
    ag[0] = np.genfromtxt(gm_X_filepath)
    ag[1] = np.genfromtxt(gm_Y_filepath)
    n_pts = len(ag[0])
    t = np.linspace(0.00, 0.005*n_pts, n_pts)
    f = {}
    f[0] = interp1d(t, ag[0])
    f[1] = interp1d(t, ag[1])

    # import the analysis result
    with open(results_folder + '/nlth_' +
              str(analysis_no+1) + '.pcl', 'rb') as file_object:
        analysis = pickle.load(file_object)

    for direction in range(2):

        # ~~~~~~~~~~~~~~~~~~~ #
        # ground acceleration #
        # ~~~~~~~~~~~~~~~~~~~ #

        ag_max = np.max(np.abs(ag[direction]))

        add_entry(edps,
                  "1-" + "PFA-" + str(0) + "-" + str(direction+1),
                  ag_max)

        for lvl in range(num_levels):

            # ~~~~~~~~~~~~ #
            # story drifts #
            # ~~~~~~~~~~~~ #

            if lvl == 0:
                
                u = retrieve_displacement_th(lvl, direction, analysis)
                dr = u[:, 1] / level_heights[lvl]

            else:

                uprev = retrieve_displacement_th(lvl-1, direction, analysis)
                u = retrieve_displacement_th(lvl, direction, analysis)
                dr = (u[:, 1] - uprev[:, 1]) / level_heights[lvl]

            # obtain peak drift
            dr_max = np.max(np.abs(dr))

            add_entry(edps,
                      "1-" + "PID-" + str(lvl+1) + "-" + str(direction+1),
                      dr_max)

            # ~~~~~~~~~~~~~~~~~~~ #
            # story accelerations #
            # ~~~~~~~~~~~~~~~~~~~ #

            a = retrieve_acceleration_th(0, direction, analysis)

            # obtain peak acceleration
            a_max = np.max(np.abs(a1[:, 1]/386. + f[direction](a1[:, 0])))

            add_entry(edps,
                      "1-" + "PFA-" + str(lvl+1) + "-" + str(direction+1),
                      a_max)


# generate pandas dataframe
df = pd.DataFrame.from_dict(edps)

# write output file
df.to_csv("response.csv", index=False)
