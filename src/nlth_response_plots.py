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
xlim = (0.00, 60.00)
ylim_accel = (-2.50, 2.50)
ylim_drift = (-0.05, 0.05)

plot_output_folder = 'tmp/plots_nlth_results'

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


def make_plot(x, y,
              xlim,
              ylim,
              xlabel, ylabel,
              title,
              save_path,
              fig_size=(12, 8),
              grids=True):
    plt.figure(figsize=fig_size)
    plt.plot(x, y, 'k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if grids:
        plt.grid()
    plt.savefig(save_path)
    plt.close()




# ~~~~~~~~~~~~~~~~ #
# analysis_results #
# ~~~~~~~~~~~~~~~~ #


if not os.path.exists(plot_output_folder):
    os.mkdir(plot_output_folder)


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

            fig_path = plot_output_folder + \
                "/drift-" + \
                str(analysis_no) + "-" + str(direction) + "-" + \
                str(lvl) + ".png"
            make_plot(u[:, 0], dr,
                      xlim,
                      ylim_drift,
                      "Time [s]", "Story Drift",
                      fig_path, fig_path)

            # ~~~~~~~~~~~~~~~~~~~ #
            # story accelerations #
            # ~~~~~~~~~~~~~~~~~~~ #

            a = retrieve_acceleration_th(lvl, direction, analysis)

            fig_path = plot_output_folder + \
                "/acceleration-" + \
                str(analysis_no) + "-" + str(direction) + "-" + \
                str(lvl) + ".png"
            make_plot(a[:, 0],
                      a[:, 1]/386. + f[direction](a[:, 0]),
                      xlim,
                      ylim_accel,
                      "Time [s]", "Acceleration [g]",
                      fig_path, fig_path)
