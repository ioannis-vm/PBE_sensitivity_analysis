import sys
sys.path.append("../OpenSeesPy_Building_Modeler")

import numpy as np
import modeler
import solver
import time
import pickle
import os
from tqdm import trange
import matplotlib.pyplot as plt

# ~~~~~~~~~~~~ #
#  parameters  #
# ~~~~~~~~~~~~ #

results_folder = 'tmp/nlth_results'
analysis_no = 0

# ~~~~~~~~~~~~~~~~ #
# analysis_results #
# ~~~~~~~~~~~~~~~~ #

# gm_filepath = 'ground_motions/test_case/parsed/1x.txt'
# ag = np.genfromtxt(gm_filepath)
# n_pts = len(ag)
# t = np.linspace(0.00, 0.005*n_pts, n_pts)
# from scipy.interpolate import interp1d
# f = interp1d(t, ag)

# import the analysis result
with open(results_folder + '/nlth_' +
          str(analysis_no+1) + '.pcl', 'rb') as f:
    analysis = pickle.load(f)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# displacement of parent nodes #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def retrieve_th(floor, direction, anal_obj):
    # retrieve the number of successful steps
    n_steps = anal_obj.n_steps_success
    # get the displacement time-history of the roof in the y direction
    nid = anal_obj.building.list_of_parent_nodes()[floor].uniq_id
    d = []
    for i in range(n_steps):
        d.append(anal_obj.node_displacements[nid][i][direction])
    d = np.array(d)
    time = np.linspace(0.00, n_steps*0.1, n_steps)
    return np.column_stack((time, d))


th0 = retrieve_th(0, 0, analysis)
th1 = retrieve_th(1, 0, analysis)
th2 = retrieve_th(2, 0, analysis)

plt.figure()
ax = plt.subplot(111)
# ax.plot(th0[:, 0], th0[:, 1])
# ax.plot(th1[:, 0], th1[:, 1])
ax.plot(th2[:, 0], th2[:, 1])
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.show()
plt.close()


# ~~~~~~~~~~~ #
# base shear  #
# ~~~~~~~~~~~ #

def retrieve_reaction_th(direction, anal_obj):
    # retrieve the number of successful steps
    n_steps = anal_obj.n_steps_success
    d = []
    for i in range(n_steps):
        d.append(anal_obj.global_reactions(i)[direction])
    d = np.array(d)
    time = np.linspace(0.00, n_steps*0.1, n_steps)
    return np.column_stack((time, d))

th0 = retrieve_reaction_th(0, analysis)/1000  # kips

plt.figure()
plt.plot(th0[:, 0], th0[:, 1])
plt.grid()
plt.legend()
plt.show()
plt.close()
