import sys
sys.path.append("../OpenSeesPy_Building_Modeler")

import numpy as np
import modeler
import solver
import time
import pickle
import os
from tqdm import trange

# ~~~~~~~~~~~~ #
#  parameters  #
# ~~~~~~~~~~~~ #

building_path = 'tmp/building.pcl'

ground_motion_dir = 'ground_motions/test_case/parsed/'
ground_motion_dt = 0.005
num_files = 13


output_folder = 'tmp/nlth_results'


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


# ~~~~~~~~ #
# analysis #
# ~~~~~~~~ #

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for i in trange(3, num_files):

    print()
    print('Working on record ' + str(i+1))
    print()

    # import the pre-processed building object
    with open(building_path, 'rb') as f:
        b = pickle.load(f)

    # define analysis object
    nlth = solver.NLTHAnalysis(b)

    # get the corresponding ground motion duration
    dx = get_duration(ground_motion_dir + str(i+1) + 'x.txt', ground_motion_dt)
    dy = get_duration(ground_motion_dir + str(i+1) + 'x.txt', ground_motion_dt)
    dz = get_duration(ground_motion_dir + str(i+1) + 'x.txt', ground_motion_dt)
    duration = np.min(np.array((dx, dy, dz)))  # note: actually should be =
        
    # run the nlth analysis
    nlth.run(duration, 0.05,
             ground_motion_dir + str(i+1) + 'x.txt',
             ground_motion_dir + str(i+1) + 'y.txt',
             ground_motion_dir + str(i+1) + 'z.txt',
             ground_motion_dt)

    # store the results
    with open(output_folder + '/nlth_' + str(i+1) + '.pcl', 'wb') as f:
        pickle.dump(nlth, f)
