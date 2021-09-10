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

output_folder = 'tmp/pushover_results_corot'


# ~~~~~~~~~~~~~~~~~~~~ #
# function definitions #
# ~~~~~~~~~~~~~~~~~~~~ #


# ~~~~~~~~ #
# analysis #
# ~~~~~~~~ #

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# import the pre-processed building object
with open(building_path, 'rb') as f:
    b = pickle.load(f)

pushover_analysis_x = solver.PushoverAnalysis(b)
control_node = b.list_of_parent_nodes()[-1]  # top floor
analysis_metadata = pushover_analysis_x.run(
    "x",
    np.array([60.]),
    control_node,
    1./2.)
n_plot_steps = analysis_metadata['successful steps']
# plot pushover curve
pushover_analysis_x.plot_pushover_curve("x", control_node)

pushover_analysis_y = solver.PushoverAnalysis(b)
control_node = b.list_of_parent_nodes()[-1]  # top floor
analysis_metadata = pushover_analysis_y.run(
    "y",
    np.array([60.]),
    control_node,
    1./2.)
n_plot_steps = analysis_metadata['successful steps']
# plot pushover curve
pushover_analysis_y.plot_pushover_curve("y", control_node)

