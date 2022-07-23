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

archetype = 'smrf_9_of_II'
output_folder = f'analysis/{archetype}/pushover'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# load archetype building here
if archetype == 'smrf_3_of_II':
    mdl, loadcase = smrf_3_of_II()
    modeshape = np.array((0.00, 0.35, 0.68, 1.00))
elif archetype == 'smrf_6_of_II':
    mdl, loadcase = smrf_6_of_II()
    modeshape = np.array((0.00, 0.16, 0.33, 0.51,
                          0.67, 0.84, 1.00))
elif archetype == 'smrf_9_of_II':
    mdl, loadcase = smrf_9_of_II()
    modeshape = np.array((0.00, 0.10, 0.21, 0.33,
                          0.45, 0.56, 0.67,
                          0.79, 0.89, 1.00))
elif archetype == 'smrf_3_of_IV':
    mdl, loadcase = smrf_3_of_IV()
    modeshape = np.array((0.00, 0.33, 0.67, 1.00))
elif archetype == 'smrf_6_of_IV':
    mdl, loadcase = smrf_6_of_IV()
    modeshape = np.array((0.00, 0.15, 0.32, 0.49,
                          0.67, 0.84, 1.00))
else:
    raise ValueError(f'Unknown archetype code: {archetype}')


# define analysis
anl = solver.PushoverAnalysis(mdl, {loadcase.name: loadcase})
anl.settings.store_forces = False
control_node = list(loadcase.parent_nodes.values())[-1]


# from osmg.graphics.preprocessing_3D import show
# show(mdl, loadcase)

# run analysis
anl.run(
    'x', [control_node.coords[2]*0.04],
    control_node, 1.0,
    modeshape=modeshape)

# retrieve results
anl.plot_pushover_curve(loadcase.name, 'x', control_node)

# from osmg.graphics.postprocessing_3D import show_deformed_shape
# show_deformed_shape(anl, loadcase.name, anl.results[loadcase.name].n_steps_success, 0.00, False)

displ, force = anl.table_pushover_curve(loadcase.name, 'x', control_node)
import pandas as pd
df = pd.DataFrame(np.column_stack((displ, force)), columns=['displ', 'force'])

from osmg.gen.querry import LoadCaseQuerry
from osmg.common import G_CONST_IMPERIAL
querry = LoadCaseQuerry(mdl, loadcase)
weight = np.sum(querry.level_masses())*G_CONST_IMPERIAL

df_m = pd.DataFrame.from_dict(
    {
        'weight (lb)': [weight],
        'height (in)': [control_node.coords[2]]},
)

# save results
df.to_csv(f'{output_folder}/curve.csv')
df_m.to_csv(f'{output_folder}/metadata.csv')
