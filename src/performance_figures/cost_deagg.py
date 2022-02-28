# %% Imports #

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pelicun.pelicun import base

# # ~~~~~~~~~~~~~~~~~~~~~ #
# # setup argument parser #
# # ~~~~~~~~~~~~~~~~~~~~~ #

# parser = argparse.ArgumentParser()
# parser.add_argument('--DV_path')
# parser.add_argument('--output_path')

# args = parser.parse_args()
# path = args.DV_path
# output_path = args.output_path

# debug
path = 'analysis/hazard_level_8/performance/0/LOSS_repair.csv'

# ~~~~ #
# main #
# ~~~~ #


loss_sample = pd.read_csv(path, index_col=0)

loss_sample = base.convert_to_MultiIndex(loss_sample, axis=1)

# eliminate building replacement cases
if ('COST', 'replacement', 'irreparable', '1', '0', '1', '1') in loss_sample.columns:
    loss_sample = loss_sample[loss_sample[('COST', 'replacement', 'irreparable', '1', '0', '1', '1')] == 0.00]

col_idx = loss_sample.columns.values

component_groups = list(set([val[1] for val in col_idx if val[0] == 'COST']))
component_groups.sort()

component_groups = np.array(component_groups)
means = np.array([loss_sample.loc[:, ('COST', comp)].to_numpy().sum(axis=1).mean()
                  for comp in component_groups])

reorder_idx = np.argsort(np.array(means))


# generate plot
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.barh(component_groups[reorder_idx], means[reorder_idx], 0.35,
        edgecolor='k',
        color='lightgray')
# ax.invert_yaxis()
plt.subplots_adjust(left=0.3)
# ax.set(xlim=(0.0, 2.50))
# plt.savefig('test.pdf')
plt.show()
