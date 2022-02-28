# %% Imports #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from pelicun.pelicun import base

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--DMG_path')
parser.add_argument('--output_path')

args = parser.parse_args()
path = args.DMG_path
output_path = args.output_path

# # debug
path = 'analysis/hazard_level_8/performance/0/DMG_sample.csv'

# ~~~~ #
# main #
# ~~~~ #

dmg_sample = pd.read_csv(path, index_col=0)

dmg_sample = base.convert_to_MultiIndex(dmg_sample, axis=1)

col_idx = dmg_sample.columns.values

component_groups = list(set([val[0] for val in col_idx]))
component_groups.sort()

plot_data = {}

val_list = []
count_list = []
max_num = 0

for group in component_groups:

    vals, counts = np.unique(dmg_sample.loc[:, group].to_numpy(),
                             return_counts=True)
    counts = counts/np.sum(counts)
    if max(vals) > max_num:
        max_num = max(vals) + 1

    val_list.append(vals)
    count_list.append(counts)

for i in range(len(count_list)):
    num_times = max_num - len(count_list[i])
    if num_times > 0:
        count_list[i] = np.concatenate((count_list[i],
                                        ([0.00]*num_times)), axis=0)

all_counts = np.row_stack((count_list))


fig, ax = plt.subplots(figsize=(10, 10))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.barh(component_groups, all_counts[:, 0],
        0.35, edgecolor='k', color='lightgreen')
ax.barh(component_groups, all_counts[:, 1],
        0.35, left=all_counts[:, 0],
        edgecolor='k', color='lightyellow')
ax.barh(component_groups, all_counts[:, 2],
        0.35, left=all_counts[:, 1]+all_counts[:, 0],
        edgecolor='k', color='lightblue')
ax.barh(component_groups, all_counts[:, 3],
        0.35, left=all_counts[:, 2]+all_counts[:, 1]+all_counts[:, 0],
        edgecolor='k', color='orange')
ax.barh(component_groups, all_counts[:, 4],
        0.35, left=all_counts[:, 3]+all_counts[:, 2]+all_counts[:, 1] +
        all_counts[:, 0],
        edgecolor='k', color='magenta')
plt.subplots_adjust(left=0.3)
ax.invert_yaxis()
# ax.set(xlim=(0.0, 2.50))
# plt.savefig('output_path')
plt.show()
