# %% Imports #

import numpy as np
import matplotlib.pyplot as plt
import argparse

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--DV_path')
parser.add_argument('--output_path')

args = parser.parse_args()
path = args.DV_path
output_path = args.output_path

# # debug
# import os
# os.chdir('analysis/hazard_level_8')
# path = 'performance/A/DV_rec_cost.csv'

# ~~~~ #
# main #
# ~~~~ #

def base(ds_string):
    return "DS " + ds_string.split("_")[0]

fragility_group = np.genfromtxt(
    path, max_rows=1, delimiter=',', dtype='str')[1::]
performance_group = np.genfromtxt(
    path, skip_header=1, max_rows=1, delimiter=',')[1::]
damage_state = np.genfromtxt(
    path, skip_header=2, max_rows=1, delimiter=',', dtype='str')[1::]
quant = np.genfromtxt(
    path, skip_header=3, delimiter=',')[:, 1::]


cost = {}
for i in range(len(fragility_group)):
    cost[fragility_group[i]] = {'DS 1': 0., 'DS 2': 0., 'DS 3': 0.}
for i in range(len(fragility_group)):
    if base(damage_state[i]) in ['DS 1', 'DS 2', 'DS 3']:
        cost[fragility_group[i]][base(damage_state[i])] += np.mean(quant[:, i])


label = []
pcost = {'DS 1': [], 'DS 2': [], 'DS 3': []}

for fg in cost.keys():
    label.append(fg)
    for ds in ['DS 1', 'DS 2', 'DS 3']:
        pcost[ds].append(cost[fg][ds])

for ds in ['DS 1', 'DS 2', 'DS 3']:
    pcost[ds] = np.array(pcost[ds])

y_pos = np.arange(len(label))

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
ax.grid(zorder=0)
ax.set_yticks(y_pos)
ax.set_yticklabels(label)
ax.barh(y_pos, pcost['DS 1'], zorder=2, label='DS 1')
ax.barh(y_pos, pcost['DS 2'], left=pcost['DS 1'], zorder=2, label='DS 2')
ax.barh(y_pos, pcost['DS 3'], left=pcost['DS 1'] + pcost['DS 2'], zorder=2, label='DS 3')
ax.invert_yaxis()  # labels read top-to-bottom
ax.legend()
plt.xlim((0.00, 3e6))
plt.title('Mean Repair Cost per Component')
# plt.show()
plt.savefig(output_path)
