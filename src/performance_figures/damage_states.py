# %% Imports #

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sns.set_theme(style="whitegrid")

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

# todo
parser = argparse.ArgumentParser()
parser.add_argument('--DMG_path')
parser.add_argument('--output_path')

args = parser.parse_args()
path = args.DMG_path
output_path = args.output_path

# # debug
# import os
# os.chdir('analysis/hazard_level_8')
# path = 'performance/A/DMG.csv'

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

percentage = {}
for i in range(len(fragility_group)):
    percentage[fragility_group[i]] = {}
for i in range(len(fragility_group)):
    percentage[fragility_group[i]][base(damage_state[i])] = []
for i in range(len(fragility_group)):
    percentage[fragility_group[i]][base(damage_state[i])].append(
        len(quant[:, i][quant[:, i] != 0.0]))

num_realizations = len(quant)

for fg in percentage.keys():
    for ds in percentage[fg].keys():
        numm = len(percentage[fg][ds]) * num_realizations
        summ = np.sum(np.array(percentage[fg][ds]))
        p = summ / numm
        percentage[fg][ds] = p

# rearrange dictionary to match the Seaborn data format

x = []
y = []
group = []
for fg in percentage.keys():
    for ds in percentage[fg].keys():
        y.append(fg)
        group.append(ds)
        x.append(percentage[fg][ds])
pinput = {'x': x, 'y': y, 'group': group}

plt.figure(figsize=(10, 10))
sns.barplot(x='x', y='y', hue='group', data=pinput, orient='h')
plt.xlim((0.00, 1.00))
# plt.show()
plt.savefig(output_path)
