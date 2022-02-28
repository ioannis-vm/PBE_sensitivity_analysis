# %% Imports #

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import sys
sys.path.insert(0, "pelicun")
from pelicun import base

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


# # ~~~~~~~~~~~~~~~~~~~~~ #
# # setup argument parser #
# # ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--LOSS_repair_path')
parser.add_argument('--output_path')

args = parser.parse_args()
path = args.LOSS_repair_path
output_path = args.output_path

# # debug
# path = 'analysis/hazard_level_8/performance/0/LOSS_repair.csv'
# output_path = 'figures/hazard_level_8/performance/0'

# ~~~~ #
# main #
# ~~~~ #

if not os.path.exists(output_path):
    os.makedirs(output_path)

loss_sample = pd.read_csv(path, index_col=0)
loss_sample = base.convert_to_MultiIndex(loss_sample, axis=1)

# eliminate building replacement cases
if ('COST', 'replacement', 'irreparable',
    '1', '0', '1', '1') in loss_sample.columns:
    loss_sample = loss_sample[loss_sample[
        ('COST', 'replacement', 'irreparable',
         '1', '0', '1', '1')] == 0.00]

col_idx = loss_sample.columns.values

damaged_component_groups = list(
    set([val[1] for val in col_idx if val[0] == 'COST']))

structural_list = [
    'B1031.001', 'B1031.011c', 'B1035.001', 'B1035.011']

architectural_list = [
    'B2011.101', 'B2022.001', 'B3011.011', 'C1011.001a',
    'C2011.001a', 'C3027.001', 'C3032.001b', 'C3032.001d']

mech_electr_list = ['C3034.001', 'D5011.011a', 'D5012.013a',
                    'D5012.021a', 'D5012.031a', 'D1014.011',
                    'D3031.011a', 'D3031.021a', 'D3041.011a',
                    'D3041.012a', 'D3041.031a', 'D3041.041a',
                    'D3041.101a', 'D3052.011a', 'D4011.031a']

flooding_list = ['D2021.011a', 'D4011.021a', 'C3021.001k']

tenant_list = ['E2022.001', 'E2022.102a', 'E2022.112a', 'E2022.023']


all_lists = [
    structural_list, architectural_list, mech_electr_list,
    flooding_list, tenant_list]

names = np.array(
    ['Structural', 'Architectural', 'Mechanical/Electrical',
     'Plumbing/Flooding', 'Contents'])

means = []
for comp_list in all_lists:
    subset = []
    for c in comp_list:
        if c in damaged_component_groups:
            subset.append(c)
    selection = loss_sample.loc[:, 'COST'].loc[:, subset]
    mean = np.array(selection).sum(axis=1).mean()
    means.append(mean)
means = np.array(means)

reorder_idx = np.argsort(np.array(means))


# generate plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.barh(names[reorder_idx], means[reorder_idx]/1e3, 0.35,
        edgecolor='k',
        color='lightgray')
plt.title('Mean repair cost per component category')
plt.xlabel('Repair Cost (thousand $)')
plt.subplots_adjust(left=0.3)
plt.savefig(f'{output_path}/repair_per_category.pdf')
# plt.show()
