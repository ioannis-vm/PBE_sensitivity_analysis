"""
Postprocess performance evaluation results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


hazard_lvls = [f'hazard_level_{i+1}' for i in range(8)]

performance_evals = ['0', 'A', 'B', 'C', 'D', 'E', 'F']
tags = ['baseline', 'edp', 'quant', 'dm', 'collapse', 'dv-comp', 'dv-repl']


# # ~~~~~~~~~~~~~ #
# # plot all cdfs #
# # ~~~~~~~~~~~~~ #

# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red', 'tab:cyan', 'black']
# selection = [0, 6]
# fig = plt.figure(figsize=(8, 6))
# for j in range(8):
#     vals = []
#     m1 = np.full(len(performance_evals), 0.00)
#     s1 = np.full(len(performance_evals), 0.00)
#     for i, performance_eval in enumerate(performance_evals):
#         vals.append(pd.read_csv(os.path.join(
#             'analysis', hazard_lvls[j],
#             'performance', performance_eval, 'Summary.csv'), index_col=0))
#         m1[i] = vals[i]['repair_cost-'].mean()
#         s1[i] = vals[i]['repair_cost-'].std(ddof=1)
#     if j == 0:
#         for i, val in enumerate(vals):
#             if i in selection:
#                 sns.ecdfplot(val['repair_cost-'], label=tags[i], color=colors[i])
#                 # fig.axvline(x=m1, color='tab:blue')
#     else:
#         for i, val in enumerate(vals):
#             if i in selection:
#                 sns.ecdfplot(val['repair_cost-'], color=colors[i])
#                 # fig.axvline(x=m1, color='tab:blue')
# fig.xlabel = 'Cost c'
# fig.ylabel = 'Probability (cost < c)'
# fig.title = 'CDF of total repair cost'
# fig.legend()
# # plt.savefig(output_path)
# plt.show()
# plt.close()


# ~~~~~~~~~~~~~~~~~~~~~ #
# plot tornado diagrams #
# ~~~~~~~~~~~~~~~~~~~~~ #

means = np.full((len(hazard_lvls), len(performance_evals)), 0.00)
stdevs = np.full((len(hazard_lvls), len(performance_evals)), 0.00)
for i in range(len(hazard_lvls)):
    vals = []
    for j, performance_eval in enumerate(performance_evals):
        vals.append(pd.read_csv(os.path.join(
            'analysis', hazard_lvls[i],
            'performance', performance_eval, 'Summary.csv'), index_col=0))
        means[i, j] = vals[j]['repair_cost-'].mean()
        stdevs[i, j] = vals[j]['repair_cost-'].std(ddof=1)

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

fig, axs = plt.subplots(8, 1, sharex=True, sharey=True)
fig.set_size_inches(8, 12)
fig.suptitle('Mean replacement cost per hazard level')
for i, ax in enumerate(axs):
    ax.barh(tags[::-1], means[i, :][::-1]/1e6, 0.45,
            edgecolor='k', color='lightgray')
    ax.set_ylabel(str(f'lvl {i+1}'))
axs[-1].set_xlabel('Replacement Cost (million $)')
plt.subplots_adjust(left=0.15)
fig.savefig('means.pdf')
# fig.show()

fig, axs = plt.subplots(8, 1, sharex=True, sharey=True)
fig.set_size_inches(8, 12)
fig.suptitle('Standard deviation of replacement cost per hazard level')
for i, ax in enumerate(axs):
    ax.barh(tags[::-1], stdevs[i, :][::-1]/1e6, 0.45,
            edgecolor='k', color='lightgray')
    ax.set_ylabel(str(f'lvl {i+1}'))
axs[-1].set_xlabel('Standard Deviation of Replacement Cost (million $)')
plt.subplots_adjust(left=0.15)
fig.savefig('stdevs.pdf')
# fig.show()



# rearrange values to generate the tornado diagrams


data_vals = np.array([[0.20, 0.80, 1.30],
                      [0.15, 0.80, 1.55],
                      [0.75, 0.80, 0.96],
                      [0.27, 0.80, 1.40]])

data_tags = np.array(['foo',
                      'bar',
                      'something',
                      'stuff'])

# calculate swing
swing = data_vals[:, 2] - data_vals[:, 0]

# determine ordering
idx = np.argsort(swing)

# generate plot
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.barh(data_tags[idx], data_vals[idx, 1] - data_vals[idx, 0], 0.15,
        left=data_vals[idx, 0], edgecolor='k',
        color='lightgray')
ax.barh(data_tags[idx], data_vals[idx, 2] - data_vals[idx, 1], 0.15,
        left=data_vals[idx, 1], edgecolor='k',
        color='lightgray')
plt.subplots_adjust(left=0.3)
ax.set(xlim=(0.0, 2.50))
ax.set(xlabel='Standard Deviation')
# plt.savefig('test.pdf')
plt.show()
