"""
Gather and summarize variance-based sensitivity analysis results
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict
import os

# cases = ['healthcare3', 'office3']
cases = ['smrf_3_of_II']
beta_m_cases = ['medium', 'low']
# output_dir = "/home/john_vm/google_drive_encr/UCB/research/projects/299_report/299_report/data/performance_si"
repl_threshold_cases = [0.4, 1.0]

for case in cases:
    for beta_m_case in beta_m_cases:
        # for repl in repl_threshold_cases:
        for repl in [1.0]:

                idx = pd.IndexSlice

                hz_lvls = [f'hazard_level_{i+1}' for i in range(16)]
                rvgroups = ['edp', 'cmp_quant', 'cmp_dm', 'cmp_dv', 'bldg_dm', 'bldg_dv']

                si_dfs = []  # initialize

                for hz in hz_lvls:
                    for rvgroup in rvgroups:
                        res_path = f'analysis/{case}/{hz}/performance/{beta_m_case}/{repl}/{rvgroup}/sensitivity_indices.csv'
                        data = pd.read_csv(res_path, index_col=0)
                        data.index = pd.MultiIndex.from_tuples([(hz, x) for x in data.index])
                        si_dfs.append(data)


                all_df = pd.concat(si_dfs)
                all_df.index.names = ['hazard level', 'RV group']

        # # first order sensitivity index
        # my_order = ['edp', 'cmp_dm', 'cmp_dv', 'cmp_quant', 'bldg_dv', 'bldg_dm']
        # df_1_m = all_df.loc[:, 's1'].unstack(level=1)
        # df_1_m.index = [x.replace('hazard_level_', '') for x in df_1_m.index]
        # df_1_m.index = [int(x) for x in df_1_m.index]
        # df_1_m.sort_index(inplace=True)
        # df_1_m = df_1_m.loc[:, my_order].transpose()
        # df_1_m.index = range(len(df_1_m.index))
        # # --//-- error bars
        # df_1_e = (all_df.loc[:, 's1_CI_h'] - all_df.loc[:, 's1_CI_l']).unstack(level=1) / 2.
        # df_1_e.index = [x.replace('hazard_level_', '') for x in df_1_e.index]
        # df_1_e.index = [int(x) for x in df_1_e.index]
        # df_1_e.index += 16
        # df_1_e.sort_index(inplace=True)
        # df_1_e = df_1_e.loc[:, my_order].transpose()
        # df_1_e.index = range(len(df_1_e.index))
        # df_1 = pd.concat((df_1_m, df_1_e), axis=1)
        # df_1.to_csv(f'{output_dir}/s1_{case}_{beta_m_case}_{repl}.txt', header=False, sep=' ')
        # # total effect sensitivity index
        # df_T_m = all_df.loc[:, 'sT'].unstack(level=1)
        # df_T_m.index = [x.replace('hazard_level_', '') for x in df_T_m.index]
        # df_T_m.index = [int(x) for x in df_T_m.index]
        # df_T_m.sort_index(inplace=True)
        # df_T_m = df_T_m.loc[:, my_order].transpose()
        # df_T_m.index = range(len(df_T_m.index))
        # # --//-- error bars
        # df_T_e = (all_df.loc[:, 'sT_CI_h'] - all_df.loc[:, 'sT_CI_l']).unstack(level=1) / 2.
        # df_T_e.index = [x.replace('hazard_level_', '') for x in df_T_e.index]
        # df_T_e.index = [int(x) for x in df_T_e.index]
        # df_T_e.index += 16
        # df_T_e.sort_index(inplace=True)
        # df_T_e = df_T_e.loc[:, my_order].transpose()
        # df_T_e.index = range(len(df_T_e.index))
        # df_T = pd.concat((df_T_m, df_T_e), axis=1)
        # df_T.to_csv(f'{output_dir}/sT_{case}_{beta_m_case}_{repl}.txt', header=False, sep=' ')







# python plot

def bar_plot(ax, data, errors, colors=None, total_width=0.8, single_width=1, legend_title=None):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, yerr=errors[name][x], width=bar_width * single_width, color=colors[i % len(colors)], edgecolor='k')

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    # if legend_title:
    #     ax.legend(bars, data.keys(), frameon=False, title=legend_title, ncol=2)

    # change the style of the axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set(ylim=(0.0, 1.0))
    # add some space between the axis and the plot
    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 5))


data = {}
erbr = {}

my_order = ['edp', 'cmp_dm', 'cmp_dv', 'cmp_quant', 'bldg_dv', 'bldg_dm']
my_order_names = ['EDP', 'C-DM', 'C-DV', 'C-QNT', 'B-DV', 'B-DM']
hz_lvls_names = [f'{i+1}' for i in range(16)]

for i, hzlvl in enumerate(hz_lvls):
    vals = []
    ers = []
    for rvgroup in my_order:
        vals.append(all_df.loc[(hzlvl, rvgroup), 's1'])
        ers.append(all_df.loc[(hzlvl, rvgroup), 's1_CI_h'] - all_df.loc[(hzlvl, rvgroup), 's1_CI_l'])
    data[hz_lvls_names[i]] = vals
    erbr[hz_lvls_names[i]] = ers

fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 4))
bar_plot(ax1, data, erbr, total_width=.8, single_width=1.)
# ax1.set_xticks(range(6), my_order_names)
# ax1.set(ylabel='$s_1$')


data = {}
erbr = {}


for i, hzlvl in enumerate(hz_lvls):
    vals = []
    ers = []
    for rvgroup in my_order:
        vals.append(all_df.loc[(hzlvl, rvgroup), 'sT'])
        ers.append(all_df.loc[(hzlvl, rvgroup), 'sT_CI_h'] - all_df.loc[(hzlvl, rvgroup), 'sT_CI_l'])
    data[hz_lvls_names[i]] = vals
    erbr[hz_lvls_names[i]] = ers

bar_plot(ax2, data, erbr, colors=None, total_width=.8, single_width=1, legend_title='Hazard Level')
# ax2.set_xticks(range(6), my_order_names)
# ax2.set(ylabel='$s_T$')

plt.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.09)

plt.show()
