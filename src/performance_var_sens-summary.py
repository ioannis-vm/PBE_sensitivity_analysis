"""
Gather and summarize variance-based sensitivity analysis results
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict


mpl.rcParams['errorbar.capsize'] = 1.5
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams['text.usetex']=True

idx = pd.IndexSlice


hz_lvls = [f'hazard_level_{i+1}' for i in range(8)]
rvgroups = ['edp', 'cmp_quant', 'cmp_dm', 'cmp_dv', 'bldg_dm', 'bldg_dv']

si_dfs = []  # initialize

for hz in hz_lvls:
    for rvgroup in rvgroups:
        res_path = f'analysis/{hz}/performance/{rvgroup}/sensitividy_indices.csv'
        data = pd.read_csv(res_path, index_col=0)
        data.index = pd.MultiIndex.from_tuples([(hz, x) for x in data.index])
        si_dfs.append(data)


all_df = pd.concat(si_dfs)
all_df.index.names = ['hazard level', 'RV group']



# # export to latex

# # first-order sensitiviy indices
# my_order = ['edp', 'cmp_dm', 'cmp_dv', 'cmp_quant', 'bldg_dv', 'bldg_dm']

# data = OrderedDict()

# for rvgroup in my_order:
#     vals = []
#     for hzlvl in hz_lvls:
#         val = all_df.loc[(hzlvl, rvgroup), 's1']
#         val_h = all_df.loc[(hzlvl, rvgroup), 's1_CI_h']
#         val_l = all_df.loc[(hzlvl, rvgroup), 's1_CI_l']
#         diff = (val_h - val_l)/2.
#         add_conf = True
#         if val < 0.0:
#             val = 0.00
#             add_conf = False
#         if np.isnan(diff):
#             add_conf = False
#         if diff < 0.001:
#             add_conf = False
#         if add_conf:
#             vals.append(f'{val:.3f}±{diff:.3f}')
#         else:
#             vals.append(f'{val:.3f}')
#     data[rvgroup] = vals

# output_table = pd.DataFrame.from_dict(data)
# output_table.index = range(1, 9)
# output_table.index.name = 'Hazard Level'


# data_num = OrderedDict()

# for rvgroup in my_order:
#     vals = []
#     for hzlvl in hz_lvls:
#         val = all_df.loc[(hzlvl, rvgroup), 's1']
#         vals.append(val)
#     data_num[rvgroup] = vals

# output_tab_num = pd.DataFrame.from_dict(data_num)
# output_tab_num.mean(axis=0)

# print(output_table.to_latex())


# # total-effect sensitiviy indices

# data = OrderedDict()

# for rvgroup in my_order:
#     vals = []
#     for hzlvl in hz_lvls:
#         val = all_df.loc[(hzlvl, rvgroup), 'sT']
#         val_h = all_df.loc[(hzlvl, rvgroup), 'sT_CI_h']
#         val_l = all_df.loc[(hzlvl, rvgroup), 'sT_CI_l']
#         diff = (val_h - val_l)/2.
#         add_conf = True
#         if val < 0.0:
#             val = 0.00
#             add_conf = False
#         if np.isnan(diff):
#             add_conf = False
#         if diff < 0.001:
#             add_conf = False
#         if add_conf:
#             vals.append(f'{val:.3f}±{diff:.3f}')
#         else:
#             vals.append(f'{val:.3f}')
#     data[rvgroup] = vals

# output_table = pd.DataFrame.from_dict(data)
# output_table.index = range(1, 9)
# output_table.index.name = 'Hazard Level'

# print(output_table.to_latex())


# data_num = OrderedDict()

# for rvgroup in my_order:
#     vals = []
#     for hzlvl in hz_lvls:
#         val = all_df.loc[(hzlvl, rvgroup), 'sT']
#         vals.append(val)
#     data_num[rvgroup] = vals

# output_tab_num = pd.DataFrame.from_dict(data_num)
# output_tab_num.mean(axis=0)

# print(output_table.to_latex())











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
    if legend_title:
        ax.legend(bars, data.keys(), frameon=False, title=legend_title, ncol=2)

    # change the style of the axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set(ylim=(0.0, 1.0))
    # add some space between the axis and the plot
    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 5))






# I think I will keep this one.

data = {}
erbr = {}

my_order = ['edp', 'cmp_dm', 'cmp_dv', 'cmp_quant', 'bldg_dv', 'bldg_dm']
my_order_names = ['EDP', 'C-DM', 'C-DV', 'C-QNT', 'B-DV', 'B-DM']
hz_lvls_names = [f'{i}' for i in range(1, 9)]

colors=plt.cm.Pastel2(np.arange(8))

for i, hzlvl in enumerate(hz_lvls):
    vals = []
    ers = []
    for rvgroup in my_order:
        vals.append(all_df.loc[(hzlvl, rvgroup), 's1'])
        ers.append(all_df.loc[(hzlvl, rvgroup), 's1_CI_h'] - all_df.loc[(hzlvl, rvgroup), 's1_CI_l'])
    data[hz_lvls_names[i]] = vals
    erbr[hz_lvls_names[i]] = ers

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 4))
bar_plot(ax1, data, erbr, colors, total_width=.8, single_width=1.)
ax1.set_xticks(range(6), my_order_names)
ax1.set(ylabel='$s_1$')


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

bar_plot(ax2, data, erbr, colors=colors, total_width=.8, single_width=1, legend_title='Hazard Level')
ax2.set_xticks(range(6), my_order_names)
ax2.set(ylabel='$s_T$')

plt.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.09)

plt.savefig('test.pdf')
