"""
Gather and summarize variance-based sensitivity analysis results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


















def bar_plot(ax, data, errors, colors=None, total_width=0.8, single_width=1, legend=True):
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
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())





data = {}
errs = {}

for rvg in rvgroups:
    vals = list(all_df.loc[idx[:, rvg], 's1'])
    evals = list(all_df.loc[idx[:, rvg], 's1_CI_h'] - all_df.loc[idx[:, rvg], 's1_CI_l'])
    data[rvg] = vals
    errs[rvg] = evals

fig, ax = plt.subplots()
bar_plot(ax, data, evals, total_width=.8, single_width=.9)
# plt.xticks(range(8), hz_lvls)
plt.show()
