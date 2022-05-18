"""
Gather and summarize variance-based sensitivity analysis results
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict
import os


case = 'office3'
output_directory = f'analysis/{case}/merged/performance'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


mpl.rcParams['errorbar.capsize'] = 1.5
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"

idx = pd.IndexSlice


hz_lvls = [f'hazard_level_{i+1}' for i in range(8)]
rvgroups = ['edp', 'cmp_quant', 'cmp_dm', 'cmp_dv', 'bldg_dm', 'bldg_dv']

stdevs1 = np.zeros((len(hz_lvls), len(rvgroups)))
stdevs2 = np.zeros((len(hz_lvls), len(rvgroups)))


for i, hz in enumerate(hz_lvls):
    for j, rvgroup in enumerate(rvgroups):
        res1_path = f'analysis/{case}/{hz}/performance/{rvgroup}/total_cost_realizations.csv'
        res2_path = f'analysis/{case}/{hz}/performance/{rvgroup}/total_cost_realizations_{rvgroup}.csv'
        data1 = pd.read_csv(res1_path, index_col=0)
        stdev = data1.to_numpy().std()
        stdevs1[i, j] = stdev
        data2 = pd.read_csv(res2_path, index_col=0)
        stdev = data2.to_numpy().std()
        stdevs2[i, j] = stdev

stdevsA = pd.DataFrame(stdevs1, index=hz_lvls, columns=rvgroups)
stdevsB = pd.DataFrame(stdevs2, index=hz_lvls, columns=rvgroups)

stdevDiff = pd.DataFrame((stdevs1 - stdevs2)/stdevs1, index=hz_lvls, columns=rvgroups)


stdevDiff.plot.bar(rot=45)
plt.show()


# dat1 = pd.read_csv('analysis/office3/hazard_level_7/performance/edp/total_cost_realizations.csv', index_col=0)
# dat2 = pd.read_csv('analysis/office3/hazard_level_7/performance/edp/total_cost_realizations_edp.csv', index_col=0)

# import seaborn as sns

# dat1.loc[:, 'A'].to_numpy().std()
# dat2.loc[:, 'A'].to_numpy().std()
