# %% Imports #

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

idx = pd.IndexSlice

output_dir = '/home/john_vm/google_drive_encr/UCB/research/projects/299_report/299_report/data/response'

# ~~~~ #
# main #
# ~~~~ #

num_hz = 16

for hz in [f'{i+1}' for i in range(num_hz)]:

    input_path = f'analysis/office3/hazard_level_{hz}/response_summary/response.csv'
    data = pd.read_csv(input_path, index_col=0)
    data.drop('units', inplace=True)
    data = data.astype(float)
    index_labels = [label.split('-') for label in data.columns]
    index_labels = np.array(index_labels)
    data.columns = pd.MultiIndex.from_arrays(index_labels.T)
    for col in data.columns:
        fig_type = col[0]
        loc = col[1]
        dir = col[2]
        xvals = data.loc[:, col]
        if fig_type in ['PID', 'RID']:
            xvals = xvals * 100.
        xvals.to_csv(f'{output_dir}/box_{fig_type}_{hz}_{loc}_{dir}.txt', sep=' ', index=None, header=None)
        if dir == '1':
            yvals = float(loc) + np.random.uniform(0.1, 0.3, len(data))
        else:
            yvals = float(loc) + np.random.uniform(-0.3, -0.1, len(data))
        if fig_type not in ['RID', 'SA_0.82']:
            np.savetxt(f'{output_dir}/scatter_{fig_type}_{hz}_{loc}_{dir}.txt', np.column_stack((xvals, yvals)), delimiter=' ')
        else:
            yvals = float(hz) + np.random.uniform(-0.1, 0.1, len(data))
            np.savetxt(f'{output_dir}/scatter_{fig_type}_{hz}_{loc}_{dir}.txt', np.column_stack((yvals, xvals)), delimiter=' ')
