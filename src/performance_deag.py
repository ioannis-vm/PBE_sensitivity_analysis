import sys
sys.path.insert(0, "src")


import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from p_58_assessment import P58_Assessment
from p_58_assessment import calc_sens
import argparse

plt.rcParams["font.family"] = "serif"

cases = ['office3', 'healthcare3']
repls = [0.40, 1.00]
output_dir_tikz = '/home/john_vm/google_drive_encr/UCB/research/projects/299_report/299_report/data/performance_deag'
response_of_case=dict(office3='office3', healthcare3='office3')
num_hz = 16
hazard_lvls = [f'hazard_level_{i+1}' for i in range(num_hz)]

for case in cases:
    for repl in repls:

        response_paths = [f'analysis/{response_of_case[case]}/{hz}/response_summary/response.csv'
                          for hz in hazard_lvls]
        performance_data_path = f'src/performance_data_{case}'


        c_modeling_uncertainty = np.sqrt(0.25**2+0.25**2)
        num_realizations = 1000
        replacement_threshold = repl
        perf_model_input_path = f'{performance_data_path}/input_cmp_quant.csv'
        cmp_fragility_input_path = f'{performance_data_path}/input_fragility.csv'
        cmp_repair_cost_input_path = f'{performance_data_path}/input_repair_cost.csv'

        dfs = []

        comps_structural = ['B1031.001', 'B1031.011a', 'B1031.011b', 'B1031.021a', 'B1035.001', 'B1035.002', 'B1035.011', 'B1035.012']
        comps_architectural = ['B2011.101', 'B2022.001', 'B3011.011', 'C1011.001a', 'C2011.001a', 'C3027.001', 'C3032.001b', 'C3032.001d', 'C3034.001']
        comps_mechanical = ['D1014.011.1', 'D1014.011.2', 'D1014.011.3', 'D1014.011.4', 'D3031.012c', 'D3031.021a', 'D3041.011a', 'D3041.012a', 'D3041.031a', 'D3041.041a', 'D3041.101a', 'D3052.013c', 'D5011.011a', 'D5012.013a', 'D5012.021a', 'D5012.031a']
        comps_flooding = ['C3021.001k', 'C3021.001k', 'D2021.011a', 'D2031.013b', 'D4011.021a', 'D4011.031a']
        comps_contents = dict(
            office3=['E2022.001', 'E2022.023', 'E2022.102a', 'E2022.112a'],
            healthcare3=['E2022.023', 'E2022.102a', 'E2022.112a', 'E1028.001', 'E1028.002', 'E1028.003', 'E1028.004', 'E1028.005', 'E1028.006', 'E1028.011', 'E1028.021', 'E1028.023', 'E1028.031', 'E1028.104', 'E1028.022', 'E1028.101', 'E1028.102', 'E1028.103', 'E1028.105', 'E1028.106', 'E1028.107', 'E1028.108', 'E1028.201', 'E1028.202', 'E1028.203', 'E1028.204', 'E1028.205', 'E1028.212', 'E1028.221', 'E1028.301', 'E1028.311', 'E1028.321', 'E1028.331', 'E1028.341', 'E1028.401', 'E1028.403', 'E1028.411', 'E1028.421', 'E1028.431', 'E1028.501']
        )

        for hz, response_path in zip(hazard_lvls, response_paths):

            asmt = P58_Assessment(
                num_realizations=num_realizations,
                replacement_threshold=replacement_threshold, fix_blg_dm_mean=True)
            asmt.read_perf_model(perf_model_input_path)
            asmt.read_fragility_input(cmp_fragility_input_path)
            asmt.read_cmp_repair_cost_input(cmp_repair_cost_input_path)
            asmt.run(response_path, c_modeling_uncertainty)

            mean_cost = asmt.cmp_cost.sum(axis=1).mean()

            asmt_grp = asmt.cmp_cost.groupby(level='component', axis=1).sum().mean()

            deag_cost = pd.DataFrame([
                asmt_grp.loc[comps_structural].sum(),
                asmt_grp.loc[comps_architectural].sum(),
                asmt_grp.loc[comps_mechanical].sum(),
                asmt_grp.loc[comps_flooding].sum(),
                asmt_grp.loc[comps_contents[case]].sum()
            ], index=['structural', 'architectural', 'mechanical', 'flooding', 'contents'], columns=[hz])
            dfs.append(deag_cost)

        res_df = pd.concat(dfs, axis=1)
        res_df.columns = hazard_lvls

        # colors = plt.cm.Pastel2(np.arange(num_hz))


        # fig, ax = plt.subplots()
        # res_df.transpose().plot(kind='bar', stacked=True, rot=0, color=colors, edgecolor='k', ax=ax)
        # ax.legend(frameon=False, title='Category', ncol=1)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # # ax.set(ylim=(0.0, 1.0))
        # # add some space between the axis and the plot
        # ax.spines['left'].set_position(('outward', 8))
        # ax.spines['bottom'].set_position(('outward', 5))
        # ax.set(ylabel='Mean Cumulative Cost')
        # plt.show()

        # write tikz data file
        res_df.columns = range(1, num_hz+1)
        for my_idx in res_df.index:
            df_temp = res_df.loc[my_idx, :]/1e6
            df_temp.to_csv(f'{output_dir_tikz}/{case}_{repl}_{my_idx}.txt', sep=' ')
