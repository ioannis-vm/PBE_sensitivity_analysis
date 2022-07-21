"""
Variance-based sensitivity analysis of the FEMA P-58 methodology
"""

import sys
sys.path.insert(0, "src")


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

rv_groups = ['edp', 'cmp_quant', 'cmp_dm', 'cmp_dv', 'bldg_dm', 'bldg_dv']

# ~~~~~~~~~~~~~~~~~~~~~~ #
# set up argument parser #
# ~~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--response_path')
parser.add_argument('--modeling_uncertainty_case')
parser.add_argument('--repl_thr')
parser.add_argument('--performance_data_path')
parser.add_argument('--analysis_output_path')

args = parser.parse_args()
response_path = args.response_path
modeling_uncertainty_case = args.modeling_uncertainty_case
replacement_threshold = float(args.repl_thr)
performance_data_path = args.performance_data_path
analysis_output_path = args.analysis_output_path

# ~~~~~~~~~~ #
# parameters #
# ~~~~~~~~~~ #

if modeling_uncertainty_case == 'medium':
    c_modeling_uncertainty = np.sqrt(0.25**2+0.25**2)
elif modeling_uncertainty_case == 'low':
    c_modeling_uncertainty = np.sqrt(0.10**2+0.10**2)
else:
    raise ValueError('Unknown modeling uncertainty case specified')

# debug
# num_realizations = 50000
num_realizations = 5
perf_model_input_path = f'{performance_data_path}/input_cmp_quant.csv'
cmp_fragility_input_path = f'{performance_data_path}/input_fragility.csv'
cmp_repair_cost_input_path = f'{performance_data_path}/input_repair_cost.csv'

desc = {'bldg_dm': 'Building DM',
        'bldg_dv': 'Building DV',
        'cmp_dm': 'Component DM',
        'cmp_dv': 'Component DV',
        'cmp_quant': 'Component Quantity',
        'edp': 'EDP'}

if not os.path.exists(analysis_output_path):
    os.makedirs(analysis_output_path)

# if not os.path.exists(figures_output_path):
#     os.makedirs(figures_output_path)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       Analysis A                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

asmt_A = P58_Assessment(
    num_realizations=num_realizations,
    replacement_threshold=replacement_threshold)
asmt_A.read_perf_model(perf_model_input_path)
asmt_A.read_fragility_input(cmp_fragility_input_path)
asmt_A.read_cmp_repair_cost_input(cmp_repair_cost_input_path)
asmt_A.run(response_path, c_modeling_uncertainty)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       Analysis B                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

asmt_B = P58_Assessment(
    num_realizations=num_realizations,
    replacement_threshold=replacement_threshold)
asmt_B.read_perf_model(perf_model_input_path)
asmt_B.read_fragility_input(cmp_fragility_input_path)
asmt_B.read_cmp_repair_cost_input(cmp_repair_cost_input_path)
asmt_B.run(response_path, c_modeling_uncertainty)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       Analyses C and D                         #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

asmt_C = P58_Assessment(
    num_realizations=num_realizations,
    replacement_threshold=replacement_threshold)
asmt_C.read_perf_model(perf_model_input_path)
asmt_C.read_fragility_input(cmp_fragility_input_path)
asmt_C.read_cmp_repair_cost_input(cmp_repair_cost_input_path)

asmt_D = P58_Assessment(
    num_realizations=num_realizations,
    replacement_threshold=replacement_threshold)
asmt_D.read_perf_model(perf_model_input_path)
asmt_D.read_fragility_input(cmp_fragility_input_path)
asmt_D.read_cmp_repair_cost_input(cmp_repair_cost_input_path)

for rv_group in rv_groups:

    if rv_group == 'bldg_dm':

        asmt_C.edp_samples = asmt_B.edp_samples
        asmt_C.cmp_quant_RV = asmt_B.cmp_quant_RV
        temp = asmt_B.cmp_fragility_RV.copy()
        subset_cols = [('collapse', '0', '1', '0', 'DS1'),
                       ('irreparable', '0', '1', '0', 'DS1')]
        temp.loc[:, subset_cols] = \
            asmt_A.cmp_fragility_RV.loc[:, subset_cols]
        asmt_C.cmp_fragility_RV = temp
        asmt_C.cmp_damage_consequence_RV = asmt_B.cmp_damage_consequence_RV
        asmt_C.cmp_cost_RV = asmt_B.cmp_cost_RV
        asmt_C.calc_cmp_damage()
        asmt_C.calc_cmp_dmg_quant()
        asmt_C.calc_cmp_cost()
        asmt_C.calc_total_cost()

        asmt_D.edp_samples = asmt_A.edp_samples
        asmt_D.cmp_quant_RV = asmt_A.cmp_quant_RV
        temp = asmt_A.cmp_fragility_RV.copy()
        subset_cols = [('collapse', '0', '1', '0', 'DS1'),
                       ('irreparable', '0', '1', '0', 'DS1')]
        temp.loc[:, subset_cols] = \
            asmt_B.cmp_fragility_RV.loc[:, subset_cols]
        asmt_D.cmp_fragility_RV = temp
        asmt_D.cmp_damage_consequence_RV = asmt_A.cmp_damage_consequence_RV
        asmt_D.cmp_cost_RV = asmt_A.cmp_cost_RV
        asmt_D.calc_cmp_damage()
        asmt_D.calc_cmp_dmg_quant()
        asmt_D.calc_cmp_cost()
        asmt_D.calc_total_cost()

    elif rv_group == 'bldg_dv':

        asmt_C.edp_samples = asmt_B.edp_samples
        asmt_C.cmp_quant_RV = asmt_B.cmp_quant_RV
        asmt_C.cmp_fragility_RV = asmt_B.cmp_fragility_RV
        asmt_C.calc_cmp_damage()
        asmt_C.cmp_damage_consequence_RV = asmt_B.cmp_damage_consequence_RV
        asmt_C.cmp_cost_RV = asmt_B.cmp_cost_RV
        asmt_C.calc_cmp_dmg_quant()
        temp = asmt_B.cmp_cost_RV.copy()
        temp.loc[:, ('replacement', '0', '1', '0', 'DS1', '1')] = \
            asmt_A.cmp_cost_RV.loc[:, ('replacement', '0', '1', '0', 'DS1', '1')]
        asmt_C.cmp_cost_RV = temp
        asmt_C.calc_cmp_cost()
        asmt_C.calc_total_cost()

        asmt_D.edp_samples = asmt_A.edp_samples
        asmt_D.cmp_quant_RV = asmt_A.cmp_quant_RV
        asmt_D.cmp_fragility_RV = asmt_A.cmp_fragility_RV
        asmt_D.calc_cmp_damage()
        asmt_D.cmp_damage_consequence_RV = asmt_A.cmp_damage_consequence_RV
        asmt_D.cmp_cost_RV = asmt_A.cmp_cost_RV
        asmt_D.calc_cmp_dmg_quant()
        temp = asmt_A.cmp_cost_RV.copy()
        temp.loc[:, ('replacement', '0', '1', '0', 'DS1', '1')] = \
            asmt_B.cmp_cost_RV.loc[:, ('replacement', '0', '1', '0', 'DS1', '1')]
        asmt_D.cmp_cost_RV = temp
        asmt_D.calc_cmp_cost()
        asmt_D.calc_total_cost()

    elif rv_group == 'cmp_dm':

        asmt_C.edp_samples = asmt_B.edp_samples
        asmt_C.cmp_quant_RV = asmt_B.cmp_quant_RV
        temp = asmt_A.cmp_fragility_RV.copy()
        subset_cols = [('collapse', '0', '1', '0', 'DS1'),
                       ('irreparable', '0', '1', '0', 'DS1')]
        temp.loc[:, subset_cols] = \
            asmt_B.cmp_fragility_RV.loc[:, subset_cols]
        asmt_C.cmp_fragility_RV = temp
        asmt_C.cmp_damage_consequence_RV = asmt_A.cmp_damage_consequence_RV
        asmt_C.cmp_cost_RV = asmt_B.cmp_cost_RV
        asmt_C.calc_cmp_damage()
        asmt_C.calc_cmp_dmg_quant()
        asmt_C.calc_cmp_cost()
        asmt_C.calc_total_cost()

        asmt_D.edp_samples = asmt_A.edp_samples
        asmt_D.cmp_quant_RV = asmt_A.cmp_quant_RV
        temp = asmt_B.cmp_fragility_RV.copy()
        subset_cols = [('collapse', '0', '1', '0', 'DS1'),
                       ('irreparable', '0', '1', '0', 'DS1')]
        temp.loc[:, subset_cols] = \
            asmt_A.cmp_fragility_RV.loc[:, subset_cols]
        asmt_D.cmp_fragility_RV = temp
        asmt_D.cmp_damage_consequence_RV = asmt_B.cmp_damage_consequence_RV
        asmt_D.cmp_cost_RV = asmt_A.cmp_cost_RV
        asmt_D.calc_cmp_damage()
        asmt_D.calc_cmp_dmg_quant()
        asmt_D.calc_cmp_cost()
        asmt_D.calc_total_cost()


    elif rv_group == 'cmp_dv':

        asmt_C.edp_samples = asmt_B.edp_samples
        asmt_C.cmp_quant_RV = asmt_B.cmp_quant_RV
        asmt_C.cmp_fragility_RV = asmt_B.cmp_fragility_RV
        asmt_C.cmp_damage_consequence_RV = asmt_B.cmp_damage_consequence_RV
        temp = asmt_A.cmp_cost_RV.copy()
        temp.loc[:, ('replacement', '0', '1', '0', 'DS1', '1')] = \
            asmt_B.cmp_cost_RV.loc[:, ('replacement', '0', '1', '0', 'DS1', '1')]
        asmt_C.cmp_cost_RV = temp
        asmt_C.calc_cmp_damage()
        asmt_C.calc_cmp_dmg_quant()
        asmt_C.calc_cmp_cost()
        asmt_C.calc_total_cost()

        asmt_D.edp_samples = asmt_A.edp_samples
        asmt_D.cmp_quant_RV = asmt_A.cmp_quant_RV
        asmt_D.cmp_fragility_RV = asmt_A.cmp_fragility_RV
        asmt_D.cmp_damage_consequence_RV = asmt_A.cmp_damage_consequence_RV
        temp = asmt_B.cmp_cost_RV.copy()
        temp.loc[:, ('replacement', '0', '1', '0', 'DS1', '1')] = \
            asmt_A.cmp_cost_RV.loc[:, ('replacement', '0', '1', '0', 'DS1', '1')]
        asmt_D.cmp_cost_RV = temp
        asmt_D.calc_cmp_damage()
        asmt_D.calc_cmp_dmg_quant()
        asmt_D.calc_cmp_cost()
        asmt_D.calc_total_cost()

    elif rv_group == 'cmp_quant':

        asmt_C.edp_samples = asmt_B.edp_samples
        asmt_C.cmp_quant_RV = asmt_A.cmp_quant_RV
        asmt_C.cmp_fragility_RV = asmt_B.cmp_fragility_RV
        asmt_C.cmp_damage_consequence_RV = asmt_B.cmp_damage_consequence_RV
        asmt_C.cmp_cost_RV = asmt_B.cmp_cost_RV
        asmt_C.calc_cmp_damage()
        asmt_C.calc_cmp_dmg_quant()
        asmt_C.calc_cmp_cost()
        asmt_C.calc_total_cost()

        asmt_D.edp_samples = asmt_A.edp_samples
        asmt_D.cmp_quant_RV = asmt_B.cmp_quant_RV
        asmt_D.cmp_fragility_RV = asmt_A.cmp_fragility_RV
        asmt_D.cmp_damage_consequence_RV = asmt_A.cmp_damage_consequence_RV
        asmt_D.cmp_cost_RV = asmt_A.cmp_cost_RV
        asmt_D.calc_cmp_damage()
        asmt_D.calc_cmp_dmg_quant()
        asmt_D.calc_cmp_cost()
        asmt_D.calc_total_cost()

    elif rv_group == 'edp':

        asmt_C.edp_samples = asmt_A.edp_samples.copy()
        asmt_C.edp_samples.loc[:, ('SA_0.82', '0', '1')] = \
            asmt_B.edp_samples.loc[:, ('SA_0.82', '0', '1')]
        asmt_C.cmp_quant_RV = asmt_B.cmp_quant_RV
        asmt_C.cmp_fragility_RV = asmt_B.cmp_fragility_RV
        asmt_C.cmp_damage_consequence_RV = asmt_B.cmp_damage_consequence_RV
        asmt_C.cmp_cost_RV = asmt_B.cmp_cost_RV
        asmt_C.calc_cmp_damage()
        asmt_C.calc_cmp_dmg_quant()
        asmt_C.calc_cmp_cost()
        asmt_C.calc_total_cost()

        asmt_D.edp_samples = asmt_B.edp_samples.copy()
        asmt_D.edp_samples.loc[:, ('SA_0.82', '0', '1')] = \
            asmt_A.edp_samples.loc[:, ('SA_0.82', '0', '1')]
        asmt_D.cmp_quant_RV = asmt_A.cmp_quant_RV
        asmt_D.cmp_fragility_RV = asmt_A.cmp_fragility_RV
        asmt_D.cmp_damage_consequence_RV = asmt_A.cmp_damage_consequence_RV
        asmt_D.cmp_cost_RV = asmt_A.cmp_cost_RV
        asmt_D.calc_cmp_damage()
        asmt_D.calc_cmp_dmg_quant()
        asmt_D.calc_cmp_cost()
        asmt_D.calc_total_cost()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    #                     Sensitivity Indices                        #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    yA = asmt_A.total_cost.to_numpy()
    yB = asmt_B.total_cost.to_numpy()
    yC = asmt_C.total_cost.to_numpy()
    yD = asmt_D.total_cost.to_numpy()

    results_df = pd.DataFrame({'A': yA, 'B': yB, 'C': yC, 'D': yD},
                              index=range(num_realizations))

    if not os.path.exists(f'{analysis_output_path}/{modeling_uncertainty_case}/{replacement_threshold}/{rv_group}'):
        os.makedirs(f'{analysis_output_path}/{modeling_uncertainty_case}/{replacement_threshold}/{rv_group}')

    # file output
    results_df.to_csv(f'{analysis_output_path}/{modeling_uncertainty_case}/{replacement_threshold}/{rv_group}/total_cost_realizations.csv')

    s1, sT = calc_sens(yA, yB, yC, yD)

    # bootstrap
    # debug
    num_repeats = 5
    # num_repeats = 5000
    bootstrap_sample_s1 = np.zeros(num_repeats)
    bootstrap_sample_sT = np.zeros(num_repeats)
    for j in range(num_repeats):
        sel = np.random.choice(num_realizations, num_realizations)
        res = calc_sens(yA[sel], yB[sel], yC[sel], yD[sel])
        bootstrap_sample_s1[j] = res[0]
        bootstrap_sample_sT[j] = res[1]
    mean_s1 = bootstrap_sample_s1.mean()
    mean_sT = bootstrap_sample_sT.mean()
    std_s1 = bootstrap_sample_s1.std()
    std_sT = bootstrap_sample_sT.std()
    if np.abs(std_s1) < 1e-10:
        conf_int_s1 = (mean_s1, mean_s1)
    else:
        conf_int_s1 = (stats.norm.ppf(0.025, mean_s1, std_s1),
                       stats.norm.ppf(0.975, mean_s1, std_s1))
    if np.abs(std_sT) < 1e-10:
        conf_int_sT = (mean_sT, mean_sT)
    else:
        conf_int_sT = (stats.norm.ppf(0.025, mean_sT, std_sT),
                       stats.norm.ppf(0.975, mean_sT, std_sT))

    sens_results_df = pd.DataFrame(
        {'s1': s1, 'sT': sT, 's1_CI_l': conf_int_s1[0], 's1_CI_h': conf_int_s1[1],
         'sT_CI_l': conf_int_sT[0], 'sT_CI_h': conf_int_sT[1]},
        index=[rv_group])
    sens_results_df.columns.name = 'Sensitivity Index'


    if not os.path.exists(f'{analysis_output_path}/{modeling_uncertainty_case}/{replacement_threshold}/{rv_group}'):
        os.makedirs(f'{analysis_output_path}/{modeling_uncertainty_case}/{replacement_threshold}/{rv_group}')

    # file output
    sens_results_df.to_csv(
        f'{analysis_output_path}/{modeling_uncertainty_case}/{replacement_threshold}/{rv_group}/sensitivity_indices.csv')
