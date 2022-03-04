"""
Variance-based sensitivity analysis of the FEMA P-58 methodology
"""

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
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--response_path')
parser.add_argument('--analysis_output_path')
parser.add_argument('--figures_output_path')

args = parser.parse_args()
response_path = args.response_path
analysis_output_path = args.analysis_output_path
figures_output_path = args.figures_output_path

# # debug ~ these will be set by `make`
# response_path = 'analysis/hazard_level_1/response_summary/response.csv'
# analysis_output_path = 'analysis/hazard_level_1/performance/edp'
# figures_output_path = 'figures/hazard_level_1/performance/edp'


# ~~~~~~~~~~ #
# parameters #
# ~~~~~~~~~~ #

c_modeling_uncertainty = 0.00
num_realizations = 100000
replacement_threshold = 0.50
perf_model_input_path = 'src/performance_data/input_cmp_quant.csv'
cmp_fragility_input_path = 'src/performance_data/input_fragility.csv'
cmp_repair_cost_input_path = 'src/performance_data/input_repair_cost.csv'


if not os.path.exists(analysis_output_path):
    os.makedirs(analysis_output_path)

if not os.path.exists(figures_output_path):
    os.makedirs(figures_output_path)

logging.basicConfig(
    filename=f'{analysis_output_path}/info_all.txt',
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG)


logging.info('Start')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                    Reading Input Data                          #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

asmt_A = P58_Assessment(num_realizations=num_realizations)
asmt_A.read_perf_model(perf_model_input_path)
asmt_A.read_fragility_input(cmp_fragility_input_path)
asmt_A.read_cmp_repair_cost_input(cmp_repair_cost_input_path)

asmt_B = P58_Assessment(num_realizations=num_realizations)
asmt_B.read_perf_model(perf_model_input_path)
asmt_B.read_fragility_input(cmp_fragility_input_path)
asmt_B.read_cmp_repair_cost_input(cmp_repair_cost_input_path)

asmt_C = P58_Assessment(num_realizations=num_realizations)
asmt_C.read_perf_model(perf_model_input_path)
asmt_C.read_fragility_input(cmp_fragility_input_path)
asmt_C.read_cmp_repair_cost_input(cmp_repair_cost_input_path)

asmt_D = P58_Assessment(num_realizations=num_realizations)
asmt_D.read_perf_model(perf_model_input_path)
asmt_D.read_fragility_input(cmp_fragility_input_path)
asmt_D.read_cmp_repair_cost_input(cmp_repair_cost_input_path)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       Analysis A                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

logging.info('Starting analysis A')
asmt_A.run(response_path, c_modeling_uncertainty)
logging.info('\tAnalysis A finished')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       Analysis B                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

logging.info('Starting analysis B')
asmt_B.run(response_path, c_modeling_uncertainty)
logging.info('\tAnalysis A finished')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       Analysis C                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

logging.info('Starting analysis C')
asmt_C.edp_samples = asmt_B.edp_samples
asmt_C.cmp_quant_RV = asmt_B.cmp_quant_RV
temp = asmt_B.cmp_fragility_RV
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
logging.info('\tAnalysis C finished')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       Analysis D                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

logging.info('Starting analysis D')
asmt_D.edp_samples = asmt_A.edp_samples
asmt_D.cmp_quant_RV = asmt_A.cmp_quant_RV
temp = asmt_A.cmp_fragility_RV
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
logging.info('\tAnalysis D finished')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                     Sensitivity Indices                        #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

logging.info('Calculating sensitivity indices for the EDPs')
yA = asmt_A.total_cost.to_numpy()
yB = asmt_B.total_cost.to_numpy()
yC = asmt_C.total_cost.to_numpy()
yD = asmt_D.total_cost.to_numpy()

results_df = pd.DataFrame({'A': yA, 'B': yB, 'C': yC, 'D': yD},
                          index=range(num_realizations))


results_df.to_csv(f'{analysis_output_path}/total_cost_realizations.csv')


s1, sT = calc_sens(yA, yB, yC, yD)
# bootstrap
num_repeats = 10000
bootstrap_sample_s1 = np.zeros(num_repeats)
bootstrap_sample_sT = np.zeros(num_repeats)
sel = np.random.choice(num_realizations, (num_repeats, num_realizations))
for j in range(num_repeats):
    res = calc_sens(yA[sel[j]], yB[sel[j]], yC[sel[j]], yD[sel[j]])
    bootstrap_sample_s1[j] = res[0]
    bootstrap_sample_sT[j] = res[1]
mean_s1 = bootstrap_sample_s1.mean()
mean_sT = bootstrap_sample_sT.mean()
std_s1 = bootstrap_sample_s1.std()
std_sT = bootstrap_sample_sT.std()
conf_int_s1 = (stats.norm.ppf(0.025, mean_s1, std_s1),
               stats.norm.ppf(0.975, mean_s1, std_s1))
conf_int_sT = (stats.norm.ppf(0.025, mean_sT, std_s1),
               stats.norm.ppf(0.975, mean_sT, std_s1))

sens_results_df = pd.DataFrame(
    {'s1': s1, 'sT': sT, 's1_CI_l': conf_int_s1[0], 's1_CI_h': conf_int_s1[1],
     'sT_CI_l': conf_int_sT[0], 'sT_CI_h': conf_int_sT[1]},
    index=['EDP'])
sens_results_df.columns.name = 'Sensitivity Index'

sens_results_df.to_csv(f'{analysis_output_path}/sensitividy_indices.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.hist(bootstrap_sample_s1, 50, density=True,
         color='#eaeaea', edgecolor='black')
xv = np.linspace(np.min(bootstrap_sample_s1),
                 np.max(bootstrap_sample_s1), 10000)
yv = stats.norm.pdf(xv, mean_s1, std_s1)
ax1.plot(xv, yv, color='k')
ax1.set_xlabel('$s_1$')
ax2.hist(bootstrap_sample_sT, 50, density=True,
         color='#eaeaea', edgecolor='black')
xv = np.linspace(np.min(bootstrap_sample_sT),
                 np.max(bootstrap_sample_sT), 10000)
yv = stats.norm.pdf(xv, mean_sT, std_sT)
ax2.plot(xv, yv, color='k')
ax2.set_xlabel('$s_T$')
fig.suptitle('Bootstrap PDF of Sensitivity Indices\nBuilding DM RV group')
plt.savefig(f'{figures_output_path}/bootstrap_PDF.pdf')
plt.close()
