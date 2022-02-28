"""
Variance-based sensitivity analysis of the FEMA P-58 methodology
"""

import sys
sys.path.insert(0, "src_experimental")


import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import norm
from p_58_assessment import P58_Assessment

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

logging.basicConfig(
    filename=None,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG)

logging.info('Start')
logging.info('Defining functions and global variables')

c_modeling_uncertainty = 0.00
num_realizations = 100000
replacement_threshold = 0.50
perf_model_input_path = 'src_experimental/new_perf/input_cmp_quant.csv'
response_path = 'analysis/hazard_level_3/response_summary/response.csv'
cmp_fragility_input_path = 'src_experimental/new_perf/input_fragility.csv'
cmp_repair_cost_input_path = 'src_experimental/new_perf/input_repair_cost.csv'


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

# EDP
# logging.info('Starting analysis C')
# asmt_C.edp_samples = asmt_A.edp_samples
# asmt_C.cmp_quant_RV = asmt_B.cmp_quant_RV
# asmt_C.cmp_fragility_RV = asmt_B.cmp_fragility_RV
# asmt_C.cmp_damage_consequence_RV = asmt_B.cmp_damage_consequence_RV
# asmt_C.cmp_cost_RV = asmt_B.cmp_cost_RV
# asmt_C.calc_cmp_damage()
# asmt_C.calc_cmp_dmg_quant()
# asmt_C.calc_cmp_cost()
# asmt_C.calc_total_cost()
# logging.info('\tAnalysis C finished')


# component repair cost (cmp DV)
# logging.info('Starting analysis C')
# asmt_C.edp_samples = asmt_B.edp_samples
# asmt_C.cmp_quant_RV = asmt_B.cmp_quant_RV
# asmt_C.cmp_fragility_RV = asmt_B.cmp_fragility_RV
# asmt_C.calc_cmp_damage()
# asmt_C.cmp_damage_consequence_RV = asmt_B.cmp_damage_consequence_RV
# asmt_C.cmp_cost_RV = asmt_B.cmp_cost_RV
# asmt_C.calc_cmp_dmg_quant()
# temp = asmt_A.cmp_cost_RV
# temp.loc[:, ('replacement', '0', '1', '0', 'DS1', '1')] = \
#     asmt_B.cmp_cost_RV.loc[:, ('replacement', '0', '1', '0', 'DS1', '1')]
# asmt_C.cmp_cost_RV = temp
# asmt_C.calc_cmp_cost()
# asmt_C.calc_total_cost()
# logging.info('\tAnalysis C finished')


# component fragilities
logging.info('Starting analysis C')
asmt_C.edp_samples = asmt_B.edp_samples
asmt_C.cmp_quant_RV = asmt_B.cmp_quant_RV
temp = asmt_A.cmp_fragility_RV
subset_cols = [('collapse', '0', '1', '0', 'DS1'),
               ('irreparable', '0', '1', '0', 'DS1')]
temp.loc[:, subset_cols] = \
    asmt_B.cmp_fragility_RV.loc[:, subset_cols]
asmt_C.cmp_fragility_RV = temp
asmt_C.calc_cmp_damage()
asmt_C.cmp_damage_consequence_RV = asmt_A.cmp_damage_consequence_RV
asmt_C.calc_cmp_dmg_quant()
asmt_C.cmp_cost_RV = asmt_B.cmp_cost_RV
asmt_C.calc_cmp_cost()
asmt_C.calc_total_cost()
logging.info('\tAnalysis C finished')



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                     Sensitivity Indices                        #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

logging.info('Calculating sensitivity indices for the EDPs')
yA = asmt_A.total_cost.to_numpy()
yB = asmt_B.total_cost.to_numpy()
yC = asmt_C.total_cost.to_numpy()


def calc_sens(yA, yB, yC):
    n = len(yA)
    f0 = 1/n * np.sum(yA)
    s1 = (1./n * (yA.T @ yC) - f0**2) / (1./n * (yA.T @ yA) - f0**2)
    sT = 1. - (1/n * (yB.T @ yC) - f0**2) / \
        (1/n * (yA.T @ yA) - f0**2)
    return s1, sT


s1, sT = calc_sens(yA, yB, yC)
print()
print("EDPs")
print(f"1st order sensitivity index: {s1:.3f}")
print(f"Total sensitivity index    : {sT:.3f}")


num_repeats = 10000
sel = np.random.choice(num_realizations, (num_repeats, num_realizations))
bootstrap_sample = np.zeros(num_repeats)
for j in range(num_repeats):
    bootstrap_sample[j] = calc_sens(yA[sel[j]], yB[sel[j]], yC[sel[j]])[0]


from scipy import stats
fig = plt.figure(figsize=(6, 6))
plt.hist(bootstrap_sample, 50, density=True, color='lightgrey', edgecolor='black')
xv = np.linspace(np.min(bootstrap_sample), np.max(bootstrap_sample), 10000)
mub = bootstrap_sample.mean()
mus = bootstrap_sample.std()
yv = stats.norm.pdf(xv, loc=mub, scale=mus)
plt.plot(xv, yv)
plt.savefig('tmp_results/si1_cmp_DM_hz3.pdf')
plt.show()
plt.close()











# # now some quick check, will go back and fix code organization

# # component fragility uncertainty

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# #                       Analysis D                               #
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# logging.info('Starting analysis D')
# analysis_D = {}

# logging.info('\tUsing simulated demands from B')
# analysis_D['edp_RV'] = analysis_B['edp_RV']

# logging.info('\tUsing component quantities from B')
# analysis_D['cmp_quant_RV_df'] = analysis_B['cmp_quant_RV_df']

# logging.info('\tUsing component damage state thresholds from A')
# temp = analysis_A['cmp_fragility_RV_df']
# subset_cols = [('collapse', '0', '1', '0', 'DS1'),
#                ('irreparable', '0', '1', '0', 'DS1')]
# temp.loc[:, subset_cols] = \
#     analysis_B['cmp_fragility_RV_df'].loc[:, subset_cols]
# analysis_D['cmp_fragility_RV_df'] = temp

# logging.info('\tDetermining component damage')
# analysis_D['cmp_damage_df'] = calc_cmp_damage(
#     perf_model_df, cmp_fragility_input_df,
#     analysis_D['edp_RV'],
#     analysis_D['cmp_fragility_RV_df'])

# logging.info('\tDetermining damage consequences')
# analysis_D['cmp_damage_consequence_RV_df'] = \
#     analysis_A['cmp_damage_consequence_RV_df']
# analysis_D['cmp_dmg_quant_df'], analysis_D['cmp_dmg_quant_eco_df'] = \
#     calc_cmp_dmg_quant(
#         perf_model_df, cmp_repair_cost_input_df,
#         analysis_D['cmp_damage_df'],
#         analysis_D['cmp_damage_consequence_RV_df'],
#         analysis_D['cmp_quant_RV_df'])

# logging.info('\tDetermining component repair cost')
# analysis_D['cmp_cost_RV_df'] = analysis_B['cmp_cost_RV_df']
# analysis_D['cmp_cost_df'] = calc_cmp_cost(
#     analysis_D['cmp_dmg_quant_df'],
#     analysis_D['cmp_cost_RV_df'],
#     cmp_repair_cost_input_df,
#     analysis_D['cmp_dmg_quant_eco_df'])

# logging.info('\tSummarizing cost')
# analysis_D['total_cost_df'] = calc_total_cost(
#     analysis_D['cmp_cost_df'],
#     cmp_repair_cost_input_df,
#     analysis_D['cmp_cost_RV_df'])
# logging.info('\tAnalysis D finished')
































# logging.info('Calculating sensitivity indices for the EDPs')
# yA = analysis_A['total_cost_df'].to_numpy()
# yB = analysis_B['total_cost_df'].to_numpy()
# yC = analysis_D['total_cost_df'].to_numpy()


# def calc_sens(yA, yB, yC):
#     n = len(yA)
#     f0 = 1/n * np.sum(yA)
#     s1 = (1./n * (yA.T @ yC) - f0**2) / (1./n * (yA.T @ yA) - f0**2)
#     sT = 1. - (1/n * (yB.T @ yC) - f0**2) / \
#         (1/n * (yA.T @ yA) - f0**2)
#     return s1, sT


# s1, sT = calc_sens(yA, yB, yC)
# print()
# print("EDPs")
# print(f"1st order sensitivity index: {s1:.3f}")
# print(f"Total sensitivity index    : {sT:.3f}")



# num_repeats = 10000
# sel = np.random.choice(num_realizations, (num_repeats, num_realizations))
# bootstrap_sample = np.zeros(num_repeats)
# for j in range(num_repeats):
#     bootstrap_sample[j] = calc_sens(yA[sel[j]], yB[sel[j]], yC[sel[j]])[0]


# from scipy import stats
# plt.hist(bootstrap_sample, 50, density=True, color='lightgrey', edgecolor='black')
# xv = np.linspace(0.60, 0.70, 10000)
# mub = bootstrap_sample.mean()
# mus = bootstrap_sample.std()
# yv = stats.norm.pdf(xv, loc=mub, scale=mus)
# plt.plot(xv, yv)
# plt.show()

















































# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>component cost setup


# logging.info('Starting analysis D')
# analysis_D = {}

# logging.info('\tUsing simulated demands from B')
# analysis_D['edp_RV'] = analysis_B['edp_RV']

# logging.info('\tUsing component quantities from B')
# analysis_D['cmp_quant_RV_df'] = analysis_B['cmp_quant_RV_df']

# logging.info('\tUsing component damage state thresholds from B')
# analysis_D['cmp_fragility_RV_df'] = analysis_B['cmp_fragility_RV_df']

# logging.info('\tDetermining component damage')
# analysis_D['cmp_damage_df'] = calc_cmp_damage(
#     perf_model_df, cmp_fragility_input_df,
#     analysis_D['edp_RV'],
#     analysis_D['cmp_fragility_RV_df'])

# logging.info('\tDetermining damage consequences')
# analysis_D['cmp_damage_consequence_RV_df'] = \
#     analysis_B['cmp_damage_consequence_RV_df']
# analysis_D['cmp_dmg_quant_df'], analysis_D['cmp_dmg_quant_eco_df'] = \
#     calc_cmp_dmg_quant(
#         perf_model_df, cmp_repair_cost_input_df,
#         analysis_D['cmp_damage_df'],
#         analysis_D['cmp_damage_consequence_RV_df'],
#         analysis_D['cmp_quant_RV_df'])

# logging.info('\tDetermining component repair cost')
# temp = analysis_A['cmp_cost_RV_df']
# temp.loc[:, ('replacement', '0', '1', '0', 'DS1', '1')] = \
#     analysis_B['cmp_cost_RV_df'].loc[:, ('replacement', '0', '1', '0', 'DS1', '1')]
# analysis_D['cmp_cost_RV_df'] = temp
# analysis_D['cmp_cost_df'] = calc_cmp_cost(
#     analysis_D['cmp_dmg_quant_df'],
#     analysis_D['cmp_cost_RV_df'],
#     cmp_repair_cost_input_df,
#     analysis_D['cmp_dmg_quant_eco_df'])

# logging.info('\tSummarizing cost')
# analysis_D['total_cost_df'] = calc_total_cost(
#     analysis_D['cmp_cost_df'],
#     cmp_repair_cost_input_df,
#     analysis_D['cmp_cost_RV_df'])
# logging.info('\tAnalysis D finished')

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>component cost setup












# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>fragility setup

# logging.info('Starting analysis D')
# analysis_D = {}

# logging.info('\tUsing simulated demands from B')
# analysis_D['edp_RV'] = analysis_B['edp_RV']

# logging.info('\tUsing component quantities from B')
# analysis_D['cmp_quant_RV_df'] = analysis_B['cmp_quant_RV_df']

# logging.info('\tUsing component damage state thresholds from A')
# temp = analysis_A['cmp_fragility_RV_df']
# subset_cols = [('collapse', '0', '1', '0', 'DS1'),
#                ('irreparable', '0', '1', '0', 'DS1')]
# temp.loc[:, subset_cols] = \
#     analysis_B['cmp_fragility_RV_df'].loc[:, subset_cols]
# analysis_D['cmp_fragility_RV_df'] = temp

# logging.info('\tDetermining component damage')
# analysis_D['cmp_damage_df'] = calc_cmp_damage(
#     perf_model_df, cmp_fragility_input_df,
#     analysis_D['edp_RV'],
#     analysis_D['cmp_fragility_RV_df'])

# logging.info('\tDetermining damage consequences')
# analysis_D['cmp_damage_consequence_RV_df'] = \
#     analysis_A['cmp_damage_consequence_RV_df']
# analysis_D['cmp_dmg_quant_df'], analysis_D['cmp_dmg_quant_eco_df'] = \
#     calc_cmp_dmg_quant(
#         perf_model_df, cmp_repair_cost_input_df,
#         analysis_D['cmp_damage_df'],
#         analysis_D['cmp_damage_consequence_RV_df'],
#         analysis_D['cmp_quant_RV_df'])

# logging.info('\tDetermining component repair cost')
# analysis_D['cmp_cost_RV_df'] = analysis_B['cmp_cost_RV_df']
# analysis_D['cmp_cost_df'] = calc_cmp_cost(
#     analysis_D['cmp_dmg_quant_df'],
#     analysis_D['cmp_cost_RV_df'],
#     cmp_repair_cost_input_df,
#     analysis_D['cmp_dmg_quant_eco_df'])

# logging.info('\tSummarizing cost')
# analysis_D['total_cost_df'] = calc_total_cost(
#     analysis_D['cmp_cost_df'],
#     cmp_repair_cost_input_df,
#     analysis_D['cmp_cost_RV_df'])
# logging.info('\tAnalysis D finished')



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>fragility setup



































analysis_A['total_cost_df'].describe()
