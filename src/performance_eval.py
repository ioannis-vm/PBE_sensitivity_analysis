"""
Performance evaluation using pelicun
"""

import sys
sys.path.insert(0, "pelicun")


import numpy as np
import time
import pickle
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from pelicun import base
from pelicun import assessment
from pelicun import file_io


# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--response_path')
parser.add_argument('--output_directory')
parser.add_argument('--c_edp_stdev')
parser.add_argument('--c_quant_stdev')
parser.add_argument('--c_dm_stdev')
parser.add_argument('--c_collapse_stdev')
parser.add_argument('--c_dv_stdev')
parser.add_argument('--c_replace_stdev')

args = parser.parse_args()

response_path = args.response_path
output_directory = args.output_directory
c_edp_stdev = float(args.c_edp_stdev)
c_quant_stdev = float(args.c_quant_stdev)
c_dm_stdev = float(args.c_dm_stdev)
c_collapse_irrep_stdev = float(args.c_collapse_stdev)
c_dv_stdev = float(args.c_dv_stdev)
c_replace_stdev = float(args.c_replace_stdev)

# # debug
# response_path = 'analysis/hazard_level_8/response_summary/response.csv'
# output_directory = 'analysis/hazard_level_8/performance/0'
# c_edp_stdev = 1.00        # uncertainty in building response
# c_quant_stdev = 1.00      # uncertainty in the component quantities
# c_dm_stdev = 1.00         # in the fragility curves
# c_collapse_irrep_stdev = 1.00  # in the collapse or irreparable fragilities
# c_dv_stdev = 1.00         # in the damage consequences
# c_replace_stdev = 1.00    # in the building replacement cost
# c_edp_stdev = 0.00001        # uncertainty in building response
# c_quant_stdev = 0.00001      # uncertainty in the component quantities
# c_dm_stdev = 0.00001         # in the fragility curves
# c_collapse_irrep_stdev = 0.00001  # in the collapse or irreparable fragilities
# c_dv_stdev = 0.00001         # in the damage consequences
# c_replace_stdev = 0.00001    # in the building replacement cost

data = {
    'value': [
        c_edp_stdev, c_quant_stdev, c_dm_stdev,
        c_collapse_irrep_stdev, c_dv_stdev, c_replace_stdev
    ]
}

df = pd.DataFrame(data, index=['edp', 'quant', 'dm', 'collapse', 'dv', 'replace'])

df.to_csv(f'{output_directory}/variable_vals.txt')

# ~~~~~~~~~~~~~ #
# configuration #
# ~~~~~~~~~~~~~ #

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

num_realizations = 10000
num_stories = 3

component_model_prefix = 'src/performance_data/CMP'

loss_map = 'src/performance_data/LOSS_map.csv'

config_opt = {
    "Verbose": True,
    "Seed": 612,
    "PrintLog": False,
    "LogShowMS": False,
    "EconomiesOfScale": {
        "AcrossFloors": False,
        "AcrossDamageStates": False}}

calibration_marginals = {
    "ALL": {"DistributionFamily": "lognormal"},
    "PID": {"DistributionFamily": "lognormal",
            "TruncateAt": [None, None],
            "AddUncertainty": None},
    "PFA": {"DistributionFamily": "lognormal",
            "TruncateAt": [None, None],
            "AddUncertainty": None,
            "Unit": "inchps2"},
    "RID": {"DistributionFamily": "lognormal"}}

fragility_data_sources = [
    "src/performance_data/fragility_Additional.csv",
    "src/performance_data/resources_modified/fragility_DB_FEMA_P58_2nd.csv"]

repair_data_sources = [
    "src/performance_data/repair_Additional.csv",
    "src/performance_data/resources_modified/bldg_repair_DB_FEMA_P58_2nd.csv"]


# ------------- #
# Initial setup #
# ------------- #

# parameters

# additional modeling uncertainty (FEMA P-58 vol1 sec 5.2.6)
# beta_modeling = np.sqrt(0.10**2 + 0.10**2)
beta_modeling = 0.
# building replacement loss threshold (FEMA P-58 vol1 sec 3.2)
replacement_threshold = 0.50


base.set_options(config_opt)

A = assessment.Assessment()

A.stories = num_stories


# ----------------- #
# Demand Assessment #
# ----------------- #

# load demand samples to serve as reference data
A.demand.load_sample(response_path)

# calibrate the demand model
A.demand.calibrate_model(calibration_marginals, c_edp_stdev, beta_modeling)

# save the model to files
# A.demand.save_model(f'{output_directory}/EDP')
# A.demand.load_model(f'{output_directory}/EDP')

# generate demand sample
A.demand.generate_sample({"SampleSize": num_realizations})

A.demand.save_sample(f'{output_directory}/EDP_sample.csv')


# ----------------- #
# Damage Assessment #
# ----------------- #
A.asset.load_cmp_model(file_prefix=component_model_prefix)

# generate component quantity sample
A.asset.generate_cmp_sample(num_realizations, c_quant_stdev)

# save the quantity sample to a file
A.asset.save_cmp_sample(f'{output_directory}/CMP_sample.csv')
# A.asset.load_cmp_sample(f'{output_directory}/CMP_sample.csv')

# A.demand.load_sample('tmp/simple_pelicun/out/EDP_sample.csv')

# load the fragility information
A.damage.load_fragility_model(fragility_data_sources)

# calculate damages

# if pipes break, flooding occurs
dmg_process = {
    "1_D2021.011a": {
        "DS2": "C3021.001k_DS1"
    },
    "2_D4011.021a": {
        "DS2": "C3021.001k_DS1"
    }
}
A.damage.calculate(num_realizations, dmg_process=dmg_process,
                   c_dm_stdev=c_dm_stdev,
                   c_collapse_irrep_stdev=c_collapse_irrep_stdev)

# save the damage sample to a file
A.damage.save_sample(f'{output_directory}/DMG_sample.csv')

 
# --------------- #
# Loss Assessment #
# --------------- #

# load the demands from file
# (if not, we assume there is a preceding demand assessment with sampling)
# A.demand.load_sample(f'{output_directory}/EDP_sample.csv')

# load the component data from file
# (if not, we assume there is a preceding assessment
#  with component sampling)
# A.asset.load_cmp_sample(f'{output_directory}/CMP_sample.csv')

# load the damage from file
# (if not, we assume there is a preceding damage assessment with sampling)
# A.damage.load_sample(f'{output_directory}/DMG_sample.csv')

# calculate repair consequences
A.bldg_repair.load_model(
    repair_data_sources, loss_map, c_dv_stdev, c_replace_stdev)

A.bldg_repair.calculate(num_realizations)

A.bldg_repair.save_sample(f'{output_directory}/LOSS_repair.csv')

agg_DF = A.bldg_repair.aggregate_losses(A.damage, replacement_threshold)

file_io.save_to_csv(agg_DF, f'{output_directory}/Summary.csv')


















# these shoudn't be here...

# # -------- #
# # loss cdf #
# # -------- #

# vals = np.genfromtxt(f'{output_directory}/Summary.csv',
#                      skip_header=1, delimiter=',')[:, 1]

# m1 = np.mean(vals)
# s1 = np.std(vals)

# print(f'mean = {m1/1e6:.2f} mil')
# print(f'stdv = {s1/1e6:.2f} mil')

# f = plt.figure(figsize=(8, 2))
# gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])
# ax1 = plt.subplot(gs[0])
# ax2 = plt.subplot(gs[1])
# sns.ecdfplot(vals, label='label', color='tab:blue', ax=ax1)
# ax1.axvline(x=m1, color='tab:blue')
# ax1.set(xlabel='Cost c')
# ax1.set(ylabel='Probability (cost < c)')
# ax1.set(title='CDF of total repair cost')
# # ax1.set(xlim=(0.00, 12e6))
# ax1.legend()
# ax2.set(ylabel='Standard Deviation')
# ax2.bar(['label'], [s1], width=0.60,
#         edgecolor=['tab:blue', 'tab:orange'], fill=False)
# plt.subplots_adjust(wspace=0.4)
# # plt.savefig(output_path)
# plt.show()
# plt.close()


# # --------- #
# # loss cdfs #
# # --------- #

# output_dirs = [f'analysis/hazard_level_{i}/performance/A' for i in range(1, 9)]

# vals = [np.genfromtxt(f'{output_directory}/Summary.csv',
#                       skip_header=1, delimiter=',')[:, 1]
#         for output_directory in output_dirs]


# from statsmodels.distributions.empirical_distribution import ECDF

# ecdfs = [ECDF(x) for x in vals]

# return_periods = np.genfromtxt('analysis/site_hazard/Hazard_Curve_Interval_Data.csv')[:, -1]

# delta_e = np.genfromtxt('analysis/site_hazard/Hazard_Curve_Interval_Data.csv')[:, 1]
# delta_lamda = np.genfromtxt('analysis/site_hazard/Hazard_Curve_Interval_Data.csv')[:, 2]
# ratios = delta_lamda / delta_e

# costs = np.linspace(0.00, 2.00*10500000.00, 1000)


# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"

# fig, ax = plt.subplots(figsize=(8, 5))
# for i in range(8):
#     ax.plot(costs/10500000.00, ecdfs[i](costs),
#             label=f'T = {return_periods[i]:.0f} y')
# ax.plot(costs/10500000, sum([ecdfs[i](costs) * ratios[i] for i in range(8)]))
# ax.axvline(x=1.00, color='tab:grey', linestyle='dashed')
# ax.axvline(x=0.50,  color='tab:grey', linestyle='dashed')
# ax.set(xlabel='Cost ratio c')
# ax.set(ylabel='Probability (cost ratio < c)')
# ax.set(title='CDF of repair cost ratio')
# ax.set(xlim=(0.00, 2))
# ax.legend(loc='lower right')
# plt.subplots_adjust(right=0.8)
# # plt.savefig('result1.pdf')
# plt.show()
# plt.close()



 
# # investigate output

# df_cmp = pd.read_csv(f'{output_directory}/CMP_sample.csv', index_col=0)
# df_dmg = pd.read_csv(f'{output_directory}/DMG_sample.csv', index_col=0)
# df_edp = pd.read_csv(f'{output_directory}/EDP_sample.csv', index_col=0)
# df_loss = pd.read_csv(f'{output_directory}/LOSS_repair.csv', index_col=0)

# c1 = df_dmg.columns.values
# c2 = df_loss.columns.values

# for i in range(len(c2)):
#     # print(c1[i])
#     print("-".join(c2[i].replace('COST-', '').split("-")[1::]), sep='\t\t')
