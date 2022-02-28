
# pelicun


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

# debug
response_path = 'analysis/hazard_level_8/response_summary/response.csv'
# output_directory = 'analysis/hazard_level_8/performance/0'
c_edp_stdev = 1.00        # uncertainty in building response
c_quant_stdev = 1.00      # uncertainty in the component quantities
c_dm_stdev = 1.00         # in the fragility curves
c_collapse_irrep_stdev = 1.00  # in the collapse or irreparable fragilities
c_dv_stdev = 1.00         # in the damage consequences
c_replace_stdev = 1.00    # in the building replacement cost


# ~~~~~~~~~~~~~ #
# configuration #
# ~~~~~~~~~~~~~ #

num_realizations = 10000
num_stories = 3

component_model_prefix = 'src/performance_data/CMP'

loss_map = 'src/performance_data/LOSS_map.csv'

config_opt = {
    "Verbose": True,
    "PrintLog": False,
    "LogShowMS": False,
    "EconomiesOfScale": {
        "AcrossFloors": True,
        "AcrossDamageStates": True},
    "SamplingMethod": "MonteCarlo"}

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
A.demand.generate_sample({"SampleSize": num_realizations, 'method': 'LHS_midpoint'})

A.demand.save_sample(f'src/new_perf/PELICUN_EDP_sample.csv')


# ----------------- #
# Damage Assessment #
# ----------------- #
A.asset.load_cmp_model(file_prefix=component_model_prefix)

# generate component quantity sample
A.asset.generate_cmp_sample(num_realizations, c_quant_stdev)

# save the quantity sample to a file
# A.asset.save_cmp_sample(f'{output_directory}/CMP_sample.csv')
# A.asset.load_cmp_sample(f'{output_directory}/CMP_sample.csv')

# A.demand.load_sample('tmp/simple_pelicun/out/EDP_sample.csv')

# load the fragility information
A.damage.load_fragility_model(fragility_data_sources)

# calculate damages

# if pipes break, flooding occurs
# dmg_process = {
#     "1_D2021.011a": {
#         "DS2": "C3021.001k_DS1"

#     },
#     "2_D4011.021a": {
#         "DS2": "C3021.001k_DS1"
#     }
# }
dmg_process=None
A.damage.calculate(num_realizations, dmg_process=dmg_process,
                   c_dm_stdev=c_dm_stdev,
                   c_collapse_irrep_stdev=c_collapse_irrep_stdev)


# save the damage sample to a file
# A.damage.save_sample(f'{output_directory}/DMG_sample.csv')

 
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

# A.bldg_repair.save_sample(f'{output_directory}/LOSS_repair.csv')

agg_DF = A.bldg_repair.aggregate_losses(A.damage, replacement_threshold)

file_io.save_to_csv(agg_DF, f'src/new_perf/Summary.csv')










vals = np.genfromtxt(f'src/new_perf/Summary.csv',
                     skip_header=1, delimiter=',')[:, 1]



f = plt.figure()
# sns.ecdfplot(ds1_c1_pel+ds1_c2_pel+ds2_c1_pel, label='pelicun', color='blue')
# sns.ecdfplot(ds1_c1_me+ds1_c2_me+ds2_c1_me, label='my code', color='orange')
sns.ecdfplot(vals, label='pelicun tot', color='tab:blue', linestyle='dashed')
sns.ecdfplot(total_cost_df, label='mine total', color='tab:orange', linestyle='dashed')
plt.show()
plt.close()









