
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


# configuration #

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


beta_modeling = 0.
replacement_threshold = 0.50

base.set_options(config_opt)
A = assessment.Assessment()
A.stories = num_stories
A.demand.load_sample(response_path)
A.demand.calibrate_model(calibration_marginals, c_edp_stdev, beta_modeling)
A.demand.generate_sample({"SampleSize": num_realizations, 'method': 'LHS_midpoint'})
A.asset.load_cmp_model(file_prefix=component_model_prefix)
A.asset.generate_cmp_sample(num_realizations, c_quant_stdev)
A.damage.load_fragility_model(fragility_data_sources)
# if pipes break, flooding occurs
# dmg_process = {
#     "1_D2021.011a": {
#         "DS2": "C3021.001k_DS1"

#     },
#     "2_D4011.021a": {
#         "DS2": "C3021.001k_DS1"
#     }
# }
dmg_process = None
A.damage.calculate(num_realizations, dmg_process=dmg_process,
                   c_dm_stdev=c_dm_stdev,
                   c_collapse_irrep_stdev=c_collapse_irrep_stdev)
A.bldg_repair.load_model(
    repair_data_sources, loss_map, c_dv_stdev, c_replace_stdev)
A.bldg_repair.calculate(num_realizations)
agg_DF = A.bldg_repair.aggregate_losses(A.damage, replacement_threshold)
vals = agg_DF.loc[:, 'repair_cost']






# our code


import sys
sys.path.insert(0, "src_experimental")
from p_58_assessment import P58_Assessment


c_modeling_uncertainty = 0.00
num_realizations = 100000
replacement_threshold = 0.50
response_path = 'analysis/hazard_level_8/response_summary/response.csv'
perf_model_input_path = 'src_experimental/new_perf/input_cmp_quant.csv'
cmp_fragility_input_path = 'src_experimental/new_perf/input_fragility.csv'
cmp_repair_cost_input_path = 'src_experimental/new_perf/input_repair_cost.csv'


asmt = P58_Assessment(num_realizations=1000,
                      replacement_threshold=replacement_threshold)
asmt.read_perf_model(perf_model_input_path)
asmt.read_fragility_input(cmp_fragility_input_path)
asmt.read_cmp_repair_cost_input(cmp_repair_cost_input_path)
asmt.run(response_path, c_modeling_uncertainty)









# comparison

# x = A.bldg_repair.sample.loc[:, ('COST')].groupby(level='loss', axis=1).sum().mean(axis=0)

# y = asmt.cmp_cost.groupby(level='component', axis=1).sum().mean(axis=0)

# (x - y).plot.bar()
# plt.show()


# xx = A.bldg_repair.sample.loc[:, ('COST')].groupby(level='loss', axis=1).sum()

# yy = asmt.cmp_cost.groupby(level='component', axis=1).sum()



# f = plt.figure()
# sns.ecdfplot(xx.loc[:, 'D3041.041a'], label='pelicun tot', color='tab:blue', linestyle='dashed')
# sns.ecdfplot(yy.loc[:, 'D3041.041a'], label='mine total', color='tab:orange', linestyle='dashed')
# plt.show()
# plt.close()




f = plt.figure()
sns.ecdfplot(vals, label='pelicun tot', color='tab:blue', linestyle='dashed')
sns.ecdfplot(asmt.total_cost, label='mine total', color='tab:orange', linestyle='dashed')
plt.show()
plt.close()
