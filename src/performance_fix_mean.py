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

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--response_path')
parser.add_argument('--rv_group')
parser.add_argument('--performance_data_path')
parser.add_argument('--analysis_output_path')
parser.add_argument('--figures_output_path')

args = parser.parse_args()
response_path = args.response_path
rv_group = args.rv_group
performance_data_path = args.performance_data_path
analysis_output_path = args.analysis_output_path
figures_output_path = args.figures_output_path

# response_path = 'analysis/office3/hazard_level_1/response_summary/response.csv'
# rv_group = 'edp'
# performance_data_path = 'src/performance_data_office3'
# analysis_output_path = 'analysis/office3/hazard_level_7/performance/test'
# figures_output_path = 'figures/office3/hazard_level_7/performance/test'

# ~~~~~~~~~~ #
# parameters #
# ~~~~~~~~~~ #

c_modeling_uncertainty = np.sqrt(0.10**2+0.10**2)
num_realizations = 20000
replacement_threshold = 0.40
perf_model_input_path = f'{performance_data_path}/input_cmp_quant.csv'
cmp_fragility_input_path = f'{performance_data_path}/input_fragility.csv'
cmp_repair_cost_input_path = f'{performance_data_path}/input_repair_cost.csv'

if not os.path.exists(analysis_output_path):
    os.makedirs(analysis_output_path)

if not os.path.exists(figures_output_path):
    os.makedirs(figures_output_path)

logging.basicConfig(
    filename=f'{analysis_output_path}/info_all.txt',
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('perf_fix_mean')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       Analysis A                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

logger.info('Starting analysis A')
if rv_group == 'bldg_dm':
    asmt_A = P58_Assessment(
        num_realizations=num_realizations,
        replacement_threshold=replacement_threshold, fix_blg_dm_mean=True)
elif rv_group == 'bldg_dv':
    asmt_A = P58_Assessment(
        num_realizations=num_realizations,
        replacement_threshold=replacement_threshold, fix_blg_dv_mean=True)
elif rv_group == 'cmp_dm':
    asmt_A = P58_Assessment(
        num_realizations=num_realizations,
        replacement_threshold=replacement_threshold, fix_cmp_dm_mean=True)
elif rv_group == 'cmp_dv':
    asmt_A = P58_Assessment(
        num_realizations=num_realizations,
        replacement_threshold=replacement_threshold, fix_cmp_dv_mean=True)
elif rv_group == 'cmp_quant':
    asmt_A = P58_Assessment(
        num_realizations=num_realizations,
        replacement_threshold=replacement_threshold, fix_quant_mean=True)
elif rv_group == 'edp':
    asmt_A = P58_Assessment(
        num_realizations=num_realizations,
        replacement_threshold=replacement_threshold, fix_edp_mean=True)
else:
    raise ValueError(f'Invalid rv_group variable: {rv_group}')

asmt_A.read_perf_model(perf_model_input_path)
asmt_A.read_fragility_input(cmp_fragility_input_path)
asmt_A.read_cmp_repair_cost_input(cmp_repair_cost_input_path)
asmt_A.run(response_path, c_modeling_uncertainty)
logger.info('\tAnalysis A finished')

yA = asmt_A.total_cost.to_numpy()

results_df = pd.DataFrame({'A': yA},
                          index=range(num_realizations))

results_df.to_csv(f'{analysis_output_path}/total_cost_realizations_{rv_group}.csv')
