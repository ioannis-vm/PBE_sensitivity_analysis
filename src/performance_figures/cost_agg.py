# %% Imports #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import argparse

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--DV_rec_cost_agg_path')
parser.add_argument('--output_path')

args = parser.parse_args()
DV_rec_cost_agg_path = args.DV_rec_cost_agg_path
output_path = args.output_path

# # debug
# DV_rec_cost_agg_path =\
#     'analysis/hazard_level_7/performance/A/DV_rec_cost_agg.csv'

# ~~~~ #
# main #
# ~~~~ #

df = pd.read_csv(
    DV_rec_cost_agg_path, delimiter=',', index_col=0)


fig = px.ecdf(df)
# fig.show()
fig.write_html(output_path)
