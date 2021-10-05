# %% Imports #

import numpy as np
import argparse
import os
import glob
import pandas as pd

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--output_dir')
parser.add_argument('--num_levels')

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
num_levels = int(args.num_levels)

# ~~~~ #
# main #
# ~~~~ #

g_accel = 386.

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


response_dirs = sorted(glob.glob(input_dir+'/*/'))
num_earthquakes = len(response_dirs)

first_column = np.arange(1, num_earthquakes+1)

edps = {}

edps['Run #'] = first_column

# gather peak interstory drift data
for level in range(1, num_levels+1):
    for direction in range(1, 3):
        tag = '1-PID-'+str(level)+'-'+str(direction)
        peak_edp = []
        for j, response_dir in enumerate(response_dirs):
            response_file = \
                response_dir+'/'+'ID'+'-'+str(level)+'-'+str(direction)+'.csv'
            contents = np.genfromtxt(response_file)
            peak = np.max(np.abs(contents))
            peak_edp.append(peak)
        edps[tag] = np.array(peak_edp)

# # gather peak floor velocity data
# for level in range(1, num_levels+1):
#     for direction in range(1, 3):
#         tag = '1-PFV-'+str(level)+'-'+str(direction)
#         peak_edp = []
#         for j, response_dir in enumerate(response_dirs):
#             response_file = \
#                 response_dir+'/'+'FV'+'-'+str(level)+'-'+str(direction)+'.csv'
#             contents = np.genfromtxt(response_file)
#             peak = np.max(np.abs(contents))
#             peak_edp.append(peak)
#         edps[tag] = np.array(peak_edp)

# gather peak floor acceleration data
for level in range(num_levels+1):
    for direction in range(1, 3):
        tag = '1-PFA-'+str(level)+'-'+str(direction)
        peak_edp = []
        for j, response_dir in enumerate(response_dirs):
            response_file = \
                response_dir+'/'+'FA'+'-'+str(level)+'-'+str(direction)+'.csv'
            contents = np.genfromtxt(response_file)
            peak = np.max(np.abs(contents))
            peak_edp.append(peak * g_accel)
        edps[tag] = np.array(peak_edp)


df = pd.DataFrame.from_dict(edps)
df.to_csv(output_dir+'/response.csv', index=False)
