# %% Imports #

import numpy as np
import argparse
import os
import pandas as pd
from scipy.interpolate import interp1d

# ~~~~~~~~~~~~~~~~~~~~~~ #
# set up argument parser #
# ~~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--output_dir')
parser.add_argument('--num_levels')
parser.add_argument('--num_inputs')
parser.add_argument('--t_1')
parser.add_argument('--yield_dr')

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
num_levels = int(args.num_levels)
num_inputs = int(args.num_inputs)
t_1 = float(args.t_1)
yield_dr = float(args.yield_dr)

response_dirs = [f'%s/gm%i' % (input_dir, i+1)
                 for input_dir, i in zip([input_dir]*num_inputs,
                                         range(num_inputs))]

# ~~~~~~~~~~ #
# parameters #
# ~~~~~~~~~~ #

# yield global drift in the X and Y direction
yield_drift = {}
yield_drift[1] = yield_dr
yield_drift[2] = yield_dr


# ~~~~ #
# main #
# ~~~~ #

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

first_column = np.arange(1, num_inputs+1)

edps = {}

# peak interstory drift data
for level in range(1, num_levels+1):
    for direction in range(1, 3):
        tag = f'PID-{level}-{direction}'
        peak_edp = []
        for j, response_dir in enumerate(response_dirs):
            response_file = \
                f'{response_dir}/ID-{level}-{direction}.csv'
            contents = pd.read_csv(response_file, engine='pyarrow').to_numpy()
            peak = np.max(np.abs(contents))
            peak_edp.append(peak)
        edps[tag] = np.array(peak_edp)

# peak floor velocity data
for level in range(num_levels+1):
    for direction in range(1, 3):
        tag = 'PFV-'+str(level)+'-'+str(direction)
        peak_edp = []
        for j, response_dir in enumerate(response_dirs):
            response_file = \
                f'{response_dir}/FV-{level}-{direction}.csv'
            contents = pd.read_csv(response_file, engine='pyarrow').to_numpy()
            peak = np.max(np.abs(contents))
            peak_edp.append(peak)
        edps[tag] = np.array(peak_edp)

# peak floor acceleration data
for level in range(num_levels+1):
    for direction in range(1, 3):
        tag = f'PFA-{level}-{direction}'
        peak_edp = []
        for j, response_dir in enumerate(response_dirs):
            response_file = \
                f'{response_dir}/FA-{level}-{direction}.csv'
            contents = pd.read_csv(response_file, engine='pyarrow').to_numpy()
            # convert to G units
            peak = np.max(np.abs(contents)) / 386.22
            peak_edp.append(peak)
        edps[tag] = np.array(peak_edp)

# RotD50 Sa at T_1
rs_paths = [
    f"{input_dir.replace('response', 'ground_motions')}/{i+1}RS.txt"
    for i in range(num_inputs)]
psas = np.full(num_inputs, 0.00)
for i, rs_path in enumerate(rs_paths):
    rs = pd.read_csv(rs_path, delimiter=' ', engine='pyarrow').to_numpy()
    f_rs = interp1d(rs[:, 0], rs[:, 1], bounds_error=False, fill_value=0.00)
    psas[i] = float(f_rs(t_1))
edps[f'SA_{t_1}-0-1'] = psas

# residual drift (FEMA P-58-1 commentary C)
# (but here we consider the building as a whole instead of individual stories)
tag = f'RID-0-1'
peak_edp = {1: [], 2: []}
for direction in range(1, 3):
    for j, response_dir in enumerate(response_dirs):
        response_file = \
            f'{response_dir}/BD-{direction}.csv'
        contents = pd.read_csv(response_file, engine='pyarrow').to_numpy()
        peak = np.max(np.abs(contents))
        if peak < yield_drift[direction]:
            peak_edp[direction].append(peak/1e4)
        elif peak < 4.0 * yield_drift[direction]:
            peak_edp[direction].append(0.3 * (peak - yield_drift[direction]))
        else:
            peak_edp[direction].append((peak - 3.00 * yield_drift[direction]))
peak_edp_max = np.maximum(peak_edp[1], peak_edp[2])
edps[tag] = peak_edp_max

df = pd.DataFrame(edps, index=first_column)
units = pd.DataFrame({'units': ['rad']*num_levels*2 +
                      ['inchps']*(num_levels+1)*2 +
                      ['g']*(num_levels+1)*2 +
                      ['g'] +
                      ['rad']},
                     index=df.columns.values)
df = pd.concat([units.T, df], axis=0)
df.index.name = 'Run #'

df.to_csv(output_dir+'/response.csv')
