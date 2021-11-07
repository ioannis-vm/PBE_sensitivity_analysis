# %% Imports #

import numpy as np
import matplotlib.pyplot as plt
import argparse

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--fig_type')
parser.add_argument('--output_filename')
parser.add_argument('--num_levels')

args = parser.parse_args()
input_dir = args.input_dir
fig_type = args.fig_type
output_filename = args.output_filename
num_levels = int(args.num_levels)

# ~~~~ #
# main #
# ~~~~ #

if fig_type == 'FA':
    num_rows = num_levels + 1
    level_list = np.arange(0, num_levels + 1)
    shift = 0
elif fig_type in ['ID', 'FV']:
    num_rows = num_levels
    level_list = np.arange(1, num_levels + 1)
    shift = -1
else:
    raise ValueError('Unsupported figure type: ' + fig_type)

baseline = np.genfromtxt(input_dir + '/FA-0-1.csv')

response = {}
fig, axs = plt.subplots(nrows=num_rows, sharex=True)
for level in level_list:
    response[level] = []
    for direction in [1, 2]:
        response_file = input_dir + '/' + fig_type + '-' +\
            str(level) + '-' + str(direction) + '.csv'
        response[level].append(np.genfromtxt(response_file))
    if level == 0:
        dt = 0.005
    else:
        dt = 0.005 / len(response[level][0]) * len(baseline)
    time_vec = np.genfromtxt(input_dir + '/' + 'time.csv')
    axs[level+shift].plot(time_vec, response[level][0],
                          linewidth=2, color='red', alpha=0.5)
    axs[level+shift].plot(time_vec, response[level][1],
                          linewidth=2, color='green', alpha=0.5)
    axs[level+shift].grid()
plt.savefig(output_filename)
plt.close()
