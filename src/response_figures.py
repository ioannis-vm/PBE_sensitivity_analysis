# %% Imports #

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--figure_type')
parser.add_argument('--direction')
parser.add_argument('--input_dir')
parser.add_argument('--output_dir')
parser.add_argument('--num_levels')

args = parser.parse_args()
fig_type = args.figure_type
direction = args.direction
input_dir = args.input_dir
output_dir = args.output_dir
num_levels = int(args.num_levels)

# ~~~~ #
# main #
# ~~~~ #

# sns.set_style('whitegrid')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

filetag = dict(PID='ID', PFA='FA', PFA_norm = 'FA', PFV='FV')

if fig_type not in filetag.keys():
    raise ValueError('Invalid fig_type')

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)


response_dirs = [f'{i}/gm{x}' for i, x in zip([input_dir]*14, range(1, 14+1))]
num_earthquakes = len(response_dirs)

# initialize numpy array
response_mat = np.full((num_levels+1, num_earthquakes), 0.00)

for level in range(num_levels+1):
    if fig_type in ['PID'] and level == 0:
        continue
    for j, response_dir in enumerate(response_dirs):
        response_file = response_dir + '/' + filetag[fig_type] + '-' +\
            str(level) + '-' + direction + '.csv'
        contents = np.genfromtxt(response_file)
        peak = np.max(np.abs(contents))
        response_mat[level, j] = peak

# add zeros for the ground level
medians = np.median(response_mat, axis=1)
q84 = np.quantile(response_mat, 0.84, axis=1)
q16 = np.quantile(response_mat, 0.16, axis=1)
y_axis = np.arange(0, num_levels+1)

if fig_type == 'PFA_norm':
    medians = np.median(response_mat/response_mat[0,:], axis=1)
    q84 = np.quantile(response_mat/response_mat[0,:], 0.84, axis=1)
    q16 = np.quantile(response_mat/response_mat[0,:], 0.16, axis=1)
    x_lab = 'PFA/PGA'
else:
    x_lab = fig_type

plt.figure(figsize=(6, 6))
plt.plot(medians, y_axis, linewidth=2, color='black')
plt.scatter(medians, y_axis, s=80, facecolors='none', edgecolors='black',
            label='median')
plt.plot(q84, y_axis, color='black', linestyle='dashed',
         label='16th and 84th quantiles')
plt.plot(q16, y_axis, color='black', linestyle='dashed')
if fig_type == 'PID':
    plt.xlim((-0.005, 0.05))
elif fig_type == 'PFV':
    plt.xlim((-50, 100.))
elif fig_type == 'PFA':
    plt.xlim((-0.05, 2.00))
else:
    plt.xlim((-0.05, 2.50))
plt.xlabel(x_lab)
plt.ylabel('Level')
plt.yticks(np.arange(0, num_levels+1))
plt.legend()
filename = output_dir + '/' + fig_type + '-' + direction + '.pdf'
plt.savefig(filename)
# plt.show()
plt.close()
