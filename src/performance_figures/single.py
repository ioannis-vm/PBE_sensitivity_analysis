# %% Imports #

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--DL_summary_path')
parser.add_argument('--output_path')

args = parser.parse_args()
path = args.DL_summary_path
output_path = args.output_path

# ~~~~ #
# main #
# ~~~~ #

DV_cost = np.genfromtxt(
    path, skip_header=1, delimiter=',')[:, 4]


plt.figure()
sns.ecdfplot(DV_cost)
plt.grid()
plt.xlabel('Cost c')
plt.ylabel('probability (cost < c)')
plt.title('CDF of total repair cost')
plt.savefig(output_path)
plt.close()
