# %% Imports #

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--DL_summary_path1')
parser.add_argument('--DL_summary_path2')
parser.add_argument('--label_1')
parser.add_argument('--label_2')
parser.add_argument('--output_path')

args = parser.parse_args()
DL_summary_path1 = args.DL_summary_path1
DL_summary_path2 = args.DL_summary_path2
label_1 = args.label_1
label_2 = args.label_2
output_path = args.output_path

# ~~~~ #
# main #
# ~~~~ #

DV_cost1 = np.genfromtxt(
    DL_summary_path1, skip_header=1, delimiter=',')[:, 4]
DV_cost2 = np.genfromtxt(
    DL_summary_path2, skip_header=1, delimiter=',')[:, 4]


plt.figure()
sns.ecdfplot(DV_cost1, label=label_1)
sns.ecdfplot(DV_cost2, label=label_2)
plt.grid()
plt.xlabel('Cost c')
plt.ylabel('probability (cost < c)')
plt.title('CDF of total repair cost')
plt.legend()
plt.savefig(output_path)
plt.close()
