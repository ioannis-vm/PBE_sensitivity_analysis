# %% Imports #

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ~~~~ #
# main #
# ~~~~ #
DV_cost_A = []
DV_cost_C = []
for i in range(8):
    DV_cost_A.append(
        np.genfromtxt(
            'analysis/hazard_level_'+str(i+1)+'/performance/A/DL_summary.csv',
            skip_header=1, delimiter=',')[:, 4])
    DV_cost_C.append(
        np.genfromtxt(
            'analysis/hazard_level_'+str(i+1)+'/performance/C/DL_summary.csv',
            skip_header=1, delimiter=',')[:, 4])

plt.figure()
for i in range(8):
    sns.ecdfplot(DV_cost_A[i], color='black', label="A-"+str(i+1))
    sns.ecdfplot(DV_cost_C[i], color='purple',
                 linestyle='dotted', label="C-"+str(i+1))
plt.grid()
plt.xlabel('Cost c')
plt.ylabel('probability (cost < c)')
plt.title('CDF of total repair cost')
plt.legend()
plt.savefig('figures/combined/performance/total_cost_A-C.pdf')
plt.close()
