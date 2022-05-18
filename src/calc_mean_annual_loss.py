import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import integrate

output_dir = "/home/john_vm/google_drive_encr/UCB/research/projects/299_report/299_report/data/performance_cdfs"
output_dir_annual = "/home/john_vm/google_drive_encr/UCB/research/projects/299_report/299_report/data/"

num_hazard_lvls = 16
case_1 = 'office3'
case_2 = 'healthcare3'
hzrd_calc_of_case = dict(office3='office3',
                         healthcare3='office3')

uncertainty = 'medium'

# NOTE: run twice, changing threshold to 0.4 and 1.0 to generate all necessary data
# for the production plots
threshold = 0.4

hazard_interval_data_path_1 = \
    f'analysis/{hzrd_calc_of_case[case_1]}/site_hazard/Hazard_Curve_Interval_Data.csv'
hazard_interval_data_path_2 = \
    f'analysis/{hzrd_calc_of_case[case_2]}/site_hazard/Hazard_Curve_Interval_Data.csv'
performance_data_path_1 = \
    [f'analysis/{case_1}/hazard_level_{i+1}'
     f'/performance/{uncertainty}/{threshold}/edp/total_cost_realizations.csv'
     for i in range(num_hazard_lvls)]
performance_data_path_2 = \
    [f'analysis/{case_2}/hazard_level_{i+1}'
     f'/performance/{uncertainty}/{threshold}/edp/total_cost_realizations.csv'
     for i in range(num_hazard_lvls)]

interval_data_1 = pd.read_csv(hazard_interval_data_path_1, index_col=0)
interval_data_2 = pd.read_csv(hazard_interval_data_path_2, index_col=0)

ys_1 = []
ys_2 = []
for i in range(num_hazard_lvls):
    df_1 = pd.read_csv(performance_data_path_1[i])
    df_2 = pd.read_csv(performance_data_path_2[i])
    ys_1.append(df_1.loc[:, 'A'])
    ys_2.append(df_2.loc[:, 'A'])


# plt.figure()
# for i in range(num_hazard_lvls):
#     sns.ecdfplot(ys_1[i], color='k')
# plt.show()

y_df_1 = pd.DataFrame(np.column_stack(ys_1), columns=range(1, num_hazard_lvls+1))
y_df_2 = pd.DataFrame(np.column_stack(ys_2), columns=range(1, num_hazard_lvls+1))

def prob_of_non_exceedance(cost, y):
    return ((y < cost).mean(axis=0))


# generate cdf x-y pairs and save for tikz
num_cost_vals = 500
costs1 = np.linspace(0.00, 1.5e7, num=num_cost_vals)
costs2 = np.linspace(0.00, 3.e7, num=num_cost_vals)

prob_1 = []
prob_2 = []
for i in range(num_hazard_lvls):
    probs1 = []
    probs2 = []
    for j in range(num_cost_vals):
        probs1.append(prob_of_non_exceedance(costs1[j], y_df_1.loc[:, i+1].to_numpy()))
        probs2.append(prob_of_non_exceedance(costs2[j], y_df_2.loc[:, i+1].to_numpy()))
    prob_1.append(probs1)
    prob_2.append(probs2)
    
for i in range(num_hazard_lvls):
    np.savetxt(f'{output_dir}/cdf_office3_{threshold}_{i+1}.txt', np.column_stack((costs1[::2]/1.e6, prob_1[i][::2])), delimiter=' ')
    np.savetxt(f'{output_dir}/cdf_healthcare3_{threshold}_{i+1}.txt', np.column_stack((costs2[::2]/1.e6, prob_2[i][::2])), delimiter=' ')



def rate_of_exceedance(cost, y, dl):
    return ((y > cost).mean(axis=0) * dl)

num_cost_vals = 5000
costs1 = np.linspace(0.00, 1.5e7, num=num_cost_vals)
costs2 = np.linspace(0.00, 3.e7, num=num_cost_vals)

rates_1 = np.zeros((num_cost_vals, num_hazard_lvls))
rates_2 = np.zeros((num_cost_vals, num_hazard_lvls))
for i in range(num_cost_vals):
    rates_1[i, :] = rate_of_exceedance(
        costs1[i], y_df_1, interval_data_1['dl'])
    rates_2[i, :] = rate_of_exceedance(
        costs2[i], y_df_2, interval_data_2['dl'])

# # plot the rates
# plt.plot(costs1, rates_1)
# plt.plot(costs2, rates_2)
# plt.show()

mean_annualized_costs_1 = np.zeros(num_hazard_lvls)
mean_annualized_costs_2 = np.zeros(num_hazard_lvls)
for i in range(len(mean_annualized_costs_1)):
    mean_annualized_costs_1[i] = integrate.simpson(rates_1[:, i], costs1)
    mean_annualized_costs_2[i] = integrate.simpson(rates_2[:, i], costs2)

plt.scatter(range(1, 17), mean_annualized_costs_1)
plt.scatter(range(1, 17), mean_annualized_costs_2)
plt.show()


df_out = pd.DataFrame(
    np.column_stack((mean_annualized_costs_1, mean_annualized_costs_2)),
    index=range(1,16+1)
)
df_out.to_csv(f'{output_dir_annual}/annualized.txt', sep=' ', header=False)

# np.savetxt(
#     ,
#     np.row_stack((mean_annualized_costs_1, mean_annualized_costs_2)),
#     delimiter=' ')
