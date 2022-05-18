import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hz_num = 14  # not zero indexed
gm_num = 1  # not zero indexed
response_type = 'ID-1-1'


gm_dir = f'analysis/office3/hazard_level_{hz_num}/ground_motions/parsed/{gm_num}x.txt'
response_dir = f'analysis/office3/hazard_level_{hz_num}/response/gm{gm_num}'

time_vec = np.genfromtxt(f'{response_dir}/time.csv')
resp_vec = np.genfromtxt(f'{response_dir}/{response_type}.csv')
uddot_g = np.genfromtxt(gm_dir)
gm_time = np.linspace(0.00, 0.005*len(uddot_g), len(uddot_g))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(gm_time, uddot_g)
ax2.plot(time_vec, resp_vec)
plt.show()



# ## Checking aggregated response results

# resp1_path = f'analysis/office3/hazard_level_15/response_summary/response.csv'
# resp1 = pd.read_csv(resp1_path, header=0, index_col=0)
# resp1.drop('units', inplace=True)
# resp1 = resp1.astype(float)
# resp2_path = f'analysis/office3/hazard_level_14/response_summary/response.csv'
# resp2 = pd.read_csv(resp2_path, header=0, index_col=0)
# resp2.drop('units', inplace=True)
# resp2 = resp2.astype(float)

# resp1.std(axis=1) / resp1.mean(axis=1)
# resp2.std(axis=1) / resp2.mean(axis=1)

# pd.concat((resp2.mean(axis=0), resp1.mean(axis=0)), axis=1)

# pd.concat((resp2.std(axis=0), resp1.std(axis=0)), axis=1)

# resp2.std(axis=0) / resp1.std(axis=0)
