"""
Ground motion selection, using ground motions from the PEER NGA West 2
Database.
"""

import sys
sys.path.append("src")

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from util import read_study_param

# ~~~~~~~~~~~~~~~~~ #
# search parameters #
# ~~~~~~~~~~~~~~~~~ #

# magnitude
min_Mw = 6.0
max_Mw = 8.0

# distance
min_Dist = 0.00
max_Dist = 50.00

# vs30
min_vs30 = 360.00
max_vs30 = 1030.00

# scaling factor
min_scaling = 0.25
max_scaling = 4.50

# number of records
num_records = 14


archetypes_all = read_study_param('data/archetype_codes_response').split()
m = int(read_study_param('data/study_vars/m'))


# read the csv file
flatfile = "data/peer_nga_west_2/Updated_NGA_West2_Flatfile_RotD50_d050_public_version.csv"
df = pd.read_csv(flatfile, index_col=1)
# replace na flags with actual nas
df.replace(-999, np.NaN, inplace=True)
# some rows contain nan values for the spectra.
# we exclude those records.
df = df[df["T0.010S"].notna()]

# generate a matrix of all available spectra
# Note: The PEER flatfile contains spectral values for periods higher
# than 10s, but we don't use those since our target spectra only go up
# to 10s.
periods = list(df.loc[1, "T0.010S":"T10.000S"].index)
periods = [float(p.replace('T', '').replace('S', '')) for p in periods]

# filter records based on attributes
df = df[df['Earthquake Magnitude'] < max_Mw]
df = df[df['Earthquake Magnitude'] > min_Mw]
df = df[df['EpiD (km)'] < max_Dist]
df = df[df['EpiD (km)'] > min_Dist]
df = df[df['Vs30 (m/s) selected for analysis'] < max_vs30]
df = df[df['Vs30 (m/s) selected for analysis'] > min_vs30]




archetypes_all = read_study_param('data/archetype_codes_response').split()

# extract unique cases
archetypes = []
for arch in archetypes_all:
    if arch not in archetypes:
        archetypes.append(arch)

for arch in archetypes:

    # read target spectra
    targets = pd.DataFrame(np.zeros((m, len(periods))), columns=periods)
    for i in range(m):
        target_spectrum_data = pd.read_csv(f"analysis/{arch}/site_hazard/spectrum_{i+1}.csv", skiprows=2)
        interp_func = interp1d(
            target_spectrum_data['T (s)'],
            target_spectrum_data['Sa (g)'])
        target = interp_func(periods)
        targets.loc[i, :] = target
    del(i, target)

    def get_records(filtered_record_df, target_sa, num):
        spectra = filtered_record_df.loc[:, "T0.010S":"T10.000S"].to_numpy()
        rsns = filtered_record_df.index

        n_spec = len(spectra)
        scaling_factors = np.zeros(n_spec)
        similarity_measure = np.zeros(n_spec)
        for r in range(n_spec):
            s = spectra[r, :]
            t = target_sa
            c = (t.T @ s) / (s.T @ s)
            scaling_factors[r] = c
            similarity_measure[r] = np.linalg.norm(t - c * s)

        res_df = pd.DataFrame({
            'RSN': rsns,
            'Scaling': scaling_factors,
            'MSE': similarity_measure
        })
        res_df.set_index('RSN', inplace=True)

        # scaling factor filter
        res_df = res_df[res_df['Scaling'] < max_scaling]
        res_df = res_df[res_df['Scaling'] > min_scaling]
        # order by lowest MSE
        res_df.sort_values(by='MSE', inplace=True)

        return res_df.iloc[0:num, :]

    target = targets.iloc[-1, :].to_numpy()
    resA = get_records(df, target, num_records)

    target = targets.iloc[-2, :].to_numpy()
    resB = get_records(df.loc[resA.index, :], target, num_records)

    all_suite_results = []
    for i in range(m-1, -1, -1):
        target = targets.iloc[i, :].to_numpy()
        if 15 > float(i) > m/2:
            # use the same RSNs as the strongest hz lvl here
            # (this improves variance consistency)
            res = get_records(df.loc[
                res.index, :], target, num_records)
        else:
            # pick any RSN now
            res = get_records(df, target, num_records)
        res['File Name (Horizontal 1)'] = df.loc[
            res.index, 'File Name (Horizontal 1)']
        res['File Name (Horizontal 2)'] = df.loc[
            res.index, 'File Name (Horizontal 2)']
        res['File Name (Vertical)'] = df.loc[
            res.index, 'File Name (Vertical)']
        new_cols = []
        for col in res.columns:
            new_cols.append((f'HzLVL_{i+1}', col))
        res.columns = new_cols
        all_suite_results.append(res)

    all_suite_results.reverse()

    for i in range(m):
        res = all_suite_results[i]
        res[(f'HzLVL_{i+1}', 'RSN')] = res.index
        res.index = range(num_records)

    all_suite_results = pd.concat(all_suite_results, axis=1)
    all_suite_results.columns = pd.MultiIndex.from_tuples(
        all_suite_results.columns)

    # def plot_suite(filtered_record_df, rsns, scaling, target_sa):
    #     result_spectra = filtered_record_df.loc[
    #         rsns, "T0.010S":"T10.000S"].to_numpy() * \
    #         np.repeat(scaling.reshape(
    #             (-1, 1)), 105, axis=1)
    #     res_mean = result_spectra.mean(axis=0)
    #     res_stdv = result_spectra.std(axis=0)
    #     import matplotlib.pyplot as plt
    #     plt.plot(periods, res_mean, 'k')
    #     plt.plot(periods, res_mean+res_stdv, 'k')
    #     plt.plot(periods, res_mean-res_stdv, 'k')
    #     plt.plot(periods, target_sa, 'r--')
    #     plt.show()

    # for i in range(m):
    #     rsns = all_suite_results.loc[:, (f'HzLVL_{i+1}', 'RSN')].to_numpy()
    #     scaling = all_suite_results.loc[:, (f'HzLVL_{i+1}', 'Scaling')].to_numpy()
    #     target = targets.iloc[i, :].to_numpy()
    #     plot_suite(df, rsns, scaling, target)

    # generate PEER website RSN input
    needed_rsns = set()
    for i in range(m):
        rsns = all_suite_results.loc[:, (f'HzLVL_{i+1}', 'RSN')]
        for rsn in rsns:
            needed_rsns.add(rsn)

    with open(f'analysis/{arch}/site_hazard/required_records.txt', 'w') as f:
        f.write(str(needed_rsns))

    all_suite_results.to_csv(f'analysis/{arch}/site_hazard/gm_selection.csv')
