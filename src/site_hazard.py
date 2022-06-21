"""
Generation of site hazard curves for ground motion selection for a
time-based assessment using OpenSHA PSHA output.

Note

  Assuming Poisson distributed earthquake occurence,


  p_occurence = 1 - exp(-t/T)

  where:
    P_exceedance is the probability of 1 or more occurences,
    t is the period of interest (1 year, 50 years, etc.),
    T is the return period (e.g. 475 years, 2475 years),
    1/T is called the `occurence rate` or `frequency`

"""

# Imports

import sys
sys.path.append("src")

import os
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from util import read_study_param

# ~~~~~~~~~~~~~~~~~~~~ #
# Parse OpenSHA output #
# ~~~~~~~~~~~~~~~~~~~~ #

names = ['0p01', '0p02', '0p03', '0p05', '0p075', '0p1', '0p15', '0p2',
         '0p25', '0p3', '0p4', '0p5', '0p75', '1p0', '1p5', '2p0', '3p0',
         '4p0', '5p0', '7p5', '10p0']
periods = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4,
           0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
hzrd_crv_files = [f'analysis/site_hazard/{name}.txt' for name in names]

accelerations = []
MAPEs = []
MAFEs = []

for filepath in hzrd_crv_files:
    with open(filepath, 'r') as f:
        line = f.read()
    contents = line.split('\n')
    for i, c in enumerate(contents):
        words = c.split(' ')
        if words[0] == 'X,' and words[1] == 'Y' and words[2] == 'Data:':
            i_begin = i + 1
        if words[0] == 'Num':
            num_points = int(words[2])
    i_end = i_begin + num_points
    data = np.genfromtxt(contents[i_begin:i_end])
    accelerations.append(data[:, 0])
    MAPEs.append(data[:, 1])
    # OpenSHA operates in probability space.
    # Probabilities have to be converted to frequencies of exceedance
    MAFEs.append(-np.log(1-data[:, 1]))


# ~~~~~~~~~~~~~~~~~~ #
# Plot hazard curves #
# ~~~~~~~~~~~~~~~~~~ #

# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 10))
# plt.grid(which='Major')
# plt.grid(which='Minor')
# for i in range(len(periods)):
#     plt.plot(accelerations[i], MAFEs[i], '-s',
#              label=f'T = {periods[i]:5.2f} s')
# plt.xscale('log')
# plt.yscale('log')
# plt.axhline(2e-2)
# plt.axhline(4e-4)
# plt.legend()
# plt.xlabel('Earthquake Intensity $e$ [g]')
# plt.ylabel('Mean annual frequency of exceedance $λ$')
# plt.show()
# plt.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Obtain period-specific hazard curve #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

archetypes_all = read_study_param('data/archetype_codes_response').split()

# extract unique cases
archetypes = []
for arch in archetypes_all:
    if arch not in archetypes:
        archetypes.append(arch)

# get the periods from the data directory
study_periods = []
for arch in archetypes:
    study_periods.append(float(read_study_param(f'data/{arch}/period')))

for Tbar, arch in zip(study_periods, archetypes):

    # Interpolate available hazard curves
    all_mafes = np.array(MAFEs)
    target_MAFE = []
    for col in range(np.shape(all_mafes)[1]):
        vec = all_mafes[:, col]
        f = interp1d(
            periods, vec, kind='linear')
        target_MAFE.append(float(f(Tbar)))
    target_acceleration = accelerations[0]  # they are the same for all curves

    # Define interpolation functions for the period-specific hazard curve

    # Interpolate: From intensity e [g] to MAFE λ
    def fHazMAFEtoSa(mafe):
        temp1 = interp1d(np.log(target_MAFE), np.log(target_acceleration),
                        kind='cubic')
        return np.exp(temp1(np.log(mafe)))

    # Interpolate: Inverse (From MAFE λ to intensity e [g])
    def fHazSatoMAFE(f):
        temp2 = interp1d(np.log(target_acceleration), np.log(target_MAFE),
                         kind='cubic')
        return np.exp(temp2(np.log(f)))


    # Specify Intensity range
    # if Tbar <= 1.00:
    #     SaMin = 0.05
    # else:
    #     SaMin = 0.05/Tbar
    SaMax = fHazMAFEtoSa(2e-4)
    SaMin = 0.005  # g

    # # plot target hazard curve
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.grid(which='Major')
    # plt.grid(which='Minor')
    # for i in range(len(periods)):
    #     plt.plot(accelerations[i], MAFEs[i],  '-',
    #              label='T = ' + str(periods[i]) + ' s',
    #              linewidth=1.0)
    # plt.plot(
    #     target_acceleration, target_MAFE, '-s',
    #     label='Target', color='k',
    #     linewidth=3)
    # plt.axvline(SaMin, color='k', linestyle='dashed')
    # plt.axvline(SaMax, color='k', linestyle='dashed')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    # plt.xlabel('Earthquake Intensity $e$ [g]')
    # plt.ylabel('Mean annual frequency of exceedance $λ$')
    # plt.show()
    # plt.close()


    # Split intensity range to m intervals

    m = int(read_study_param('data/study_vars/m'))

    # Determine interval midpoints and endpoints

    e_vec = np.linspace(SaMin, SaMax, m*2+1)
    mafe_vec = fHazSatoMAFE(e_vec)
    mafe_des = 1./475.
    mafe_mce = 1./2475.
    e_des = fHazMAFEtoSa(mafe_des)
    e_mce = fHazMAFEtoSa(mafe_mce)

    # + this makes sure that two of the midpoints
    #   will fall exactly on the design and MCE level
    #   scenarios.

    if mafe_vec[-1] < mafe_des < mafe_vec[0]:
        # identify index closest to design lvl
        dif = np.full(m*2+1, 0.00)
        for i, e in enumerate(e_vec):
            dif[i] = e_des - e
        k = 2 * np.argmin(dif[1::2]**2) + 1
        corr = np.full(len(e_vec), 0.00)
        corr[0:k+1] = np.linspace(0, dif[k], k+1)
        corr[k::] = np.linspace(dif[k], 0, m*2-k+1)
        e_vec = e_vec + corr
        mafe_vec = fHazSatoMAFE(e_vec)

    if mafe_vec[-1] < mafe_mce < mafe_vec[0]:
        # identify index closest to MCE lvl
        dif = np.full(m*2+1, 0.00)
        for i, e in enumerate(e_vec):
            dif[i] = e_mce - e
        k2 = 2 * np.argmin(dif[1::2]**2) + 1
        corr = np.full(len(e_vec), 0.00)
        corr[k+1:k2] = np.linspace(0, dif[k2], k2 - (k + 1))
        corr[k2::] = np.linspace(dif[k2], 0, m*2-k2+1)
        e_vec = e_vec + corr
        mafe_vec = fHazSatoMAFE(e_vec)


    e_Endpoints = e_vec[::2]
    MAFE_Endpoints = mafe_vec[::2]
    e_Midpoints = e_vec[1::2]
    MAFE_Midpoints = mafe_vec[1::2]
    MAPE_Midpoints = 1 - np.exp(-MAFE_Midpoints)
    return_period_midpoints = 1 / MAFE_Midpoints
    delta_e = np.array([e_Endpoints[i+1]-e_Endpoints[i]
                        for i in range(m)])
    delta_lamda = np.array([MAFE_Endpoints[i]-MAFE_Endpoints[i+1]
                            for i in range(m)])

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.grid(which='Major')
    # plt.grid(which='Minor')
    # plt.plot(
    #     target_acceleration, target_MAFE, '-',
    #     label='Hazard Curve', color='black')
    # plt.scatter(e_Endpoints, MAFE_Endpoints,
    #             s=80, facecolors='none', edgecolors='k',
    #             label='Interval Endpoints')
    # plt.scatter(e_Midpoints, MAFE_Midpoints,
    #             s=40, facecolors='k', edgecolors='k',
    #             label='Interval Midpoints')
    # for i, txt in enumerate(range(1, m+1)):
    #     plt.annotate(txt, (e_Midpoints[i], MAFE_Midpoints[i]))
    # plt.axvline(SaMin, color='k', linestyle='dashed',
    #             label='Intensity Range')
    # plt.axvline(SaMax, color='k', linestyle='dashed')
    # plt.axhline(mafe_mce, color='red', label='MCE')
    # plt.axhline(mafe_des, color='blue', label='Design')
    # plt.legend()
    # plt.xlabel('Earthquake Intensity $e$ [g] ( Sa(T* = %.3f s) )' % (Tbar))
    # plt.ylabel('Mean annual frequency of exceedance $λ$')
    # plt.xlim((0.00-0.05, SaMax + 0.05))
    # plt.ylim((1e-4, 1))
    # plt.yscale('log')
    # plt.show()
    # plt.close()


    # store hazard curve interval data
    # (will be used to obtain mean annual rate of exceedance of
    #  decision variables)

    out_path = f'analysis/{arch}/site_hazard'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    interv_df = pd.DataFrame(
        np.column_stack(
            (e_Midpoints, delta_e,
             delta_lamda,
             MAFE_Midpoints,
             MAPE_Midpoints,
             return_period_midpoints)),
        columns=['e', 'de', 'dl', 'freq', 'prob', 'T'],
        index=range(1, m+1))

    interv_df.to_csv(f'{out_path}/Hazard_Curve_Interval_Data.csv')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Obtain Uniform Hazard Spectra for each midpoint #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    uhs_dfs = []

    spectrum_idx = 0
    for spectrum_idx in range(m):
        rs = []
        target_mafe = MAFE_Midpoints[spectrum_idx]
        for curve_idx in range(len(periods)):
            MAFEs[curve_idx][MAFEs[curve_idx] == 0] = 1.0e-14
            log_mafe = np.log(MAFEs[curve_idx])
            log_sa = np.log(accelerations[curve_idx])
            log_target_mafe = np.log(target_mafe)
            f = interp1d(log_mafe, log_sa, kind='linear')
            log_target_sa = float(f(log_target_mafe))
            target_sa = np.exp(log_target_sa)
            rs.append(target_sa)
        uhs = np.column_stack((periods, rs))
        uhs_df = pd.DataFrame(uhs, columns=['T', 'Sa'])
        uhs_df.set_index('T', inplace=True)
        uhs_dfs.append(uhs_df)

    # write UHS data to files
    for i, uhs_df in enumerate(uhs_dfs):
        uhs_df.to_csv(f'{out_path}/UHS_{i+1}.csv')
