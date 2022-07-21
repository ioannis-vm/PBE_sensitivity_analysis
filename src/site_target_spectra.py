"""
Generation of conditional mean spectra using OpenSHA PSHA output.
"""

# Imports

import sys
sys.path.append("src")

import os
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from util import read_study_param


archetypes_all = read_study_param('data/archetype_codes_response').split()
m = int(read_study_param('data/study_vars/m'))

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

    spec_data_path = f'analysis/{arch}/site_hazard'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Target Spectra for Ground Motion Selection #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # Uniform Hazard Spectra
    uhss = [np.genfromtxt(f'analysis/{arch}/site_hazard/UHS_{i+1}.csv', skip_header=1, delimiter=',')
            for i in range(m)]

    # mean spectra and standard deviations (from deaggregation)
    gmms = [np.genfromtxt(f'analysis/{arch}/site_hazard/gmm_{i+1}.txt', delimiter=',')
                 for i in range(m)]

    spec_periods = gmms[0][:, 0]
    means = [gmm[:, 1] for gmm in gmms]
    stdevs = [gmm[:, 2] for gmm in gmms]

    # correlation model
    def correl(t1, t2):
        """
        Baker JW, Jayaram N. Correlation of Spectral Acceleration Values
        from NGA Ground Motion Models. Earthquake Spectra.
        2008;24(1):299-317. doi:10.1193/1.2857544
        """
        assert 0.01 <= t1 <= 10
        assert 0.01 <= t2 <= 10
        tmin = min(t1, t2)
        tmax = max(t1, t2)
        # compute c_1
        c_1 = 1. - np.cos(np.pi/2. - 0.366 * np.log(tmax/(max(tmin, 0.109))))
        # compute c_2
        if tmax < 0.2:
            c_2 = 1. - 0.105 * (1. - 1. / (1. + np.exp(100. * tmax - 5.))) \
                * ((tmax - tmin)/(tmax-0.0099))
        else:
            c_2 = 0.00
        # compute c_3
        if tmax < 0.109:
            c_3 = c_2
        else:
            c_3 = c_1
        # compute c_4
        c_4 = c_1 + 0.5 * (np.sqrt(c_3) - c_3) * \
            (1. + np.cos(np.pi * tmin / 0.109))
        # return correlation
        cor = 0.00
        if tmax < 0.109:
            cor = c_2
        elif tmin > 0.109:
            cor = c_1
        elif tmax < 0.20:
            cor = min(c_2, c_4)
        else:
            cor = c_4
        return cor

    # # ~ compare with fig. 11
    # import matplotlib.pyplot as plt
    # num = 200
    # x = np.logspace(-2.0, 1.0, num)
    # y = np.logspace(-2.0, 1.0, num)
    # X, Y = np.meshgrid(x, y)
    # Z = np.full((num, num), 0.00)
    # for i in range(num):
    #     for j in range(num):
    #         Z[i, j] = correl(X[i, j], Y[i, j])
    # fig = plt.subplots()
    # CS = plt.contour(X, Y, Z)
    # plt.clabel(CS, inline=True, fontsize=10)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()
    # plt.close()


    cmss = np.full((len(uhss[0][:, 0]) + 1, m), 0.00)
    for i in range(m):
        ts = uhss[i][:, 0]
        ts_expanded = ts.copy()
        idx = np.argwhere(ts_expanded > Tbar)[0, 0]
        ts_expanded = np.concatenate(
            (ts_expanded[0:idx], np.array((Tbar,)),
             ts_expanded[idx::]),
            axis=0)
        uhs = uhss[i][:, 1]
        fuhs = interp1d(
            ts, uhs, kind='cubic')
        fm = interp1d(
            spec_periods, means[i],
            kind='cubic')
        fs = interp1d(
            spec_periods, stdevs[i],
            kind='cubic')
        correlationsTmin = np.full(len(ts_expanded), 0.00)
        correlations = np.full(len(ts_expanded), 0.00)
        for j, t in enumerate(ts_expanded.tolist()):
            correlations[j] = correl(t, Tbar)
        e = np.log(fuhs(Tbar)/fm(Tbar))/fs(Tbar)
        # print(f'%.2f %.2f' % (e, eps[i]))
        cmss[:, i] = fm(ts_expanded) * np.exp(e * correlations * fs(ts_expanded))
        # print(uhs[idx_tmax]/cmss[idx_tmax, i])

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.grid(which='Major')
    # plt.grid(which='Minor')
    # for i, opsh in enumerate(uhss):
    #     plt.plot(opsh[:, 0], opsh[:, 1])
    #     plt.plot(ts_expanded, cmss[:, i])
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Period T [s]')
    # plt.ylabel('PSa [g] (RotD50)')
    # plt.title('Uniform Hazard Spectra')
    # plt.xlim((1e-2, 1e1))
    # plt.legend()
    # plt.show()
    # plt.close()

    # adjust for directivity using the Bayless and Somerville 2013 model.
    bay_coeff = np.array([
        [0.5, 0., 0.],
        [0.75, 0., 0.],
        [1., -0.12, 0.075],
        [1.5, -0.175, 0.09],
        [2., -0.21, 0.095],
        [3., -0.235, 0.099],
        [4., -0.255, 0.103],
        [5., -0.275, 0.108],
        [7.5, -0.29, 0.112],
        [10., -0.3, 0.115]
    ])
    fgeom = np.log(np.array([37.76, 22.27, 16.40, 12.80, 10.52, 8.90, 7.97, 7.23, 6.67, 6.22, 5.90, 5.57, 5.38, 5.18, 4.97, 4.86]))
    fd = bay_coeff[:, 1] + bay_coeff[:, 2].reshape((-1, 1)).T * fgeom.reshape((-1, 1))
    f_fd = []
    for i in range(m):
        f_fd.append(interp1d(
            bay_coeff[:, 0], fd[i, :],
            kind='linear', fill_value=0.00, bounds_error=False))

    cms_drctv = cmss.copy()



    for i in range(m):
        cms_drctv[:, i] = cms_drctv[:, i] * np.exp(f_fd[i](ts_expanded))


    # # Composite Spectra
    # import matplotlib.pyplot as plt
    # plt.figure()
    # for i, opsh in enumerate(uhss):
    #     plt.plot(ts_expanded, cms_drctv[:, i], linestyle='dotted')
    # plt.gca().set_prop_cycle(None)
    # for i, opsh in enumerate(uhss):
    #     plt.plot(ts_expanded, cms_drctv[:, i])
    # for i, opsh in enumerate(uhss):
    #     plt.plot(ts_expanded, cmss[:, i])
    # plt.axvline(Tbar, color='tab:grey', linestyle='dashed')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Period T [s]')
    # plt.ylabel('PSa [g] (RotD50)')
    # plt.xlim((1e-2, 1e1))
    # # plt.legend()
    # plt.show()
    # # plt.savefig(save_path)
    # # tikzplotlib.save('target_spectra.tex')
    # plt.close()


    # Store target spectra in the PEER-compatible input format
    for i in range(m):
        np.savetxt(
            f'{spec_data_path}/spectrum_'+str(i+1)+'.csv',
            np.column_stack((ts_expanded, cms_drctv[:, i])),
            header='Hazard Level '+str(i+1)+',\r\n,\r\nT (s),Sa (g)',
            delimiter=',', comments='', fmt='%.5f', newline='\r\n')
