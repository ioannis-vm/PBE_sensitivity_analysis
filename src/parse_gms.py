"""
Process PEER ground motion records
"""

# ~~~~~~~ #
# Imports #
# ~~~~~~~ #

import sys
sys.path.append("src")

import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from ground_motion_utils import import_PEER
from ground_motion_utils import response_spectrum
from util import read_study_param


# ~~~~~~~~~~~~~~~~~~ #
# record information #
# ~~~~~~~~~~~~~~~~~~ #

archetypes_all = read_study_param('data/archetype_codes_response').split()
# extract unique cases
archetypes = []
for arch in archetypes_all:
    if arch not in archetypes:
        archetypes.append(arch)
m = int(read_study_param('data/study_vars/m'))


arch = archetypes[0]
input_dir = f'data/{arch}/ground_motions'
df = pd.read_csv(f'analysis/{arch}/site_hazard/gm_selection.csv', header=[0, 1], index_col=0)
flatfile = "data/peer_nga_west_2/Updated_NGA_West2_Flatfile_RotD50_d050_public_version.csv"
df_flatfile = pd.read_csv(flatfile, index_col=1)
periods = list(df_flatfile.loc[1, "T0.010S":"T10.000S"].index)
periods = [float(p.replace('T', '').replace('S', '')) for p in periods]


all_ground_motions = []
all_spectra = []
all_durations = []
for hz in range(m):
    durations = []
    ground_motions = []
    spectra = []
    for r in range(len(df)):

        # read data
        s = df.loc[r, (f'HzLVL_{hz+1}', 'Scaling')]
        rsn = df.loc[r, (f'HzLVL_{hz+1}', 'RSN')]
        flnmX1 = f'RSN{rsn}_' + df.loc[r, (f'HzLVL_{hz+1}', 'File Name (Horizontal 1)')].replace('\\', '_')
        flnmX2 = f'RSN{rsn}_' + df.loc[r, (f'HzLVL_{hz+1}', 'File Name (Horizontal 2)')].replace('\\', '_')
        flnmV = f'RSN{rsn}_' + df.loc[r, (f'HzLVL_{hz+1}', 'File Name (Vertical)')].replace('\\', '_')
        ag_x = import_PEER(input_dir, flnmX1)
        ag_y = import_PEER(input_dir, flnmX2)
        if flnmV:
            ag_z = import_PEER(input_dir, flnmV)
        else:
            ag_z = np.column_stack((
                np.linspace(0.00, 60.00, len(ag_x[:, 0])),
                np.full(len(ag_x[:, 0]), 0.00)
            ))
        ag_x[:, 1] = ag_x[:, 1] * s
        ag_y[:, 1] = ag_y[:, 1] * s
        ag_z[:, 1] = ag_z[:, 1] * s

        #
        # refine so that dt = 0.005
        #

        # get record dt
        dt_x = ag_x[1, 0]-ag_x[0, 0]
        dt_y = ag_y[1, 0]-ag_y[0, 0]
        if flnmV:
            dt_z = ag_z[1, 0]-ag_z[0, 0]
        else:
            dt_z = dt_x
        assert dt_x == dt_y
        assert dt_z == dt_y

        # get the maximum time in s
        tmax = min(ag_x[-1, 0], ag_y[-1, 0], ag_z[-1, 0])
        # determine the new number of time-value pairs required
        n_new = int(tmax/0.005)
        t_new = np.linspace(0.0, tmax, n_new)

        # interpolate old time-history
        ifun = interp1d(ag_x[:, 0], ag_x[:, 1], kind='linear')
        ag_x = ifun(t_new)
        ifun = interp1d(ag_y[:, 0], ag_y[:, 1], kind='linear')
        ag_y = ifun(t_new)
        ifun = interp1d(ag_z[:, 0], ag_z[:, 1], kind='linear')
        ag_z = ifun(t_new)

        # store the records
        ground_motions.append([ag_x, ag_y, ag_z])
        durations.append(tmax)

        # retrieve response spectrum
        spectrum = df_flatfile.loc[rsn, "T0.010S":"T10.000S"].to_numpy()
        spectra.append(np.column_stack((periods, spectrum)))
    all_durations.append(durations)
    all_spectra.append(spectra)
    all_ground_motions.append(ground_motions)

# Export to corresponding directory
for hz, gms in enumerate(all_ground_motions):
    output_dir = f'analysis/{arch}/hazard_level_{hz+1}/ground_motions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, gm in enumerate(gms):
        np.savetxt(f"{output_dir}/{i+1}x.txt", ground_motions[i][0])
        np.savetxt(f"{output_dir}/{i+1}y.txt", ground_motions[i][1])
        np.savetxt(f"{output_dir}/{i+1}z.txt", ground_motions[i][2])
for hz, spcs in enumerate(all_spectra):
    output_dir = f'analysis/{arch}/hazard_level_{hz+1}/ground_motions'
    for i, spc in enumerate(spcs):
        np.savetxt(f'{output_dir}/{i+1}RS.txt', spc)
for hz, durs in enumerate(all_durations):
    output_dir = f'analysis/{arch}/hazard_level_{hz+1}/ground_motions'
    for i, dur in enumerate(durs):
        np.savetxt(f'{output_dir}/{i+1}D.txt', np.array((dur,)))
