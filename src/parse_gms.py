"""
Process PEER ground motion records
"""

# ~~~~~~~ #
# Imports #
# ~~~~~~~ #

import sys
sys.path.append("src")

import numpy as np
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from ground_motion_utils import import_PEER
from ground_motion_utils import response_spectrum
import argparse

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# ~~~~~~~~~~~~~~~~~~~~~ #
# setup argument parser #
# ~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--output_dir')
parser.add_argument('--plot_dir')
args = parser.parse_args()


# ~~~~~~~~~~ #
# Parameters #
# ~~~~~~~~~~ #

input_dir = args.input_dir
output_dir = args.output_dir
plot_dir = args.plot_dir

# # debug
# input_dir = 'analysis/hazard_level_8/ground_motions/peer_raw'
# output_dir = 'analysis/hazard_level_8/ground_motions/parsed'
# plot_dir = 'figures/hazard_level_8/ground_motions'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Extract record information from the csv #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def parse_SearchResults(target_line,
                        columns_of_interest,
                        subsequent_blank_lines):
    idx = []
    count = False
    with open(input_dir+'/_SearchResults.csv', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()  # remove whitespace, newlines etc
            if line == target_line:
                idx.append(i + 2)  # store the line where info starts
                count = True  # start counting lines
            if count:  # if we are after the line where info starts
                # must find where info ends.
                # there is an empty line there
                if line == '':
                    idx.append(i)
                    break
    idx.append(len(open(
        input_dir+'/_SearchResults.csv').readlines()
        )
    )

    record_metadata = np.genfromtxt(
        input_dir+'/_SearchResults.csv', delimiter=',',
        dtype=str, comments='###', skip_header=idx[0],
        skip_footer=idx[2]-idx[1]-subsequent_blank_lines,
        usecols=columns_of_interest,
        deletechars=" !#$%&'()*+, -./:;<=>?@[\\]^{|}~", autostrip=True
    )
    return record_metadata


record_metadata = parse_SearchResults(
    '-- Summary of Metadata of Selected Records --',
    (0, 4, 9, 11, 19, 20, 21), 3)

target_spectrum = parse_SearchResults(
    '-- Scaled Spectra used in Search & Scaling --',
    (0, 1), 1)

target_spectrum = np.asarray(
    target_spectrum, dtype=np.float64, order='C')


# ~~~~~~~~~~~~~~ #
# load the files #
# ~~~~~~~~~~~~~~ #

ground_motions = []
num_records = len(record_metadata)

for i in range(num_records):

    # read data
    s = float(record_metadata[i, 1])  # scaling factor
    ag_x = import_PEER(input_dir, record_metadata[i, 4])
    ag_y = import_PEER(input_dir, record_metadata[i, 5])
    if record_metadata[i, 6].strip() != '-----':
        ag_z = import_PEER(input_dir, record_metadata[i, 6])
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
    if record_metadata[i, 6].strip() != '-----':
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# calculate response spectra #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

response_spectra = []
n_pts = 200
sa_mat = np.full((n_pts, 180), 0.00)

for i in range(num_records):

    rs_x = response_spectrum(ground_motions[i][0], 0.005, 0.05, n_Pts=n_pts)
    rs_y = response_spectrum(ground_motions[i][1], 0.005, 0.05, n_Pts=n_pts)
    periods = rs_x[:, 0]
    x_vec = rs_x[:, 1]
    y_vec = rs_y[:, 1]
    xy = np.column_stack((x_vec, y_vec))
    angles = np.linspace(0.00, 180.00, 180)
    for j, ang in enumerate(np.nditer(angles)):
        rot_mat = np.array(
            [
                [np.cos(ang)],
                [np.sin(ang)]
            ]
        )
        sa_mat[:, j] = np.abs(np.reshape(xy @ rot_mat, (1, -1)))
    rotd50 = np.median(sa_mat, axis=1)
    response_spectra.append(np.column_stack((periods, rotd50)))
    # store response spectra
    np.savetxt(f'{output_dir}/{i+1}RS.txt',
               np.column_stack((periods, rotd50)))



# ~~~~~~~~~~~~~~ #
# generate plots #
# ~~~~~~~~~~~~~~ #

#
# Plot 1: time-acceleration combined plot
#

direction_label = ['x', 'y', 'z']

labels = []
for i in range(num_records):
    labels.append(record_metadata[i, 2].replace('"', '') +
                  "\n" + record_metadata[i, 3].replace('"', ''))

axs = []
# plot the first record
num_pts = len(ground_motions[0][0])
t_vals = np.linspace(0.00, num_pts*0.005, num_pts)
plt.figure(figsize=(10, num_records))
axs.append(plt.subplot(num_records, 1, 1))
plt.plot(t_vals, ground_motions[0][0],
         'red', label=labels[0], linewidth=0.4)
plt.plot(t_vals, ground_motions[0][1],
         'green', linewidth=0.4)
plt.plot(t_vals, ground_motions[0][2],
         'blue', linewidth=0.4)
plt.legend(loc='upper right')
# and then the rest
for i in range(1, num_records):
    num_pts = len(ground_motions[i][0])
    t_vals = np.linspace(0.00, num_pts*0.005, num_pts)
    axs.append(plt.subplot(num_records, 1, i+1, sharex=axs[0], sharey=axs[0]))
    plt.plot(t_vals, ground_motions[i][0],
             'red', label=labels[i], linewidth=0.4)
    plt.plot(t_vals, ground_motions[i][1],
             'green', linewidth=0.4)
    plt.plot(t_vals, ground_motions[i][2],
             'blue', linewidth=0.4)
    plt.legend(loc='upper right')
# hide x-tickmarks except for the last plot
for i in range(num_records-1):
    plt.setp(axs[i].get_xticklabels(), visible=False)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
plt.savefig(plot_dir + '/time_history.pdf')
# plt.show()
plt.close()

#
# Plot 2: Response spectrum
#


plt.figure(figsize=(10, 10))
plt.plot(target_spectrum[:, 0],
         target_spectrum[:, 1],
         'k', linewidth=2, label='Target')
for i, rs in enumerate(response_spectra):
    plt.plot(rs[:, 0], rs[:, 1], linewidth=0.75, label=labels[i])
rs_T = rs[:, 0]
rs_SA = np.column_stack([rs[:, 1] for rs in response_spectra])
rs_mean = np.mean(rs_SA, axis=1)
rs_stdev = np.std(rs_SA, axis=1)
plt.plot(rs_T, rs_mean, linewidth=2, color='red', label='$\mu$')
plt.plot(rs_T, rs_mean+rs_stdev, linewidth=2, color='red',
         linestyle='dashed', label='$\mu\pm\sigma$')
plt.plot(rs_T, rs_mean-rs_stdev, linewidth=2, color='red', linestyle='dashed')
# plt.xscale('log')
# plt.yscale('log')
plt.legend(bbox_to_anchor=(1.05, 1.00), loc='upper left')
plt.xlabel('Period T [s]')
plt.ylabel('PSA [g]')
plt.xscale('log')
plt.yscale('log')
plt.xlim((1e-2, 3))
plt.ylim((1e-2, 6))
plt.tight_layout()
# plt.show()
plt.savefig(plot_dir + '/RS.pdf')
plt.close()

# ~~~~~~~~~~~~~~~~~~~~ #
# save results on disk #
# ~~~~~~~~~~~~~~~~~~~~ #

# Export to corresponding directory
for i in range(num_records):
    np.savetxt(f"{output_dir}/{i+1}x.txt", ground_motions[i][0])
    np.savetxt(f"{output_dir}/{i+1}y.txt", ground_motions[i][1])
    np.savetxt(f"{output_dir}/{i+1}z.txt", ground_motions[i][2])
