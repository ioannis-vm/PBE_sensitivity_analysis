"""
Process PEER ground motion records
"""

import sys
sys.path.append("FEMA_P-58_Python_Implementation/Ground_Motions")

# ~~~~~~~ #
# Imports #
# ~~~~~~~ #

import numpy as np
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from ground_motion_utils import import_PEER
from ground_motion_utils import response_spectrum
from ground_motion_utils import code_spectrum


# ~~~~~~~~~~ #
# Parameters #
# ~~~~~~~~~~ #

input_folder_name = "ground_motions/test_case/peer_raw"
output_folder_name = 'ground_motions/test_case/parsed'
plot_folder_name = 'ground_motions/test_case/plots'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Extract record information from the csv #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

idx = []
count = False
with open(input_folder_name+'/_SearchResults.csv', 'r') as f:
    for i, line in enumerate(f):
        line = line.strip()  # remove whitespace, newlines etc
        if line == '-- Summary of Metadata of Selected Records --':
            idx.append(i)  # store the line where info starts
            count = True  # start counting lines
        if count:  # if we are after the line where info starts
            # must find where info ends.
            # there is an empty line there
            if line == '':
                idx.append(i)
                break
idx.append(len(open(
    input_folder_name+'/_SearchResults.csv').readlines()
    )
)

record_metadata = np.genfromtxt(
    input_folder_name+'/_SearchResults.csv', delimiter=',',
    dtype=str, comments='###', skip_header=idx[0]+2,
    skip_footer=idx[2]-idx[1]-3,
    usecols=(0, 4, 9, 11, 19, 20, 21),
    deletechars=" !#$%&'()*+, -./:;<=>?@[\\]^{|}~", autostrip=True
)


# ~~~~~~~~~~~~~~ #
# load the files #
# ~~~~~~~~~~~~~~ #

print('Reading ground motion files')

ground_motions = []
num_records = len(record_metadata)

for i in trange(num_records):

    # read data
    s = float(record_metadata[i, 1])  # scaling factor
    ag_x = import_PEER(input_folder_name, record_metadata[i, 4])
    ag_y = import_PEER(input_folder_name, record_metadata[i, 5])
    ag_z = import_PEER(input_folder_name, record_metadata[i, 6])
    ag_x[:, 1] = ag_x[:, 1] * s
    ag_y[:, 1] = ag_y[:, 1] * s
    ag_z[:, 1] = ag_z[:, 1] * s

    #
    # refine so that dt = 0.005
    #

    # get record dt
    dt_x = ag_x[1, 0]-ag_x[0, 0]
    dt_y = ag_y[1, 0]-ag_y[0, 0]
    dt_z = ag_z[1, 0]-ag_z[0, 0]
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

print('Calculating response spectra')

response_spectra = []

for i in trange(num_records):

    rs_x = response_spectrum(ground_motions[i][0], 0.005, 0.05, T_max=2.0)
    rs_y = response_spectrum(ground_motions[i][1], 0.005, 0.05, T_max=2.0)
    rs_geomean = np.full(np.shape(rs_x), 0.00)
    rs_geomean[:, 0] = rs_x[:, 0]
    rs_geomean[:, 1] = np.sqrt(rs_x[:, 1]**2 + rs_y[:, 1]**2)

    response_spectra.append(rs_geomean)


# ~~~~~~~~~~~~~~ #
# generate plots #
# ~~~~~~~~~~~~~~ #

print('Generating plots')

#
# Plot 1: time-acceleration combined plot
#

direction_label = ['x', 'y', 'z']

labels = []
for i in range(num_records):
    labels.append(record_metadata[i, 2].replace('"', '') +
                  "\n" + record_metadata[i, 3].replace('"', ''))

for k in range(3):
    axs = []
    # plot the first record
    num_pts = len(ground_motions[0][k])
    t_vals = np.linspace(0.00, num_pts*0.005, num_pts)
    plt.figure(figsize=(10, num_records))
    axs.append(plt.subplot(num_records, 1, 1))
    plt.plot(t_vals, ground_motions[0][k], 'k', label=labels[0])
    plt.legend(loc='upper right')
    # and then the rest
    for i in trange(1, num_records):
        num_pts = len(ground_motions[i][k])
        t_vals = np.linspace(0.00, num_pts*0.005, num_pts)
        axs.append(plt.subplot(num_records, 1, i+1, sharex=axs[0], sharey=axs[0]))
        plt.plot(t_vals, ground_motions[i][k], 'k', label=labels[i])
        plt.legend(loc='upper right')
    # hide x-tickmarks except for the last plot
    for i in range(num_records-1):
        plt.setp(axs[i].get_xticklabels(), visible=False)
    # set axis limits
    plt.xlim(0.00, 50.00)

    if not os.path.exists(plot_folder_name):
        os.mkdir(plot_folder_name)
    plt.savefig(plot_folder_name + '/time_history_' +
                direction_label[k] + '.pdf')
    plt.close()

#
# Plot 2: Response spectrum
#

# parameters
Ss = 1.33 * 1.50
S1 = 0.75 * 1.50
Tl = 8.00

rs_code = code_spectrum(np.linspace(0.00, 2.00, 500),
                        Ss, S1, Tl)

plt.figure(figsize=(10, 10))
plt.plot(rs_code[:, 0], rs_code[:, 1], 'k:', linewidth=2, label='Code')
for i, rs in enumerate(response_spectra):
    plt.plot(rs[:, 0], rs[:, 1], linewidth=0.75, label=labels[i])
plt.legend(loc='upper right')
plt.xlabel('Period T [s]')
plt.ylabel('PSA [g]')
plt.savefig(plot_folder_name + '/RS.pdf')
plt.close()

# ~~~~~~~~~~~~~~~~~~~~ #
# save results on disk #
# ~~~~~~~~~~~~~~~~~~~~ #

print('Storing results on disk')

if not os.path.exists(output_folder_name):
    os.mkdir(output_folder_name)
    # Export to corresponding directory
for i in trange(num_records):
    np.savetxt(output_folder_name+'/'+str(i+1)+'x.txt', ground_motions[i][0])
    np.savetxt(output_folder_name+'/'+str(i+1)+'y.txt', ground_motions[i][1])
    np.savetxt(output_folder_name+'/'+str(i+1)+'z.txt', ground_motions[i][2])


print('Done.')
