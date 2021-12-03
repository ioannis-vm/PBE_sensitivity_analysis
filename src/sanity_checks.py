import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from src.ground_motion_utils import import_PEER
from src.ground_motion_utils import response_spectrum

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Make sure no analysis failed #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

hz_paths = ['analysis/hazard_level_'+str(i) for i in range(1, 9)]
gm_paths = ['/ground_motions/parsed/'+str(i)+'x.txt' for i in range(1, 15)]
re_paths = ['/response/gm'+str(i)+'/time.csv' for i in range(1, 15)]


d = {'analysis': [], 'gm_dur': [], 'out_dur': []}
for i, hz in enumerate(hz_paths):
    for j, gm in enumerate(gm_paths):
        full_path = hz+gm
        full_res_path = hz+re_paths[j]
        count = len(open(full_path).readlines())
        duration = count * .005
        res_dur = np.genfromtxt(full_res_path)[-1]
        d['analysis'].append(str(i+1)+'-'+str(j+1))
        d['gm_dur'].append(duration)
        d['out_dur'].append(res_dur)

df = pd.DataFrame.from_dict(d)

# print(df.to_string())

for i in range(len(df)):
    diff = df.loc[i].gm_dur - df.loc[i].out_dur
    if np.abs(diff) > 1:
        print('%i %.0f' % (i,diff))





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Compare MCE ground motions to MCE design spectrum that was used #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Extract record information from the csv #

input_dir = 'analysis/hazard_level_7/ground_motions/peer_raw'

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


# load the files #

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


# calculate response spectra

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
    rotd100 = np.max(sa_mat, axis=1)
    response_spectra.append(np.column_stack((periods, rotd100)))


# Plot response spectra

labels = []
for i in range(num_records):
    labels.append(record_metadata[i, 2].replace('"', '') +
                  "\n" + record_metadata[i, 3].replace('"', ''))


labels = []
for i in range(num_records):
    labels.append(record_metadata[i, 2].replace('"', '') +
                  "\n" + record_metadata[i, 3].replace('"', ''))

def k(T):
    if T <= 0.5:
        res = 1.0
    elif T >= 2.5:
        res = 2.0
    else:
        x = np.array([0.5, 2.5])
        y = np.array([1., 2.])
        f = interp1d(x, y)
        res = f(np.array([T]))[0]
    return res

def Tmax(ct, exponent, height, Sd1):
    def cu(Sd1):
        if Sd1 <= 0.1:
            cu = 1.7
        elif Sd1 >= 0.4:
            cu = 1.4
        else:
            x = np.array([0.1, 0.15, 0.2, 0.3, 0.4])
            y = np.array([1.7, 1.6, 1.5, 1.4, 1.4])
            f = interp1d(x, y)
            cu = f(np.array([Sd1]))[0]
        return cu

    Ta = ct * height**exponent
    return cu(Sd1) * Ta

def get_floor_displacements(building, fxx, direction):

    parent_nodes = building.list_of_parent_nodes()

    if direction == 0:
        for i, node in enumerate(parent_nodes):
            node.load = np.array([fxx[i]*1e3, 0.00, 0.00, 0.00, 0.00, 0.00])
    elif direction == 1:
        for i, node in enumerate(parent_nodes):
            node.load = np.array([0.00, fxx[i]*1e3, 0.00, 0.00, 0.00, 0.00])
    else:
        raise ValueError('Invalid Direction')

    linear_gravity_analysis = solver.LinearGravityAnalysis(b)
    linear_gravity_analysis.run()

    u1_el = linear_gravity_analysis.node_displacements[
        parent_nodes[0].uniq_id][0][direction]
    u2_el = linear_gravity_analysis.node_displacements[
        parent_nodes[1].uniq_id][0][direction]
    u3_el = linear_gravity_analysis.node_displacements[
        parent_nodes[2].uniq_id][0][direction]

    return np.array([u1_el, u2_el, u3_el])

def cs(T, Sds, Sd1, R, Ie):
    Tshort = Sd1/Sds
    if T < Tshort:
        res = Sds / R * Ie
    else:
        res = Sd1 / R * Ie / T
    return res

Cd = 5.5
R = 1.0
Ie = 1.0

Sds = 1.452
Sd1 = 2/3.
Tshort = Sd1/Sds

periodsCode = np.logspace(-2, 1, num=100)
accels_Code = np.array([cs(periodsCode[i], Sds, Sd1, R, Ie) for i in range(len(periodsCode))])





    
plt.figure(figsize=(8, 8))
plt.plot(target_spectrum[:, 0],
         target_spectrum[:, 1]*1.3,
         'k--', linewidth=2, label='Target RotD50 * 1.3')
plt.plot(periodsCode,
         accels_Code*3/2,
         'k--', linewidth=2, label='MCE spectrum')
for i, rs in enumerate(response_spectra):
    plt.plot(rs[:, 0], rs[:, 1], linewidth=0.75)
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
plt.legend()
plt.xlabel('Period T [s]')
plt.ylabel('PSA [g]')
plt.xlim((1e-2, 3))
plt.show()
# plt.savefig(plot_dir + '/RS.pdf')
plt.close()
