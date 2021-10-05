from scipy import interpolate
import requests
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def retrieve_hazard_curves(long, lat, vs30, cache=False):
    """
    Sends a get request to the USGS hazard API to retrieve hazard data
    long: longitude, lat: latitude, vs30: vs30.
    """
    # Check if the query has been made in the past
    fname = long+'_'+lat+'_'+vs30+'.pcl'
    if os.path.exists('./src/USGS_Data_Cache/' + fname):
        data = pickle.load(open('./src/USGS_Data_Cache/' + fname, "rb"))
    else:
        # Construct URL to send the request
        URL = 'https://earthquake.usgs.gov/nshmp-haz-ws/hazard/E2008/WUS/' + \
            long + '/' + lat + '/any/' + vs30
        # Send the request
        r = requests.get(url=URL)
        # Cache the response
        # Convert data to JSON format
        data = r.json()
        if cache is True:
            # Output response to a file
            pickle.dump(data, open('./src/USGS_Data_Cache/' + fname, "wb"))
    # Initialize empty containers
    descs = []
    accelerations = {}
    MAFEs = {}
    # Store relevant hazard curve
    for curve in data['response']:
        name = curve['metadata']['imt']['value']
        desc = curve['metadata']['imt']['display']
        accelerations[name] = np.array(curve['metadata']['xvalues'])
        MAFEs[name] = np.array(curve['data'][0]['yvalues'])
        # Replace zeros with None
        MAFEs[name][MAFEs[name] == 0.00] = None
        descs.append(desc)
    # Parse descriptions to get periods
    periods = []
    for desc in descs:
        if desc == 'Peak Ground Acceleration':
            periods.append(0.00)
        else:
            s = desc.replace(' Second Spectral Acceleration', '')
            periods.append(float(s))

    return periods, accelerations, MAFEs


def plot_hazard_curves(accelerations, MAFEs, save_path):
    plt.figure()
    plt.grid(which='Major')
    plt.grid(which='Minor')
    for key in MAFEs.keys():
        plt.plot(accelerations[key], MAFEs[key],  '-s', label=key)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Earthquake Intensity $e$ [g]')
    plt.ylabel('Mean annual frequency of exceedance $λ$')
    plt.savefig(save_path)
    plt.close()


def uniform_hazard_spectrum(MAFE, periods,
                            accelerations, MAFEs):
    rs = []  # Initialize empty list to place acceleration values
    for i, key in enumerate(MAFEs.keys()):  # for each hazard curve
        x = MAFEs[key]
        assert (max(x) >= MAFE and min(x) <= MAFE), \
            'No data available for this MAFE: {}'.format(MAFE)
        y = accelerations[key]
        # combine to remove nan values retaining inices
        data = np.column_stack((x, y))
        data = data[~np.isnan(data).any(axis=1), :]
        # perform linear interpolation to obtain intensity value fiven mafe
        f = interpolate.interp1d(data[:, 0], data[:, 1], kind='linear')
        rs.append(f(MAFE))
    rs = np.array(rs)
    periods = np.array(periods)
    return np.column_stack((periods, rs))


def target_hazard_curve(target_period, target_intensities, periods,
                        accelerations, MAFEs, plot=False):
    # Find the two closest available hazard curves
    diffPos = np.array(periods)-target_period
    diffPos[diffPos <= 0] = 1E20
    idx1 = np.argsort(diffPos)[0]
    diffNeg = np.array(periods)-target_period
    diffNeg[diffNeg >= 0] = -1E20
    idx2 = np.argsort(np.abs(diffNeg))[0]
    keys = [list(MAFEs.keys())[idx1],
            list(MAFEs.keys())[idx2]]
    closest_periods = []
    closest_periods.append(periods[idx1])
    closest_periods.append(periods[idx2])
    closest_periods = np.array(closest_periods)
    e = []
    for intensity in target_intensities:
        interpolated_MAFEs = []  # Initialize empty list to place MAFE values
        for key in keys:  # for each hazard curve
            y = MAFEs[key]
            x = accelerations[key]
            # combine to remove nan values retaining inices
            data = np.column_stack((x, y))
            data = data[~np.isnan(data).any(axis=1), :]
            # perform linear interpolation to obtain MAFE given intensity
            f = interpolate.interp1d(
                data[:, 0], data[:, 1],
                kind='linear', fill_value='extrapolate')
            interpolated_MAFEs.append(f(intensity))
        f2 = interpolate.interp1d(
            closest_periods, interpolated_MAFEs,
            kind='linear', fill_value='extrapolate')
        e.append(f2(target_period))
    return np.array(e)


def plot_target_hazard_curve(
        target_intensities, target_MAFE, accelerations, MAFEs, save_path):
    plt.figure()
    plt.grid(which='Major')
    plt.grid(which='Minor')
    for key in MAFEs.keys():
        plt.plot(accelerations[key], MAFEs[key],  '--', label=key)
    plt.plot(
        target_intensities, target_MAFE, '-s',
        label='Target', color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Earthquake Intensity $e$ [g]')
    plt.ylabel('Mean annual frequency of exceedance $λ$')
    plt.savefig(save_path)
    plt.close()


def plot_target_hazard_curve_single(target_intensities, target_MAFE, save_path):
    plt.figure()
    plt.grid(which='Major')
    plt.grid(which='Minor')
    plt.plot(
        target_intensities, target_MAFE, '-',
        label='Target', color='black')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Earthquake Intensity $e$ [g]')
    plt.ylabel('Mean annual frequency of exceedance $λ$')
    plt.savefig(save_path)
    plt.close()

def plot_uniform_hazard_spectrum(periods, rs, MAFE, save_path):
    f2 = interpolate.interp1d(periods, rs, kind='cubic')  # interpolate
    x = np.linspace(0.00, 3.00, 200)
    y = f2(x)
    plt.figure()
    plt.scatter(periods, rs)
    plt.plot(x, y)
    plt.ylim(0., 4.)
    plt.title(
        'Uniform Hazard Spectrum \n Mean annual frequency of exceedance: '
        + str(MAFE)
        )
    plt.xlabel('Period T [s]')
    plt.ylabel('Acceleration [g]')
    plt.grid()
    plt.savefig(save_path)
    plt.close()




if '__name__' == '__main__':
    pass
