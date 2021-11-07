# %% Imports

import sys
sys.path.append("src")

from USGS_Hazard_Data import retrieve_hazard_curves
from USGS_Hazard_Data import plot_hazard_curves
from USGS_Hazard_Data import uniform_hazard_spectrum
from USGS_Hazard_Data import plot_uniform_hazard_spectrum
from USGS_Hazard_Data import target_hazard_curve
from USGS_Hazard_Data import plot_target_hazard_curve
from USGS_Hazard_Data import plot_target_hazard_curve_single

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
import numpy as np
import os

longitude = -122.299
latitude = 37.895
vs30 = 1150

# Location-specific earthquake hazard data from USGS
periods, accelerations, MAFEs = retrieve_hazard_curves(
    str(longitude), str(latitude), str(vs30), cache=True
)

# Plot hazard curves
if not os.path.exists('figures/site_hazard'):
    os.makedirs('figures/site_hazard')
plot_hazard_curves(accelerations, MAFEs, 'figures/site_hazard/hazard_curves.pdf')

# Building's fundamental translational periods in two orthogonal directions
# are averaged out to give the first mode average period Tbar
Tbar = 0.735  # from ASCE's T_max


# Determine SaMin based on P-58 guidelines
if Tbar <= 1.00:
    SaMin = 0.05
else:
    SaMin = 0.05/Tbar


# Interpolate to obtain Tbar-specific hazard curve
target_intensities = np.logspace(np.log(SaMin), -0.02, 16)
target_MAFE = target_hazard_curve(Tbar, target_intensities,
                                  periods, accelerations, MAFEs)

data = np.column_stack((target_intensities, target_MAFE))
np.savetxt('analysis/site_hazard/hazard_curve.csv', data, delimiter=',')

plot_target_hazard_curve(
    target_intensities, target_MAFE, accelerations, MAFEs,
    'figures/site_hazard/target_hazard_curve.pdf')

plot_target_hazard_curve_single(
    target_intensities, target_MAFE,
    'figures/site_hazard/target_hazard_curve_lin.pdf')


# Determine SaMax from the hazard curve based on P-58 guidelines
# Interpolate: From intensity e [g] to MAFE λ
fHazMAFEtoSa = interp1d(target_MAFE, target_intensities,
                        kind='linear', fill_value='extrapolate')
# Interpolate: Inverse (From MAFE λ to intensity e [g])
fHazSatoMAFE = interp1d(target_intensities, target_MAFE,
                        kind='linear', fill_value='extrapolate')

SaMax = fHazMAFEtoSa(0.0002)

# Split to m intervals

# Split intensity range to m intervals
########
m = 8  #
########

# x-axis: EQ intensity [g] (@ endpoints)
e_Endpoints = np.linspace(SaMin, SaMax, m+1)
# y-axis: MAFE λ (@ endpoints)
MAFE_Endpoints = fHazSatoMAFE(e_Endpoints)
# x-axis: EQ intensity [g] (@ midpoints)
e_Midpoints = np.array([fHazMAFEtoSa(
    (MAFE_Endpoints[i]+MAFE_Endpoints[i+1])/2.00) for i in range(m)])
MAFE_Midpoints = fHazSatoMAFE(e_Midpoints)
MAFE_Delta = np.array([MAFE_Endpoints[i]-MAFE_Endpoints[i+1]
                       for i in range(m)])


if not os.path.exists('analysis/site_hazard'):
    os.makedirs('analysis/site_hazard')
np.savetxt('analysis/site_hazard/Hazard_Curve_Interval_Data.csv',
           np.column_stack((e_Midpoints, MAFE_Midpoints, MAFE_Delta)))


# Calculate uniform hazard spectra for each interval

for i in range(m):
    uhs_data = uniform_hazard_spectrum(
        MAFE_Midpoints[i], periods, accelerations, MAFEs)
    np.savetxt('analysis/site_hazard/uhs_'+str(i+1)+'.csv', uhs_data,
               header='Hazard Level '+str(i)+',\r\n,\r\nT (s),Sa (g)',
               delimiter=',', comments='', fmt='%.5f', newline='\r\n')
    plot_uniform_hazard_spectrum(periods, uhs_data[:, 1], MAFE_Midpoints[i],
                                 'figures/site_hazard/uhs_'+str(i+1)+'.pdf')
