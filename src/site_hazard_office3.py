"""
Generation of site hazard curve for ground motion selection for a
time-based assessment using OpenSHA PSHA output.

* Required Input *

Required input for this script is obtained from OpenSHA.
The module used is this:
HazardCurveGUI-2021_11_13-b736875b.jar
Found here:
http://opensha.usc.edu/apps/opensha/nightlies/latest/

Parameters:
  ERF & Time Span
    Erq Rup Forecast: Mean UCERF3
    Mean UCERF3 Presets: (Custom)
    Apply aftershock filter: check
    Aleatory Mag-Aara StdDev: 0.0
    Background Seismicity: Include
    Treat b.s. as: Point Sources
    Fault Grid Spacing: 1.0 km
    Probability model: Poisson
    Use Mean Upper Depth: uncheck
    Rup Mag Averaging Tol. : 1.0
    Rupture Rake to Use: Def. Model Mean
    Fault Models: Both
    Ignore Cache: check
    Time Span: 1 year
  IMR, IMT & Site
    Set Site Params: set long, lat and then
      control panel > set site params from web
      (obtains the rest of the site params)
  Set IMR:
    NGAWest2 2014 Averaged Attenuation
    Additional Uncertainty: Disabled
    Component: RotD50
    Gaussian Truncation: None

Using HazardCurveGUI-2021_11_13-b736875b.jar, plot PGA, and then
switch to Sa and plot all available Sa options in ascending order,
without clearing the plot. Then export the data in a txt file from
File > Save.  This is how hazard_curves.txt was obtained.

Note

  Assuming Poisson distributed earthquake occurence,


  p_occurence = 1 - exp(-t/T)

  where:
    P_exceedance is the probability of 1 or more occurences,
    t is the period of interest (1 year, 50 years, etc.),
    T is the return period (e.g. 475 years, 2475 years),
    1/T is called the `occurence rate` or `frequency`

"""

# %% Imports

import sys
sys.path.append("src")

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import os
import tikzplotlib
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


# Building-specific first-mode response period
# (Tbar = (Tx + Ty) / 2)

# ~~~~~~~~~~ #
Tbar = 0.82  #
# ~~~~~~~~~~~#

# long = -122.259
# lat = 37.871
# vs30 = 733.4

# ~~~~~~~~~~~~~~~~~~~~ #
# Parse OpenSHA output #
# ~~~~~~~~~~~~~~~~~~~~ #

filepath = 'src/psha_office3/hazard_curves.txt'

with open(filepath, 'r') as f:
    line = f.read()

contents = line.split('\n')

i_begin = []
num_points = []
periods = []
for i, c in enumerate(contents):
    words = c.split(' ')
    if words[0] == 'X,' and words[1] == 'Y' and words[2] == 'Data:':
        i_begin.append(i + 1)
    if words[0] == 'Num':
        num_points.append(int(words[2]))
    if words[0] == 'IMT' and words[1] == '=':
        if words[2] == 'PGA':
            periods.append(0.00)
        elif words[2] == 'SA;':
            periods.append(float(words[6].replace(';', '')))

i_end = []
for pair in zip(i_begin, num_points):
    i_end.append(pair[0] + pair[1])

accelerations = []
MAPEs = []
for i, pair in enumerate(zip(i_begin, i_end)):
    data = np.genfromtxt(contents[pair[0]: pair[1]])
    accelerations.append(data[:, 0])
    MAPEs.append(data[:, 1])

# OpenSHA operates in probability space.
# Probabilities have to be converted to frequencies of exceedance
MAFEs = []
for mape in MAPEs:
    MAFEs.append(
        -np.log(1-mape)
    )


# ~~~~~~~~~~~~~~~~~~ #
# Plot hazard curves #
# ~~~~~~~~~~~~~~~~~~ #

# if not os.path.exists('figures/office3/site_hazard'):
#     os.makedirs('figures/office3/site_hazard')
# save_path = 'figures/office3/site_hazard/hazard_curves.pdf'

# plt.figure(figsize=(12, 10))
# plt.grid(which='Major')
# plt.grid(which='Minor')
# for i in range(len(periods)):
#     plt.plot(accelerations[i], MAFEs[i], '-s',
#              label='T = ' + str(periods[i]) + ' s (OpenSHA)')
# plt.xscale('log')
# plt.yscale('log')
# plt.axhline(2e-2)
# plt.axhline(4e-4)
# plt.legend()
# plt.xlabel('Earthquake Intensity $e$ [g]')
# plt.ylabel('Mean annual frequency of exceedance $λ$')
# # plt.show()
# plt.savefig(save_path)
# plt.close()




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Obtain period-specific hazard curve #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


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
# save_path = 'figures/office3/site_hazard/target_hazard_curve.pdf'
# plt.figure(figsize=(12, 10))
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
# # plt.show()
# plt.savefig(save_path)
# plt.close()


# Split intensity range to m intervals

# ~~~~~ #
m = 16  #
# ~~~~~ #

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

for i, thing in enumerate(return_period_midpoints):
    print(f'{thing:.0f}')



    
save_path = 'figures/office3/site_hazard/hazard_levels.pdf'
plt.figure(figsize=(8, 6))
plt.grid(which='Major')
plt.grid(which='Minor')
plt.plot(
    target_acceleration, target_MAFE, '-',
    label='Hazard Curve', color='black')
plt.scatter(e_Endpoints, MAFE_Endpoints,
            s=80, facecolors='none', edgecolors='k',
            label='Interval Endpoints')
plt.scatter(e_Midpoints, MAFE_Midpoints,
            s=40, facecolors='k', edgecolors='k',
            label='Interval Midpoints')
for i, txt in enumerate(range(1, m+1)):
    plt.annotate(txt, (e_Midpoints[i], MAFE_Midpoints[i]))
plt.axvline(SaMin, color='k', linestyle='dashed',
            label='Intensity Range')
plt.axvline(SaMax, color='k', linestyle='dashed')
plt.axhline(mafe_mce, color='red', label='MCE')
plt.axhline(mafe_des, color='blue', label='Design')
plt.legend()
plt.xlabel('Earthquake Intensity $e$ [g] ( Sa(T* = %.3f s) )' % (Tbar))
plt.ylabel('Mean annual frequency of exceedance $λ$')
plt.xlim((0.00-0.05, SaMax + 0.05))
plt.ylim((1e-4, 1))
plt.yscale('log')
plt.show()
# plt.savefig(save_path)
# import tikzplotlib
# tikzplotlib.save('hzcurv.tex')
plt.close()


# stroe hazard curve interval data
# (will be used to obtain mean annual rate of exceedance of
#  decision variables)

if not os.path.exists('analysis/office3/site_hazard'):
    os.makedirs('analysis/office3/site_hazard')

interv_df = pd.DataFrame(
    np.column_stack(
        (e_Midpoints, delta_e,
         delta_lamda,
         MAFE_Midpoints,
         MAPE_Midpoints,
         return_period_midpoints)),
    columns=['e', 'de', 'dl', 'freq', 'prob', 'T'],
    index=range(1, m+1))

interv_df.to_csv('analysis/office3/site_hazard/Hazard_Curve_Interval_Data.csv')


# At this point, with the obtained midpoint probabilities of exceedance,
# we have to go back to OpenSHA and:
# - generate uniform hazard spectra (UHS*.txt files)
# - deaggregate hazard @ Tmin and Tmax for each interval
#   (DeagTmin*.txt, DeagTmax*.txt files)
# - compute mean spectrum and stdev using attenuation relationship
#   (mTmin*.txt and sTmin*.txt files)
# and then proceed to part 2 to generate composite spectra


# Part 2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Target Spectra for Ground Motion Selection #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# We implement what is outlined in:
#   Kwong NS, Chopra AK. Selecting, scaling, and orienting three
#   components of ground motions for intensity-based assessments at
#   far-field sites. Earthquake
#   Spectra. 2020;36(3):1013-1037. doi:10.1177/8755293019899954
#
# (currently only for the Horizontal component)

# # Sanity Check
# # Calculate uniform hazard spectra for each interval (manually)
# interp_funcs = {}
# for i, period in enumerate(periods):
#     mafe = MAFEs[i]
#     acce = accelerations[i]
#     interp_funcs[period] = interp1d(
#         mafe, acce, kind='linear')

# uhss = []
# for j, period in enumerate(periods):
#     uhss.append(interp_funcs[period](MAFE_Midpoints))
# uhss = np.array(uhss)

# # interpolate between the periods
# x_pts = np.logspace(-4, 1, 100)
# y_pts = np.full((len(x_pts), m), 0.00)

# for i in range(m):
#     vec = uhss[:, i]
#     f = interp1d(
#         periods, vec, kind='linear')
#     y_pts[:, i] = f(x_pts)

# plt.figure()
# plt.grid(which='Major')
# plt.grid(which='Minor')
# for i_plt in range(m):
#     plt.plot(
#         x_pts, y_pts[:, i_plt],
#         label='R.P. = %.0f yrs' % return_period_midpoints[i_plt])
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Period T [s]')
# plt.ylabel('PSa [g]')
# plt.title('Uniform Hazard Spectra')
# plt.xlim((1e-2, 1e1))
# plt.legend()
# plt.show()
# plt.close()


# Uniform Hazard Spectra
uhss = [np.genfromtxt('src/psha_office3/UHS'+str(i)+'.txt', skip_header=38)
        for i in range(1, m+1)]

# mean spectra and standard deviations (from deaggregation)
meansTmax = [np.genfromtxt('src/psha_office3/m'+str(i)+'.txt', skip_header=13)
             for i in range(1, m+1)]
stdevsTmax = [np.genfromtxt('src/psha_office3/s'+str(i)+'.txt', skip_header=13)
              for i in range(1, m+1)]


# # export UHSs for presentation
# filepath = '~/google_drive_encr/UCB/research/projects/299_report/presentation/data/target_spectrum/'
# for i in range(m):
#     pd.DataFrame(uhss[i]).to_csv(f'{filepath}/uhs_{i+1}.txt', index=None, sep=' ', header=None)



def read_epsilon(filename):
    with open(filename, 'r') as f:
        contents = f.read()
    contents = contents.split('\n')
    for cont in contents:
        words = cont.strip().split(' ')
        if words[0] == 'Ebar':
            epsilon = float(words[2])
            break
    return epsilon
    

epsTmax = [read_epsilon('src/psha_office3/deag'+str(i)+'.txt')
           for i in range(1, m+1)]


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

# ~ compare with fig. 11
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
    idx = np.argwhere(ts_expanded>Tbar)[0, 0]
    ts_expanded = np.concatenate((ts_expanded[0:idx], np.array((Tbar,)), ts_expanded[idx::]), axis=0)
    uhs = uhss[i][:, 1]
    fuhs = interp1d(
        ts, uhs, kind='linear')
    fmTmax = interp1d(
        meansTmax[i][:, 0], meansTmax[i][:, 1],
        kind='linear')
    fsTmax = interp1d(
        stdevsTmax[i][:, 0], stdevsTmax[i][:, 1],
        kind='linear')
    correlationsTmin = np.full(len(ts_expanded), 0.00)
    correlationsTmax = np.full(len(ts_expanded), 0.00)
    for j, t in enumerate(ts_expanded.tolist()):
        correlationsTmax[j] = correl(t, Tbar)
    eTmax = np.log(fuhs(Tbar)/fmTmax(Tbar))/fsTmax(Tbar)
    # print(f'%.2f %.2f' % (eTmax, epsTmax[i]))
    cmss[:, i] = fmTmax(ts_expanded) * np.exp(eTmax * correlationsTmax * fsTmax(ts_expanded))
    # print(uhs[idx_tmax]/cmssTmax[idx_tmax, i])


# # export nondirective CMSs for presentation
# filepath = '~/google_drive_encr/UCB/research/projects/299_report/presentation/data/target_spectrum/'
# for i in range(m):
#     pd.DataFrame(np.column_stack((ts_expanded, cmss[:, i]))).to_csv(f'{filepath}/cms_{i+1}.txt', index=None, sep=' ', header=None)


# # uniform hazard spectra & cms
# plt.figure()
# plt.grid(which='Major')
# plt.grid(which='Minor')
# for i, opsh in enumerate(uhss):
#     # plt.plot(opsh[:, 0], opsh[:, 1],
#     #          label='R.P. = %.0f yrs' % return_period_midpoints[i])
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
# save_path = 'figures/office3/site_hazard/composite_spectra.pdf'
# plt.figure()
# # plt.grid(which='Major')
# # plt.grid(which='Minor')
# for i, opsh in enumerate(uhss):
#     plt.plot(ts_expanded, cms_drctv[:, i], linestyle='dotted')
# plt.gca().set_prop_cycle(None)
# for i, opsh in enumerate(uhss):
#     plt.plot(ts_expanded, cms_drctv[:, i],
#              label='R.P. = %.0f yrs' % return_period_midpoints[i],
#              color='k')
# for i, opsh in enumerate(uhss):
#     plt.plot(ts_expanded, cmss[:, i],
#              label='R.P. = %.0f yrs' % return_period_midpoints[i],
#              color='k', linestyle='dashed')
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

if not os.path.exists('analysis/office3/site_hazard'):
    os.makedirs('analysis/office3/site_hazard')

for i in range(m):
    np.savetxt(
        'analysis/office3/site_hazard/spectrum_'+str(i+1)+'.csv',
        np.column_stack((ts_expanded, cms_drctv[:, i])),
        header='Hazard Level '+str(i+1)+',\r\n,\r\nT (s),Sa (g)',
        delimiter=',', comments='', fmt='%.5f', newline='\r\n')



# useful output for MSE scaling for use in
# the PEER NGA-West2 website
weights_per = np.logspace(-2, 1, 30)
wstr_per = ''
for weight in weights_per:
    wstr_per += '%.3f, ' % weight
print(wstr_per)
weights_w = np.ones(30)
wstr_w = ''
for weight in weights_w:
    wstr_w += '%.0f, ' % weight
print(wstr_w)
