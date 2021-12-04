"""
Site-specific design response spectrum
according to ASCE 7-16 chapter 21

The code is specific to the particular archetype building. A different
building might require a different logical flow, as outlined in the
code. For this building, the following decisions were made:
- The procedure outlined in chapter 21.2, is followed, using method 1
  (21.2.1.1).
"""

# imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm

# ~~~~~~~~~ #
#    MCE    #
# ~~~~~~~~~ #

# probabilistic MCE_R spectrum
# note: number 7 corresponds to a 2% prob. of exc. in 50 years
#       RotD50
spec_prob = np.genfromtxt('src/psha/UHS7.txt', skip_header=38)

ts = spec_prob[:, 0]

# modify to approximate RotD100 from RotD50 (sec. 21.2)
f_d50_to_d100 = interp1d(
    [0.01, 0.20, 1.00, 5.00, 10.0],
    [1.10, 1.10, 1.30, 1.50, 1.50], kind='linear')
spec_prob[:, 1] = spec_prob[:, 1] * f_d50_to_d100(ts)

# transform uniform hazard to uniform risk
# 21.2.1.1
# https://seismicmaps.org/
crs = 0.903
cr1 = 0.893
t_l = 8.00

f_cr = interp1d(
    [.01, .20, 1.0, 10.],
    [crs, crs, cr1, cr1], kind='linear')
spec_prob[:, 1] = spec_prob[:, 1] * f_cr(ts)

# deterministic MCE_R spectrum (sec. 21.2.2)
file_labels = ['0p01', '0p02', '0p03', '0p05', '0p075', '0p1', '0p15',
               '0p2', '0p25', '0p3', '0p4', '0p5', '0p75', '1p0',
               '1p5', '2p0', '3p0', '4p0', '5p0', '7p5', '10p0']

medns = np.array([np.genfromtxt(
    'src/psha/design/deterministic/m' + k + '.txt', skip_header=13)[:, 1]
             for k in file_labels]).T
stdvs = np.array([np.genfromtxt(
    'src/psha/design/deterministic/s' + k + '.txt', skip_header=13)[:, 1]
             for k in file_labels]).T
# modify to approximate RotD100
medns = medns * np.repeat(
    np.reshape(
        f_d50_to_d100(ts), (-1, 1)),
    len(ts), axis=1)
# calculate 84-th percentile
mean_of_logSa = np.log(medns)
p84_of_logSa = norm.ppf(0.84, loc=mean_of_logSa, scale=stdvs)
p84_of_Sa = np.exp(p84_of_logSa)
p84_of_Sa_max = np.max(p84_of_Sa, axis=1)

# determine lower bound for deterministic
f_a = 1.00  # site class D
f_v = 2.50

def determ_lower_bound(period_array, f_a, f_v, t_l):
    sa = np.full(len(period_array), 0.00)
    t1 = 0.08 * f_v / f_a
    ts = 0.40 * f_v / f_a
    for i, t in enumerate(period_array):
        if t < t1:
            sa[i] = 0.6 * f_a + (1.5 - 0.6) * f_a / t1 * t
        elif t < ts:
            sa[i] = 1.5 * f_a
        elif t < t_l:
            sa[i] = 0.6 * f_v / t
        else:
            sa[i] = 0.6 * f_v * t_l / t**2
    return sa

ifun_prob = interp1d(
    ts,
    spec_prob[:, 1], kind='linear')
ifun_detr = interp1d(
    ts, p84_of_Sa_max, kind='linear')
ifun_dlow = interp1d(
    ts, determ_lower_bound(ts, f_a, f_v, t_l),
    kind='linear')

def sam(period_array):
    sa = np.full(len(period_array), 0.00)
    for i, t in enumerate(period_array):
        sa[i] = min(ifun_prob(t), max(ifun_detr(t), ifun_dlow(t)))
    return sa

ts_refined = np.logspace(-2, 1, num=1000)
mce_spectrum = sam(ts_refined)

plt.figure()
plt.plot(
    ts,
    spec_prob[:, 1],
    color='black', label='probabilistic')
plt.plot(
    ts,
    p84_of_Sa_max,
    color='red', label='deterministic')
plt.plot(
    ts,
    determ_lower_bound(ts, f_a, f_v, t_l),
    linestyle='dashed',
    color='blue', label='determ. lower bound')
plt.plot(
    ts_refined,
    mce_spectrum,
    linestyle='dashed', linewidth=4,
    color='green', label='SaM')
plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.title("MCE-level Sa")
plt.show()


# ~~~~~~~~~~~ #
#    Design   #
# ~~~~~~~~~~~ #

# section 21.3

design_spectrum = mce_spectrum * 2. / 3.

# lower bound (sec. 21.3)
# site class D --> (ii)
# Fa = 1.00
# S1 = 0.864  (https://seismicmaps.org/) --> Fv = 2.5
s_s = 2.237
s_1 = 0.864
f_a_dslb = 1.0
f_v_dslb = 2.5
s_ms_dslb = f_a_dslb * s_s
s_m1_dslb = f_v_dslb * s_1
s_ds_dslb = 2. / 3. * s_ms_dslb
s_d1_dslb = 2. / 3. * s_m1_dslb


def design_lower_bound(period_array):
    sa = np.full(len(period_array), 0.00)
    t0 = 0.20 * s_d1_dslb / s_ds_dslb
    t_short = s_d1_dslb / s_ds_dslb
    for i, t in enumerate(period_array):
        if t < t0:
            sa[i] = s_ds_dslb * (0.40 + 0.60 * t / t0) * 0.80
        elif t < t_short:
            sa[i] = s_ds_dslb * 0.80
        elif t < t_l:
            sa[i] = s_d1_dslb / t * 0.80
        else:
            sa[i] = s_d1_dslb * t_l / t**2 * 0.80
    return sa

design_spectrum_lb = design_lower_bound(ts_refined)

ifun_des = interp1d(
    ts_refined,
    design_spectrum, kind='linear')
ifun_des_lb = interp1d(
    ts_refined,
    design_spectrum_lb, kind='linear')


def sad(period_array):
    sa = np.full(len(period_array), 0.00)
    for i, t in enumerate(period_array):
        sa[i] = max(ifun_des(t), ifun_des_lb(t))
    return sa

design_env = sad(ts_refined)


# design acceleration parameters (sec. 21.4)
sds_equiv = np.max(sad(np.linspace(0.20, 5.00, num=1000))) * 0.90
# vs_30 = 733.4 --> sd1 determined in the range of 1 < T  < 5
sd1_equiv = np.max(
    np.linspace(1.00, 5.00, num=1000) *
    sad(np.linspace(1.00, 5.00, num=1000)))

print(f"S_ds = {sds_equiv}")
print(f"S_d1 = {sd1_equiv}")


t_short = sd1_equiv / sds_equiv
t0 = 0.20 * sd1_equiv / sds_equiv
design_equiv = np.full(len(ts_refined), 0.00)
for i, t in enumerate(ts_refined):
    if t < t0:
        design_equiv[i] = sds_equiv * (0.40 + 0.60 * t / t0)
    elif t < t_short:
        design_equiv[i] = sds_equiv
    elif t < t_l:
        design_equiv[i] = sd1_equiv / t
    else:
        design_equiv[i] = sd1_equiv * t_l / t**2


plt.figure()
plt.plot(
    ts_refined,
    design_spectrum,
    color='black', label='design')
plt.plot(
    ts_refined,
    design_spectrum_lb,
    color='red', label='design lower bound')
plt.plot(
    ts_refined,
    design_env,
    linestyle='dashed', linewidth=4,
    color='green', label='Design Sa')
plt.plot(
    ts_refined,
    design_equiv,
    linestyle='dashed', linewidth=4,
    color='blue', label='Design Sa (sec. 21.4)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title("Design-level Sa")
plt.show()


