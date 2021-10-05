import numpy as np
import matplotlib.pyplot as plt
import re


def import_PEER(rel_path, filename):
    """
    Import a ground motion record from a specified PEER ground
    motion record file.
    Output is a two column matrix of time - acceleration pairs.
    Acceleration is in [g] units.
    """
    filename = rel_path + '/' + filename

    # Get all data except for the last line, where it may have fewer
    # columns and cause an error
    ag = np.genfromtxt(filename, delimiter='  ', skip_header=4, skip_footer=1)
    # Manually read the last line and append
    with open(filename) as f:
        for line in f:
            pass
        last_line = line
    last = np.fromstring(last_line, sep='  ')
    ag = np.append(ag, last)

    # Read metadata
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 2:
                # Units
                units = (line.split(sep=' ')[-1]).strip()
            elif i == 3:
                # Number of points
                npts = int(re.sub('NPTS=\s+', '', line.split(sep=', ')[0]))
                # Time step
                tmp = re.sub('DT=\s+', '', line.split(sep=', ')[1])
                tmp = re.sub('\s* SEC', '', tmp)
                dt = float(tmp)
            elif i > 29:
                break

    # Assert correct number of points and units
    assert npts == len(ag), \
        'Number of points reported in file does not match recovered points'
    assert units == 'G', \
        'Expected file to be in G units, but it isn\'t'

    # Obtain the corresponding time values
    t = np.array([x*dt for x in range(npts)])

    # Return data in the form of a matrix
    return np.column_stack((t, ag))


def response_spectrum(th, dt, zeta, T_max=5, n_Pts=200):
    """
    Calculate the linear response spectrum of an acceleration
    time history of fixed time interval dt and values given in vector
    th, and damping ratio zeta.
    T_max is the maximum period calculated
    n_Pts is the density of the periods, starting from 0.05 s.
    Periods below 0.1*dt are ignored due to insufficient step size.
    T = 0.00 is included and matched with the peak acceleration of
    the provided time history.
    """
    T = np.logspace(np.log10(0.05), np.log10(T_max), n_Pts)
    Tval = T[dt < 0.1*T]
    omega = 2 * np.pi / Tval
    c = 2 * zeta * omega
    k = omega**2
    n = len(th)
    # Initial calculations
    u = np.full(len(Tval), 0.00)  # initialize constant array
    u_prev = np.full(len(Tval), 0.00)  # initialize constant array
    umax = np.full(len(Tval), 0.00)  # initialize constant array
    khut = 1.00/dt**2 + c/(2.*dt)
    alpha = 1.00/dt**2 - c/(2.*dt)
    beta = k - 2./dt**2
    for i in range(1, n):
        phut = -th[i] - alpha*u_prev - beta*u
        u_prev = u
        u = phut/khut  # update
        # update maximum displacements
        umax[np.abs(u) > umax] = np.abs(u[np.abs(u) > umax])
    # Determine pseudo-spectral acceleration
    sa = umax * omega**2
    # Reformat data for the RS
    Ts = np.concatenate((np.array([0.00]), Tval))
    sas = np.concatenate((np.array([np.max(np.abs(th))]), sa))
    rs = np.column_stack((Ts, sas))
    return(rs)


def code_spectrum(T_vals: np.ndarray, Ss: float, S1: float,
                  Tl=8.00) -> np.ndarray:
    """
    Generate a simplified ASCE code response spectrum.
    """
    num_vals = len(T_vals)
    code_sa = np.full(num_vals, 0.00)
    T_short = S1 / Ss
    code_sa[T_vals <= T_short] = Ss
    code_sa[T_vals >= Tl] = S1 * Tl / T_vals[T_vals >= Tl]**2
    sel = np.logical_and(T_vals > T_short, T_vals < Tl)
    code_sa[sel] = S1 / T_vals[sel]
    return np.column_stack((T_vals, code_sa))


if __name__ == '__main__':
    pass
