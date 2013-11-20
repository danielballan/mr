# Copyright 2012 Daniel B. Allan
# dallan@pha.jhu.edu, daniel.b.allan@gmail.com
# http://pha.jhu.edu/~dallan
# http://www.danallan.com
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses>.

from __future__ import division
import logging
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


def msd(traj, mpp, fps, max_lagtime=100, detail=False):
    """Compute the mean displacement and mean squared displacement of one
    trajectory over a range of time intervals.

    Parameters
    ----------
    traj : DataFrame with one trajectory, including columns frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100
    detail : See below. Default False.

    Returns
    -------
    DataFrame([<x>, <y>, <x^2>, <y^2>, msd], index=t)

    If detail is True, the DataFrame also contains a column N,
    the estimated number of statistically independent measurements
    that comprise the result at each lagtime.

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.

    See also
    --------
    imsd() and emsd()
    """
    pos = traj.set_index('frame')[['x', 'y']]
    t = traj['frame']
    # Reindex with consecutive frames, placing NaNs in the gaps.
    pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
    max_lagtime = min(max_lagtime, len(t))  # checking to be safe
    lagtimes = 1 + np.arange(max_lagtime)
    disp = pd.concat([pos.sub(pos.shift(lt)) for lt in lagtimes],
                     keys=lagtimes, names=['lagt', 'frames'])
    results = mpp*disp.mean(level=0)
    results.columns = ['<x>', '<y>']
    results[['<x^2>', '<y^2>']] = mpp**2*(disp**2).mean(level=0)
    results['msd'] = mpp**2*(disp**2).mean(level=0).sum(1)  # <r^2>
    # Estimated statistically independent measurements = 2N/t
    if detail:
        results['N'] = 2*disp.icol(0).count(level=0).div(Series(lagtimes))
    results['lagt'] = results.index.values/fps
    return results[:-1]


def imsd(traj, mpp, fps, max_lagtime=100, statistic='msd'):
    """Compute the mean squared displacements of probes individually.

    Parameters
    ----------
    traj : DataFrame of trajectories of multiple probes, including
        columns probe, frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100
    statistic : {'msd', '<x>', '<y>', '<x^2>', '<y^2>'}, default is 'msd'
        The functions msd() and emsd() return all these as columns. For
        imsd() you have to pick one.

    Returns
    -------
    DataFrame([Probe 1 msd, Probe 2 msd, ...], index=t)

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.
    """
    _verify_user_is_not_boneheaded(mpp, fps)
    ids = []
    msds = []
    # Note: Index is set by msd, so we don't need to worry
    # about conformity here.
    for pid, ptraj in traj.groupby('probe'):
        msds.append(msd(ptraj, mpp, fps, max_lagtime, False))
        ids.append(pid)
    results = pd.concat(msds, keys=ids)
    # Swap MultiIndex levels so that unstack() makes probes into columns.
    results = results.swaplevel(0, 1)[statistic].unstack()
    lagt = results.index.values.astype('float64')/float(fps)
    results.set_index(lagt, inplace=True)
    results.index.name = 'lag time [s]'
    return results


def emsd(traj, mpp, fps, max_lagtime=100, detail=False):
    """Compute the mean squared displacements of an ensemble of probes.

    Parameters
    ----------
    traj : DataFrame of trajectories of multiple probes, including
        columns probe, frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100
    detail : Set to True to include <x>, <y>, <x^2>, <y^2>. Returns
        only <r^2> by default.

    Returns
    -------
    Series[msd, index=t] or, if detail=True,
    DataFrame([<x>, <y>, <x^2>, <y^2>, msd], index=t)

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.
    """
    _verify_user_is_not_boneheaded(mpp, fps)
    ids = []
    msds = []
    for pid, ptraj in traj.reset_index(drop=True).groupby('probe'):
        msds.append(msd(ptraj, mpp, fps, max_lagtime, True))
        ids.append(pid)
    msds = pd.concat(msds, keys=ids, names=['probe', 'frame'])
    results = msds.mul(msds['N'], axis=0).mean(level=1)  # weighted average
    results = results.div(msds['N'].mean(level=1), axis=0)  # normalized
    # Above, lagt is lumped in with the rest for simplicity and speed.
    # Here, rebuild it from the frame index.
    results.set_index('lagt', inplace=True)
    results.index.name = 'lag time [s]'
    if not detail:
        return results['msd']
    return results

ten_points_two_decades = np.logspace(0, 2, num=10).round()


def tp_msd(traj, mpp, fps, a, lagframes=ten_points_two_decades, bins=False):
    """Compute the two-point mean-squared displacement.

    Parameters
    ----------
    traj : DataFrame of trajectories of multiple probes, including
        columns probe, frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    a : particle radius in microns
    lagframes : intervals of frames out to which MSD is computed
       Default 10 log-spaced intevals from 1 to 100
    bins : bin by particle separation distance R, default False

    Returns
    -------
    Series(msd, index=lagtime)

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.
    """
    _verify_user_is_not_boneheaded(mpp, fps)
    D = tp_corr(traj, lagframes, bins=False, _reduce=True)
    D*= mpp
    result = 2/a*D
    result.index = result.index.to_series().astype('float64')/fps
    return result


def tp_corr(traj, lagframes=ten_points_two_decades, bins=False, _reduce=False):
    """Compute the diffusion tensor, i.e., the two-point correlation function
    of probe displacement.

    Parameters
    ----------
    traj : DataFrame of trajectories of multiple probes, including
        columns probe, frame, x, and y
    lagframes : intervals of frames out to which MSD is computed
       Default 10 log-spaced intevals from 1 to 100
    bins : bin by particle separation distance R, default False

    Returns
    -------
    DataFrame([R, para, perp], index=[lagtime, R])
    para is D_rr; perp is D_tt.

    Notes
    -----
    Prepare trajectory data by subtracting drift and "eroding," meaning 
    disarding point in the vicinity of a gap or an end, because these points
    can be spurrious.
    """
    result = pd.concat([_tp_corr(traj, lf, bins, _reduce) for lf in lagframes])
    return result


def disp(a, step, dropna=False):
    result = a.loc[a.index.values + step].values - a  # do not align
    if not dropna:
        return result
    else:
        return result.dropna()


def _tp_corr(traj, lagframe, bins=False, _reduce=False):
    """Compute the two-point correlation function at a single lagtime.

    Returns
    -------
    DataFrame([R, para, perp], index=[lagframe, lagframe, ...]
    """
    traj.set_index('frame', inplace=True, drop=False)  # to be sure
    dr_cache = {}
    _disp = lambda x: disp(x, lagframe)
    grouped = traj[['x', 'y']].groupby(traj['probe'])
    # PERF: Copy data out of groups to avoid repeated index lookup.
    try:
        probes = {k: v for k, v in grouped}
    except MemoryError:
        probes = grouped
        grouped.get = grouped.get_group  # monkey patch
    probe_ids = traj['probe'].unique()
    probe_ids.sort()
    D = []
    dr_cache = {}
    count = len(probe_ids)
    logger.info("%d-frame lag, %d probes, %d permutations" %
                (lagframe, count, count*(count - 1)//2))
    # PERF: Below, where I reindex, I am manually aligning the data
    # rather than allowing pandas for do it automatically, repeatedly
    # for each operation. I get a substantial speed boost.
    for p1 in probe_ids:
        r1 = probes.get(p1)
        if p1 in dr_cache:
            dr1 = dr_cache[p1]
        else:
            dr1 = _disp(r1).dropna()
            dr_cache[p1] = dr1
        index = dr1.index
        dr1 = dr1.values
        r1 = r1.reindex(index).values
        for p2 in probe_ids[probe_ids > p1]:
            r2 = probes.get(p2)
            if p2 in dr_cache:
                dr2 = dr_cache[p2]
            else:
                dr2 = _disp(r2)
                dr_cache[p2] = dr2
            dr2 = dr2.reindex(index).values
            r = r2.reindex(index).values - r1  # vector
            R = np.sqrt(np.sum(r**2, axis=1))
            n = r/R.reshape(-1, 1)  # unit vector
            para = np.sum(dr1 * n, 1) * np.sum(dr2 * n, 1)
            p = n[:, ::-1] * [-1, 1]  # p is perpendicular to n
            perp = np.sum(dr1 * p, 1) * np.sum(dr2 * p, 1)
            # PERF: Assemble a DataFrame-like array, but don't call the
            # DataFrame constructor until outside the loop.
            df_shaped_array = np.column_stack([R, para, perp])
            D.append(df_shaped_array)
        del dr_cache[p1]
    D = DataFrame(np.vstack(D), columns=['R', 'para', 'perp']).dropna()
    if len(D) == 0:
        warnings.warn("No data for lagtime interval of %d frames" % lagframe,
                      UserWarning)
        return None
    if bins:
        # Reduce data spatially.
        _, bins = np.histogram(D['R'], bins=bins)
        grouper = np.digitize(D['R'], bins)
        D = D.groupby(grouper).mean()
    D.index = lagframe*np.ones_like(D.index)
    if _reduce:
        # Reduce all data into a single point for tp_msd.
        # The result is a scalar, which we rebox as a DataFrame
        # for consistency. (It will be passed to pd.concat.)
        D = DataFrame([D[['R', 'para']].prod(1).mean(0)], index=[lagframe])
    return D


def compute_drift(traj, smoothing=0):
    """Return the ensemble drift, x(t).

    Parameters
    ----------
    traj : DataFrame of trajectories, including columns x, y, frame, and probe
    smoothing : integer
        Smooth the drift using a forward-looking rolling mean over
        this many frames.

    Returns
    -------
    drift : DataFrame([x, y], index=frame)

    Examples
    --------
    compute_drift(traj).plot() # Default smoothing usually smooths too much.
    compute_drift(traj, 0).plot() # not smoothed
    compute_drift(traj, 15).plot() # Try various smoothing values.

    drift = compute_drift(traj, 15) # Save good drift curves.
    corrected_traj = subtract_drift(traj, drift) # Apply them.
    """
    # Probe by probe, take the difference between frames.
    delta = pd.concat([t.set_index('frame', drop=False).diff()
                       for p, t in traj.groupby('probe')])
    # Keep only deltas between frames that are consecutive.
    delta = delta[delta['frame'] == 1]
    # Restore the original frame column (replacing delta frame).
    delta['frame'] = delta.index
    dx = delta.groupby('frame').mean()
    if smoothing > 0:
        dx = pd.rolling_mean(dx, smoothing, min_periods=0)
    x = dx.cumsum(0)[['x', 'y']]
    return x


def subtract_drift(traj, drift=None):
    """Return a copy of probe trajectores with the overall drift subtracted out

    Parameters
    ----------
    traj : DataFrame of trajectories, including columns x, y, and frame
    drift : optional DataFrame([x, y], index=frame) like output of
         compute_drift(). If no drift is passed, drift is computed from traj.

    Returns
    -------
    traj : a copy, having modified columns x and y
    """

    if drift is None:
        drift = compute_drift(traj)
    return traj.set_index('frame', drop=False).sub(drift, fill_value=0)


def is_typical(msds, frame=23, lower=0.1, upper=0.9):
    """Examine individual probe MSDs, distinguishing outliers from those
    in the central quantile.

    Parameters
    ----------
    msds : DataFrame like the output of imsd()
        Columns correspond to probes, indexed by lagtime measured in frames.
    frame : integer frame number
        Compare MSDs at this lagtime. Default is 23 (1 second at 24 fps).
    lower : float between 0 and 1, default 0.1
        Probes with MSD up to this quantile are deemed outliers.
    upper : float between 0 and 1, default 0.9
        Probes with MSD above this quantile are deemed outliers.

    Returns
    -------
    Series of boolean values, indexed by probe number
    True = typical probe, False = outlier probe

    Example
    -------
    m = mr.imsd(traj, MPP, FPS)
    # Index by probe ID, slice using boolean output from is_typical(), and then
    # restore the original index, frame number.
    typical_traj = traj.set_index('probe').ix[is_typical(m)].reset_index()\
        .set_index('frame', drop=False)
    """
    a, b = msds.ix[frame].quantile(lower), msds.ix[frame].quantile(upper)
    return (msds.ix[frame] > a) & (msds.ix[frame] < b)


def vanhove(pos, lagtime=23, mpp=1, ensemble=False, bins=24):
    """Compute the van Hove correlation function at given lagtime (frame span).

    Parameters
    ----------
    pos : DataFrame of x or (or!) y positions, one column per probe, indexed
        by frame
    lagtime : integer interval of frames
        Compare the correlation function at this lagtime. Default is 23
        (1 second at 24 fps).
    mpp : microns per pixel, DEFAULT TO 1 because it is usually fine to use
        pixels for this analysis
    ensemble : boolean, defaults False
    bins : integer or sequence
        Specify a number of equally spaced bins, or explicitly specifiy a
        sequence of bin edges. See np.histogram docs.

    Returns
    -------
    vh : If ensemble=True, a DataFrame with each probe's van Hove correlation
        function, indexed by displacement. If ensemble=False, a Series with
        the van Hove correlation function of the whole ensemble.

    Example
    -------
    pos = traj.set_index(['frame', 'probe'])['x'].unstack() # probes as columns
    vh = vanhove(pos)
    """
    # Reindex with consecutive frames, placing NaNs in the gaps.
    pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
    if lagtime > pos.index.values.max():
        raise ValueError("There is a no data out to frame %s." % lagtime)
    disp = mpp*pos.sub(pos.shift(lagtime))
    # Let np.histogram choose the best bins for all the data together.
    values = disp.values.flatten()
    values = values[np.isfinite(values)]
    global_bins = np.histogram(values, bins=bins)[1]
    # Use those bins to histogram each column by itself.
    vh = disp.apply(
        lambda x: Series(np.histogram(x, bins=global_bins, density=True)[0]))
    vh.index = global_bins[:-1]
    if ensemble:
        return vh.sum(1)/len(vh.columns)
    else:
        return vh


def diagonal_size(single_trajectory, pos_columns=['x', 'y'], t_column='frame'):
    """Measure the diagonal size of a trajectory.

    Parameters
    ----------
    single_trajectory : DataFrame containing a single trajectory
    pos_columns = ['x', 'y']
    t_column = 'frame'

    Returns
    -------
    float : length of diangonal of rectangular box containing the trajectory

    Example
    -------
    >>> many_trajectories.groupby('probe').agg(mr.diagonal_size)
    >>> criterion = lambda x: mr.diagonal_size(x) > 5
    >>> many_trajectories.groupby('probe').filter(criterion)
    """

    pos = single_trajectory.set_index(t_column)[pos_columns]
    return np.sqrt(np.sum(pos.apply(np.ptp)**2))


def relate_frames(t, frame1, frame2):
    a = t[t.frame == frame1]
    b = t[t.frame == frame2]
    j = a.set_index('probe')[['x', 'y']].join(
        b.set_index('probe')[['x', 'y']], rsuffix='_b')
    j['dx'] = j.x_b - j.x
    j['dy'] = j.y_b - j.y
    j['dr'] = np.sqrt(j['dx']**2 + j['dy']**2)
    j['direction'] = np.arctan2(j.dy, j.dx)
    return j


def direction_corr(t, frame1, frame2):
    """Compute the cosine between every pair of probes' displacements.

    Parameters
    ----------
    t : DataFrame containing columns probe, frame, x, and y
    frame1 : frame number
    frame2 : frame number

    Returns
    -------
    DataFrame, indexed by probe, including dx, dy, and direction
    """
    j = relate_frames(t, frame1, frame2)
    cosine = np.cos(np.subtract.outer(j.direction, j.direction))
    r = np.sqrt(np.subtract.outer(j.x, j.x)**2 +
                np.subtract.outer(j.y, j.y)**2)
    upper_triangle = np.triu_indices_from(r, 1)
    result = DataFrame({'r': r[upper_triangle],
                        'cos': cosine[upper_triangle]})
    return result


def velocity_corr(t, frame1, frame2):
    """Compute the velocity correlation between
    every pair of probes' displacements.

    Parameters
    ----------
    t : DataFrame containing columns probe, frame, x, and y
    frame1 : frame number
    frame2 : frame number

    Returns
    -------
    DataFrame, indexed by probe, including dx, dy, and direction
    """
    j = relate_frames(t, frame1, frame2)
    cosine = np.cos(np.subtract.outer(j.direction, j.direction))
    r = np.sqrt(np.subtract.outer(j.x, j.x)**2 +
                np.subtract.outer(j.y, j.y)**2)
    dot_product = cosine*np.abs(np.multiply.outer(j.dr, j.dr))
    upper_triangle = np.triu_indices_from(r, 1)
    result = DataFrame({'r': r[upper_triangle],
                        'dot_product': dot_product[upper_triangle]})
    return result


def theta_entropy(pos, bins=24, plot=True):
    """Plot the distrbution of directions and return its Shannon entropy.

    Parameters
    ----------
    pos : DataFrame with columns x and y, indexed by frame
    bins : number of equally-spaced bins in distribution. Default 24.
    plot : plot direction historgram if True

    Returns
    -------
    float : Shannon entropy

    Examples
    --------
    >>> theta_entropy(t[t['probe'] == 3].set_index('frame'))
    """

    disp = pos - pos.shift(1)
    direction = np.arctan2(disp['y'], disp['x'])
    bins = np.linspace(-np.pi, np.pi, bins + 1)
    if plot:
        Series(direction).hist(bins=bins)
    return shannon_entropy(direction.dropna(), bins)


def shannon_entropy(x, bins):
    """Compute the Shannon entropy of the distribution of x."""
    hist = np.histogram(x, bins)[0]
    hist = hist.astype('float64')/hist.sum()  # normalize probablity dist.
    entropy = -np.sum(np.nan_to_num(hist*np.log(hist)))
    return entropy


def min_rolling_theta_entropy(pos, window=24, bins=24):
    """Compute the minimum Shannon entropy in any window.

    Parameters
    ----------
    pos : DataFrame with columns x and y, indexed by frame
    window : number of observations per window
    bins : number of equally-spaced bins in distribution. Default 24.

    Returns
    -------
    float : Shannon entropy

    Examples
    --------
    >>> theta_entropy(t[t['probe'] == 3].set_index('frame'))

    >>> S = t.set_index('frame').groupby('probe').apply(
    ...     mr.min_rolling_theta_entropy)
    """

    disp = pos - pos.shift(1)
    direction = np.arctan2(disp['y'], disp['x'])
    bins = np.linspace(-np.pi, np.pi, bins + 1)
    f = lambda x: shannon_entropy(x, bins)
    return pd.rolling_apply(direction.dropna(), window, f).min()


def _verify_user_is_not_boneheaded(mpp, fps):
    if not mpp > 0:
        raise ValueError("microns per pixel (mpp) should be larger than 0." +
                         " Did you accidentally do integer division?")
    if not fps > 0:
        raise ValueError("frames per second (fps) should be larger than 0." +
                         " Did you accidentally do integer division?")
