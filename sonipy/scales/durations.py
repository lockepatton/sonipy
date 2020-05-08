from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

s_to_ms = 1000.

def from_max_ms(x, dtime_max):
    """
    Calculates a time scale in units of x value / time (ms), by matching the largest time step in the x array to the input dtime_max.

    Parameters
    ----------
    x : arr
        Array of x positions that will correspond to blip times.
    dtime_max : float
        Maximum allowed time difference between blips (in ms).

    Returns
    -------
    float
        Returns a time scale calculated from the maximum dt.

    """
    dx = np.subtract(x[1:], x[0:-1])
    dx_max = max(dx)
    scale = float(dx_max) / float(dtime_max)  # x value / ms
    return scale


def from_min_ms(x, dtime_min):
    """
    Calculates a time scale in units of x value / time (ms), by matching the smallest time step in the x array to the input dtime_min.

    Parameters
    ----------
    x : arr
        Array of x positions that will correspond to blip times.
    dtime_min : float
        Minimum allowed time difference between blips (in ms).

    Returns
    -------
    float
        Returns a time scale calculated from the minimum dt.

    """
    dx = np.subtract(x[1:], x[0:-1])
    dx_min = np.min(dx[np.nonzero(dx)])  # min(dt)
    scale = float(dx_min) / float(dtime_min)  # x value / ms
    return scale


def from_total_ms(x, time_total):
    """
    Calculates a time scale in units of x value / time (ms), by matching the total length of x array to the input time_total.

    Parameters
    ----------
    x : arr
        Array of x positions that will correspond to blip times.
    time_total : float
        Total time difference between first and last blip (in ms).

    Returns
    -------
    float
        Returns a time scale calculated from the total dt.

    """
    dx_total = abs(x[0] - x[-1])
    scale = float(dx_total) / float(time_total) # x value / ms
    return scale


def getScale(x, time_total=None, dtime_min=None, dtime_max=None):
    """
    Function that takes in a total time between blips, minimum difference between successive blips or maximum difference between successive blips and returns a time scale in units of x value / ms.

    Parameters
    ----------
    x : arr
        Array of x positions that will correspond to blip times.
    time_total : float
        Total time difference between first and last blip (in ms).
    dtime_min : float
        Minimum allowed time difference between blips (in ms).
    dtime_max : float
        Maximum allowed time difference between blips (in ms).

    Returns
    -------
    float
        Returns a time scale in units of x value / time (ms).

    """

    # checking for which inputs were given
    inputs = []
    if time_total != None:
        inputs.append('time_total')
    if dtime_max != None:
        inputs.append('dtime_max')
    if dtime_min != None:
        inputs.append('dtime_min')
    n_inputs = len(inputs)

    # raising exception if anything other than one input were given
    if n_inputs != 1:
        raise Exception('User needs to define time_total, dtime_max, or dtime_min to find time scale. You inputted {} inputs, which were {}.'.format(
            n_inputs, inputs))

    if 'time_total' in inputs:
        scale = from_total_ms(x, time_total)
    if 'dtime_max' in inputs:
        scale = from_max_ms(x, dtime_max)
    if 'dtime_min' in inputs:
        scale = from_min_ms(x, dtime_min)

    if scale == 0:
        raise Exception("Time scale is 0.")

    return scale

class DurationsScale(object):
    """
    This class builds the x component of the sonification - the durations scale. It populates the namespace of duration objects based on the x scale (x value / time (ms)) input.

    See getScale() function docstring for details on input options avaliable to you.

    Parameters
    ----------
    scale : float
        A time durations scale in units of x value / time (ms).

    """

    def __init__(self, scale):
        self.scale = scale

    def getDurations(self, x):
        """
        Calculates millisecond pitch durations for variable data, based upon deltas between input x values. Note the last deltas will by default equal the dtime_max * scale.

        Parameters
        ----------
        x : arr
            Array of x positions that will correspond to blip times.

        Returns
        -------
        arr
            Durations in ms between time successive blips.

        """
        self.x = np.array(x, dtype='float')
        self.dx = np.subtract(self.x[1:], self.x[0:-1])

        self.dx_min = np.min(self.dx[np.nonzero(self.dx)])
        self.dx_max = np.max(self.dx)

        self.dtime_min = self.dx_min / self.scale
        self.dtime_max = self.dx_max / self.scale
        self.time_total = np.max(self.x) / self.scale

        self.durations = self.dx / self.scale # in ms

        self.starttimes = (self.x- np.min(self.x))/ (self.scale  * s_to_ms) # in s
        # self.starttimes = np.cumsum(np.append([0],self.durations)) / s_to_ms  # in s - alternatively

        return self.starttimes

    def printEdgeCases(self):
        """
        Prints minimum and maximum d(time) values and sound durations to help user understand time and sound min/max changes.
        """

        print ('min/max d(time) values\t', self.dt_min, self.dt_max)
        print ('min/max sound durations (ms)\t', self.dtime_min, self.dtime_max)

    def plotDurations(self, figsize=(4,5), kwargs={ "bins" : 100}):
        """
        Plots histogram of time duration between succesive blips (in ms).

        Parameters
        ----------
        bins : int
            Number of bins within histogram plot.
        args : dict
            Matplotlib histogram input arguments.

        Returns
        -------
        arr
            Matplotlib fig, ax.

        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.hist(self.durations, **kwargs)
        return fig, ax

    def __getitem__(self, item):
        return getattr(self, item)
