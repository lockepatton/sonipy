from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

s_to_ms = 1000.

def from_max_ms(times, time_max):
    """
    Calculates a time scale in units of x value / time (ms), by matching the largest time step in the times array to the input time_max.

    Parameters
    ----------
    times : arr
        Array of x positions that will correspond to blip times.
    time_max : float
        Maximum allowed time difference between blips (in ms).

    Returns
    -------
    float
        Returns a time scale calculated from the maximum dt.

    """
    dt = np.subtract(times[1:], times[0:-1])
    dt_max = max(dt)
    scale = float(dt_max) / float(time_max)  # x value / ms
    return scale


def from_min_ms(times, time_min):
    """
    Calculates a time scale in units of x value / time (ms), by matching the smallest time step in the times array to the input time_min.

    Parameters
    ----------
    times : arr
        Array of x positions that will correspond to blip times.
    time_min : float
        Minimum allowed time difference between blips (in ms).

    Returns
    -------
    float
        Returns a time scale calculated from the minimum dt.

    """
    dt = np.subtract(times[1:], times[0:-1])
    dt_min = np.min(dt[np.nonzero(dt)])  # min(dt)
    scale = float(dt_min) / float(time_min)  # x value / ms
    return scale


def from_total_ms(times, time_total):
    """
    Calculates a time scale in units of x value / time (ms), by matching the total length of times array to the input time_total.

    Parameters
    ----------
    times : arr
        Array of x positions that will correspond to blip times.
    time_total : float
        Total time difference between first and last blip (in ms).

    Returns
    -------
    float
        Returns a time scale calculated from the total dt.

    """
    dt_total = abs(times[0] - times[-1])
    scale = float(dt_total) / float(time_total) # x value / ms
    return scale


def getScale(times, time_total=None, time_min=None, time_max=None):
    """
    Function that takes in a total time between blips, minimum difference between successive blips or maximum difference between successive blips and returns a time scale in units of x value / ms.
    Note, if multiple inputs are specific, the function first tries total_time, then time_max, then time_min to create the scale.


    Parameters
    ----------
    times : arr
        Array of x positions that will correspond to blip times.
    time_total : float
        Total time difference between first and last blip (in ms).
    time_min : float
        Minimum allowed time difference between blips (in ms).
    time_max : float
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
    if time_max != None:
        inputs.append('time_max')
    if time_min != None:
        inputs.append('time_min')
    n_inputs = len(inputs)

    # raising exception if anything other than one input were given
    if n_inputs != 1:
        raise Exception('User needs to define time_total, time_max, or time_min to find time scale. You inputted {} inputs, which were {}.'.format(
            n_inputs, inputs))

    if 'time_total' in inputs:
        scale = from_total_ms(times, time_total) # GO BACK HERE
    if 'time_max' in inputs:
        scale = from_max_ms(times, time_max)
    if 'time_min' in inputs:
        scale = from_min_ms(times, time_min)

    if scale == 0:
        raise Exception("Time scale is 0.")

    return scale

class DurationsScale(object):
    """
    This class builds the x component of the sonification - the durations scale. It populates the namespace of duration objects based on the x scale (x value / time (ms)) input.
    This includes the

    See getScale() function docstring for details on input options avaliable to you.

    Parameters
    ----------
    scale : float
        A time durations scale in units of x value / time (ms).

    """

    def __init__(self, scale):
        self.scale = scale

    def getDurations(self, times):
        """
        Calculates millisecond pitch durations for variable data, based upon deltas between input x values. Note the last deltas will by default equal the time_max * scale.

        Parameters
        ----------
        times : arr
            Array of x positions that will correspond to blip times.

        Returns
        -------
        arr
            Durations in ms between time successive blips.

        """
        self.times = np.array(times, dtype='float')
        self.dt = np.subtract(self.times[1:], self.times[0:-1])

        self.dt_min = np.min(self.dt[np.nonzero(self.dt)])
        self.dt_max = np.max(self.dt)

        self.time_min = self.dt_min / self.scale
        self.time_max = self.dt_max / self.scale

        self.durations = self.dt / self.scale # in ms

        self.starttimes = (self.times- np.min(self.times))/ (self.scale  * s_to_ms) # in s
        # self.starttimes = np.cumsum(np.append([0],self.durations)) / s_to_ms  # in s - alternatively

        return self.starttimes

    def printEdgeCases(self):
        """
        Prints minimum and maximum d(time) values and sound durations to help user understand time and sound min/max changes.
        """

        print ('min/max d(time) values\t', self.dt_min, self.dt_max)
        print ('min/max sound durations (ms)\t', self.time_min, self.time_max)

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
