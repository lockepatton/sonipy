C4 = 261.6  # Hz
piano_max = 4186.01  # Hz
piano_min = 27.5000  # Hz - not audible

import warnings
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import os
import thinkdsp
from pydub.generators import Sine
from scipy.interpolate import interp1d

def cent_per_value(f_min, f_max, v_min, v_max):
    """
    This function takes in a frequency max and min, and y value max and min and returns a y scale parameter in units of cents/y value.
    Cents are a logarithmic unit of tone intervals (https://en.wikipedia.org/wiki/Cent_(music)).

    Parameters
    ----------
    f_min : float
        Minimum frequency.
    f_max : float
        Maximum frequency.
    v_min : float
        Minimum y value.
    v_max : float
        Maximum y value.

    Returns
    -------
    float
        A y-scale parameter in units of cents/y value.

    """
    step = 1200 * np.log2(f_max / f_min)  / (v_max - v_min)
    return step

def get_f_min(f_max, cents_per_value, v_min, v_max):
    """
    This function takes in a y value max and min, a maximum frequency and a y scale parameter in units of cents/y value, and returns the minimum frequency that fits to such a scale.
    Cents are a logarithmic unit of tone intervals (https://en.wikipedia.org/wiki/Cent_(music)).

    Parameters
    ----------
    f_max : float
        Maximum frequency.
    cents_per_value : float
        A y scale parameter in units of cents/y value.
    v_min : float
        Minimum y value.
    v_max : float
        Maximum y value.

    Returns
    -------
    float
        Minimum frequency.

    """
    f_min = f_max / ( 2 ** ((v_max - v_min) * cents_per_value / 1200))
    return f_min

class FrequencyScale(object):
    """
    This class builds a frequency scale and populates the namespace of frequency objects based on the given inputs from the following combos:
        - frequency min, frequency max, y value min and y value max
        - frequency_max, cents_per_value, y value min and y value max
    Both of these options will match the maximum frequency to the maximum value.
    Cents are a logarithmic unit of tone intervals (https://en.wikipedia.org/wiki/Cent_(music)).

    Parameters
    ----------
    frequency_min : float
        Minimum frequency.
    frequency_max : float
        Maximum frequency.
    cents_per_value : float
        A y scale parameter in units of cents/y value.
    value_min : float
        Description of parameter `value_min`.
    value_max : float
        Description of parameter `value_max`.
    verbose : bool
        Flag to toggle printing functions.

    """


    def __init__(self, frequency_min=None, frequency_max=C4 * 4, cents_per_value=None,
                 value_min=0, value_max=1, verbose=False):

        if verbose:
            print('initial vals (fmin, fmax, vmin, vmax):',frequency_min, frequency_max, value_min, value_max)

        # TODO: Build check to alert user which parameters are truly being used if too many are inputted.

        # defining cents_per_value
        if (cents_per_value == None):
            self.cents_per_value = cent_per_value(frequency_min, frequency_max,
                                                  value_min, value_max)
        else:
            self.cents_per_value = cents_per_value

        if verbose:
            print('self.cents_per_value',self.cents_per_value)

        # defining frequency_min
        self.value_min = value_min
        self.value_max = value_max
        self.frequency_max = frequency_max
        self.frequency_min = get_f_min(self.frequency_max, self.cents_per_value, self.value_min, self.value_max)


        if verbose:
            print('initial vals:',self.frequency_min,self.frequency_max, self.value_min, self.value_max)

        freq = lambda v : self.frequency_min * 2 ** ((v - self.value_min) * self.cents_per_value / 1200)
        self.freq_translate_to_range = lambda array : list(map(freq, array))

        if verbose:
            print('Frequency Scale Built')


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
    scale = dt_max / time_max  # MJD / ms
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
    scale = dt_min / time_min  # MJD / ms
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
    scale = dt_total / time_total  # MJD / ms
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
    if time_total is not None:
        scale = from_total_ms(times, time_total)
        time_total = time_total
    if time_max is not None:
        scale = from_max_ms(times, time_max)
        time_max = time_max
    if time_min is not None:
        scale = from_min_ms(times, time_min)
        time_min = time_min
    if scale==0:
        raise ("Time scale is 0.")
    try:
        return scale
    except:
        raise "error: need to define time_total, time_max, or time_min to find time scale"


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
        self.times = times
        self.dt = np.subtract(self.times[1:], self.times[0:-1])
        self.dt_min, self.dt_max = np.min(self.dt[np.nonzero(self.dt)]), max(self.dt)

        self.time_min, self.time_max = self.dt_min * self.scale, self.dt_max * self.scale

        # last signal = longest signal time
        self.durations = np.append(self.dt / self.scale, [self.time_max * self.scale])

        # m = interp1d([self.dt_min, self.dt_max], [self.time_min, self.time_max])
        # self.durations = np.append(m(self.dt), [self.time_max]) #last signal = longest signal time
        return self.durations

    def printEdgeCases(self):
        """
        Prints edge cases to help user understand time and sound min/max changes
        """

        print ('min/max d(time) values\t', self.dt_min, self.dt_max)
        print ('min/max sound durations (ms)\t',self.time_min, self.time_max)

    def plotDurations(self, bins=100, kwargs={}):
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
        fig, ax = plt.subplots(1, 1)
        ax.hist(self.durations, bins=bins, **kwargs)
        fig.show()
        return fig, ax


class StringTone(FrequencyScale):
    """
    Class for building string tones and saving their audio files. This is kept for the unusual use case where one wants the tone to sit at the given pitch and restart at a new pitch for each new datapoint while scanning left to right.

    Parameters
    ----------
    durations : arr
        Durations in ms between time successive blips.
    values : arr
        Array of y positions that will correspond to pitch values.
    fadetime : float
        Total fading in and out time, split equally between fade in time and fade out time (ms).
    **kwargs : dictionary
        Dictionary with pitch y value arguments (frequency_min, frequency_max, cents_per_value, value_min, value_max).

    """

    def __init__(self, durations, values, fadetime=100, **kwargs):
        super(StringTone, self).__init__(**kwargs)

        self.values = values
        self.durations = durations
        self.frequencies = self.freq_translate_to_range(self.values)
        # print('frequencies',self.frequencies)

        def createSineTones(f, d):
            return Sine(f).to_audio_segment(duration=d).fade_in(fadetime / 2).fade_out(fadetime / 2)

        self.stringtones = map(createSineTones, self.frequencies, self.durations)
        self.stringtone = sum(self.stringtones)
        print ('stringtone created')

    def SaveTone(self, path='.', filename='stringtone.mp4'):
        """
        Saves .mp4 file from stringtone system.

        Parameters
        ----------
        path : str
            desired output path for "tones" folder
        filename : str
            File name. Include .mp4 or other media format.
        """

        self.filepath = os.path.join(path, 'tones/')

        if os.path.exists(path):
            if not os.path.exists(self.filepath):
                os.makedirs(self.filepath)

            fullfilepath = os.path.join(self.filepath, filename)
            self.stringtone.export(fullfilepath, format="mp4")
            print('Saved stringtone as %s.' % fullfilepath)
        else:
            raise('path does not exist.')


class MultiTone(FrequencyScale):
    """
    Class for building a MultiTone, and saving it as audio files.

    It inputs the durations and y values (values) to be sonified. To account for how y scales with pitch, it takes in a dictionary of frequency_args. See the FrequencyScale object docstring to see details about inputs (frequency_min, frequency_max, cents_per_value, value_min, value_max).

    To account for how x scales with time, it accepts the durations argument.

    For quick and easy use, use the SonifyTool.

    Parameters
    ----------
    durations : arr
        Durations in ms between time successive blips.
    values : arr
        Array of y positions that will correspond to pitch values.
    length : float
        Duration of each blip in seconds.
    fade : bool
        Flag to toggle including fade on each blip. Recommended to include, especially when blips are dense.
    **kwargs : dictionary
        Dictionary with pitch y value arguments (frequency_min, frequency_max, cents_per_value, value_min, value_max).
    """

    def __init__(self, durations, values, length=.5, fade=True, **kwargs):
        super(MultiTone, self).__init__(**kwargs)

        def runningSum(durations):
            return np.cumsum(durations)

        self.values = values
        self.durations = durations
        self.starttimes = runningSum(self.durations) / 1000

        self.frequencies = self.freq_translate_to_range(self.values)
        # print('frequencies',self.frequencies)
        # print('starttimes',self.starttimes)


        if fade:
            def createBlip(f, s):
                """Generates blip with fade."""
                cos_sig = thinkdsp.CosSignal(freq=f, amp=1.0, offset=0)
                wave = cos_sig.make_wave(duration=length, start=s)
                waveyslen = len(wave.ys)
                cut = int(waveyslen/3)
                scalearray = np.append(np.linspace(0,1,cut)**.25, np.linspace(1,0,waveyslen-cut))
                wave.ys = [w * scale for w, scale in zip(wave.ys,scalearray)]
                return wave

        else:
            def createBlip(f, s):
                """Generates blip."""
                cos_sig = thinkdsp.CosSignal(freq=f, amp=1.0, offset=0)
                return cos_sig.make_wave(duration=length, start=s)

        self.multitones = map(createBlip, self.frequencies, self.starttimes)
        self.multitone = sum(self.multitones)
        print ('multitones created')

    def SaveTone(self, path='.', filename='multitone.wav'):
        """
        Saves .wav file from multitone system.
        """

        self.filepath = os.path.join(path, 'tones/')

        if os.path.exists(path):
            if not os.path.exists(self.filepath):
                os.makedirs(self.filepath)

            fullfilepath = os.path.join(self.filepath, filename)
            self.multitone.write(fullfilepath)
            print('Saved multitone as %s.' % fullfilepath)
        else:
            raise('Path does not exist.')

def SonifyTool(values, times, frequency_args={"frequency_min" : C4, "frequency_max" : 4*C4 }, duration_args={"time_total" : 2}, duration_scale=None, length=.1):
    """
    This is a built-in-one sonification tool for creating a MultiTone.

    It inputs the x values (times) and y values (values) to be sonified.

    To account for how y scales with pitch, it takes in a dictionary of frequency_args. See the FrequencyScale object docstring to see details about inputs (frequency_min, frequency_max, cents_per_value, value_min, value_max).

    To account for how x scales with time, it accepts a duration_scale in x value / time (ms). If a duration scale is not specified, it takes in a dictionary of duration_args. See the DurationsScale object and getScale() function for more info on your avaliable inputs (time_total, time_min, time_max).

    Finally, you must specify the length you'd like each blip to last in seconds in the parameter length. Default is .1 seconds.

    Parameters
    ----------
    values : arr
        Array of y positions that will correspond to pitch values.
    times : arr
        Array of x positions that will correspond to blip times.
    frequency_args : dictionary
        Dictionary with pitch y value arguments (frequency_min, frequency_max, cents_per_value, value_min, value_max). Defaults to min of C4 and max of four times C4.
    duration_args : dictionary
        Dictionary with duration x value arguments  (time_total, time_min, or time_max).
    duration_scale : float
        Scale argument for time axis (x value / time (ms)), as an alternate to duration_args input.
    length : float
        Duration of each blip in seconds.

    Returns
    -------
    object
        Multitone object.

    """

    values, times = np.array(values), np.array(times)

    # determining time scale
    if duration_scale != None:
        scale = duration_scale
    elif duration_scale == None:
        scale = getScale(times=times, **duration_args)
    else:
        raise "Time scale must be defined by duration_scale or within duration_args"

    # calculation Durations between tone events
    DurScale = DurationsScale(scale)
    durations = DurScale.getDurations(times=times)

    # Frequency Default Arguments
    # If value_min or value_max unspecified, defaulting to max/min of 'values'
    if 'value_min' not in frequency_args:
        frequency_args['value_min'] = min(values)
    if 'value_max' not in frequency_args:
        frequency_args['value_max'] = max(values)

    # creating multitone
    Tone = MultiTone(values=values, durations=durations, length=length, **frequency_args)

    return Tone
