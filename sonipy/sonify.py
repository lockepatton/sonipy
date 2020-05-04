from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import thinkdsp
import warnings
from pydub.generators import Sine

from sonipy.scales.frequency import FrequencyScale
from sonipy.scales.durations import DurationsScale, getScale
from sonipy.scales.durations import from_min_ms, from_max_ms, from_total_ms

C4 = 261.6  # Hz
piano_max = 4186.01  # Hz
piano_min = 27.5000  # Hz - not audible

class MultiTone(FrequencyScale):
    """
    Class for building a MultiTone, and saving it as audio files.

    It takes the durations (differences in x values that can be calculated using DurationsScale.getDurations) and y values (values) as inputs to be sonified.

    To account for how y scales with pitch, it takes in a dictionary of frequency_args. See the FrequencyScale object docstring to see details about inputs (frequency_min, frequency_max, cents_per_value, value_min, value_max).

    To account for how x scales with time, it accepts the durations argument.

    For quick and easy use, we recommend use of the SonifyTool wrapper.

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
    verbose : bool
        Flag for printing. Default False.
    alertMultitoneCreated : bool
        Flag for alerting when Multitone created. Default True.
    **kwargs : dictionary
        Dictionary with pitch y value arguments (frequency_min, frequency_max, cents_per_value, value_min, value_max).
    """

    def __init__(self, starttimes, values, length=.5, fade=True,
                 verbose=False, alertMultitoneCreated=True, **kwargs):

        #### Dealing wtih input edge cases.
        # if value_min and value_max are not inputed, set to min and max of values x array
        if "value_min" not in kwargs.keys():
            kwargs["value_min"] = np.min(values)
        if "value_max" not in kwargs.keys():
            kwargs["value_max"] = np.max(values)

        # if value_min and value_max are None, set to min and max of values x array
        if kwargs["value_min"] == None:
            kwargs["value_min"] = np.min(values)
            warnings.warn("value_min equalled None. Replaced with min(values)")
        if kwargs["value_max"] == None:
            kwargs["value_max"] = np.max(values)
            warnings.warn("value_max equalled None. Replaced with max(values)")

        # raise exception if value_min or value_max are not finite.
        if not np.isfinite(kwargs["value_min"]):
            raise Exception("{} value_min input is not finite.".format(kwargs["value_min"]))
        if not np.isfinite(kwargs["value_max"]):
            raise Exception("{} value_max input is not finite.".format(kwargs["value_max"]))

        # Calling upon frequency MultiTone class with kwargs inputs
        super(MultiTone, self).__init__(verbose=verbose, **kwargs)

        self.values = values
        self.starttimes = starttimes
        self.frequencies = self.freq_translate_to_range(self.values)

        if verbose:
            print('frequencies', self.frequencies)
            print('starttimes', self.starttimes)

        if fade:
            def createBlip(f, s):
                """Generates blip with fade."""
                cos_sig = thinkdsp.CosSignal(freq=f, amp=1.0, offset=0)
                wave = cos_sig.make_wave(duration=length, start=s)
                waveyslen = len(wave.ys)
                cut = int(waveyslen / 3)
                scalearray = np.append(np.linspace(
                    0, 1, cut)**.25, np.linspace(1, 0, waveyslen - cut))
                wave.ys = [w * scale for w, scale in zip(wave.ys, scalearray)]
                return wave

        else:
            def createBlip(f, s):
                """Generates blip."""
                cos_sig = thinkdsp.CosSignal(freq=f, amp=1.0, offset=0)
                return cos_sig.make_wave(duration=length, start=s)

        self.multitones = map(createBlip, self.frequencies, self.starttimes)

        self.multitone = sum(self.multitones)

        if alertMultitoneCreated:
            print ('multitones created')

    def play(self):
        # TODO: test this in jupyter notebook
        """
        Wrapper for ThinkDSP .make_audio() function which plays interactively.
        """
        self.multitone.make_audio()

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


def SonifyTool(values, times,
               frequency_args={"frequency_min": C4, "frequency_max": 4 * C4},
               duration_args={"time_total": 2.}, duration_scale=None, length=.1, fade=True,
               alertMultitoneCreated=True, verbose=False):
    """
    This is a built-in-one sonification tool for creating a MultiTone.

    It inputs the x values (times) and y values (values) to be sonified.

    To account for how y scales with pitch, it takes in a dictionary of frequency_args. See the FrequencyScale object docstring to see details about inputs (frequency_min, frequency_max, cents_per_value, value_min, value_max).

    To account for how x scales with time, it accepts a duration_scale in x value / time (ms). If a duration scale is not specified, it pays attention to a dictionary of duration_args. See the DurationsScale object and getScale() function for more info on your avaliable inputs (time_total, time_min, time_max).

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
    fade : bool
        Flag to toggle including fade on each blip. Recommended to include, especially when blips are dense.
    verbose : bool
        Flag for printing. Default False.
    alertMultitoneCreated : bool
        Flag for alerting when Multitone created. Default True.

    Returns
    -------
    object
        Multitone object
    """

    ##### Cleaning Input values and times

    # Using pandas dataframe to test that len(values) = len(times)
    df = pd.DataFrame({"values" : values, "times" : times}, dtype='float')
    # Check for nan's and infitities and remove those pairs of datapoints.
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

    # if cleaning removed datapoints, toss a warning, and if verbose, print out a dataframe of the datapoints removed.
    if len(df_clean) < len(df):
        trouble_datapts_idx = list(set(df.index)-set(df_clean.index))
        warnings.warn("Your input values or times have nan's or infinities. We found {} pair(s), which have been removed. To see which values are used, turn verbose=True.".format(len(trouble_datapts_idx)))

        if verbose:
            print("The following datapoints were removed for having nan's or infinities:")
            print(df.loc[trouble_datapts_idx])

    # Sort values based on x values (times)
    df_clean_sorted = df_clean.sort_values(by=['times'])

    # pull true values and times to be used from cleaned dataframe
    values = np.array(df_clean_sorted["values"])
    times = np.array(df_clean_sorted["times"])

    # Prioritizing time scale (duration) inputs
    if duration_scale != None:
        scale = duration_scale
        if verbose:
            print ("Using duration_scale argument.")
    elif duration_scale == None:
        scale = getScale(times=times, **duration_args)
        if verbose:
            print ("Using duration_args argument.")
    else:
        raise ("Time scale must be defined by duration_scale or within duration_args")

    # calculation starttimes (in seconds) for all tone events
    DurScale = DurationsScale(scale)
    starttimes = DurScale.getDurations(times=times)

    # creating multitone
    Tone = MultiTone(values=values, starttimes=starttimes, length=length, verbose=verbose, alertMultitoneCreated=alertMultitoneCreated, **frequency_args)

    # adding necisssary details to the namespace
    # durations
    Tone.DurationsObj = DurScale
    Tone.duration_scale = scale
    Tone.times = times

    # frequency
    Tone.frequency_args = frequency_args
    Tone.values = values

    return Tone
