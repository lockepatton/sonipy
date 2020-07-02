from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sonipy.thinkdsp as thinkdsp
import warnings

from sonipy.scales.frequency import FrequencyScale
from sonipy.scales.durations import DurationsScale, getScale
from sonipy.scales.durations import from_min_ms, from_max_ms, from_total_ms

try:
    from IPython.display import Audio
except:
    warnings.warn("Can't import Audio from IPython.display; " "Tone.play() will not work." "Use Tone.SaveTone() to save and listen instead.")

C4 = 261.6  # Hz
piano_max = 4186.01  # Hz
piano_min = 27.5000  # Hz - not audible
s_to_ms = 1000.


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

    def __init__(self, starttimes, values, bliplength=.5, fade=True,
                 verbose=False, alertMultitoneCreated=True, **kwargs):

        if len(starttimes) != len(values):
            raise Exception("startimes and values are not the same length.")

        # Dealing wtih input edge cases.
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
            raise Exception(
                "{} value_min input is not finite.".format(kwargs["value_min"]))
        if not np.isfinite(kwargs["value_max"]):
            raise Exception(
                "{} value_max input is not finite.".format(kwargs["value_max"]))

        # Calling upon frequency MultiTone class with kwargs inputs
        super(MultiTone, self).__init__(verbose=verbose, **kwargs)

        self.y_values = values
        self.x_starttimes = starttimes
        self.y_frequencies = self.y_freq_translate_to_range(self.y_values)

        if verbose:
            print('frequencies', self.y_frequencies)
            print('starttimes', self.x_starttimes)

        if fade:
            def createBlip(f, s):
                """Generates blip with fade."""
                cos_sig = thinkdsp.CosSignal(freq=f, amp=1.0, offset=0)
                wave = cos_sig.make_wave(duration=bliplength, start=s)
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
                return cos_sig.make_wave(duration=bliplength, start=s)

        self.bliplength = bliplength
        self.multitones = map(
            createBlip, self.y_frequencies, self.x_starttimes)

        self.multitone = sum(self.multitones)
        self.multitone.normalize()

        if alertMultitoneCreated:
            print ('multitones created')

    def play(self, autoplay=True):
        """
        Function to play tones interactively in jupyter notebooks, based on ThinkDSP .make_audio() function.

        Parameters
        ----------
        autoplay : bool
            Flag on whether to autoplay file.
        """
        audio = Audio(data=self.multitone.ys.real,
                      rate=self.multitone.framerate, autoplay=autoplay)
        return audio

    def save(self, path='.', filename='multitone.wav', default_tones_folder='tones/'):
        """
        Saves .wav file from multitone system in tones folder.
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

    def plotBlips(self,):
        """
        Plots time vs. amplitude for all individual blip soundbites.
        """
        for blip in self.multitones:
            blip.plot()
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

    def plotTone(self,):
        """
        Plots time vs. amplitude for total soundfile.
        """
        self.multitone.plot()
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

    def plotSpectrogram(self, seg_length=1024, win_flag=True, set_xlim=True):
        """
        Plots a spectrogram of the Tone.

        Parameters
        ----------
        seg_length : int
            number of samples in each segment
        win_flag : bool
            Boolian flag to apply hamming window to each segment
        set_xlim : Bool
            Boolian flag to adjust xlimits between 0 and total time.
        """

        sp = self.multitone.make_spectrogram(seg_length, win_flag=win_flag)
        sp.plot(high=self.y_frequency_max * 1.1)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        if set_xlim:
            plt.xlim(0, self.x_time_total / 1000.)

    def plotSpectrum(self):
        """
        Plots a spectrum of the Tone.
        """
        spectrum = self.multitone.make_spectrum()
        spectrum.plot()


def SonifyTool(x, y,
               frequency_args={"frequency_min": C4, "frequency_max": 4 * C4},
               duration_args={"time_total": 2. * s_to_ms}, duration_scale=None, bliplength=.1, fade=True,
               alertMultitoneCreated=True, verbose=False):
    """
    This is a built-in-one sonification tool for creating a MultiTone.

    It inputs the x values (x) and y values (y) to be sonified.

    To account for how y scales with pitch, it takes in a dictionary of frequency_args. See the FrequencyScale object docstring to see details about inputs (frequency_min, frequency_max, cents_per_value, value_min, value_max).

    To account for how x scales with time, it accepts a duration_scale in x value / time (ms). If a duration scale is not specified, it pays attention to a dictionary of duration_args. See the DurationsScale object and getScale() function for more info on your avaliable inputs (time_total, time_min, time_max).

    Finally, you must specify the length you'd like each blip to last in seconds in the parameter bliplength. Default is .1 seconds.

    Parameters
    ----------
    y : arr
        Array of y positions that will correspond to pitch values.
    x : arr
        Array of x positions that will correspond to blip times.
    frequency_args : dictionary
        Dictionary with pitch y value arguments (frequency_min, frequency_max, cents_per_value, value_min, value_max). Defaults to min of C4 and max of four times C4.
    duration_args : dictionary
        Dictionary with duration x value arguments  (time_total, time_min, or time_max).
    duration_scale : float
        Scale argument for time axis (x value / time (ms)), as an alternate to duration_args input.
    bliplength : float
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

    # Cleaning Input x and y values

    # Using pandas dataframe to test that len(y) = len(x)
    df = pd.DataFrame({"x": x, "y": y}, dtype='float')
    # Check for nan's and infitities and remove those pairs of datapoints.
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

    # if cleaning removed datapoints, toss a warning, and if verbose, print out a dataframe of the datapoints removed.
    if len(df_clean) < len(df):
        trouble_datapts_idx = list(set(df.index) - set(df_clean.index))
        warnings.warn("Your input x or y arrays have nan's or infinities. We found {} pair(s), which have been removed. To see which values are used, turn verbose=True.".format(
            len(trouble_datapts_idx)))

        if verbose:
            print("The following datapoints were removed for having nan's or infinities:")
            print(df.loc[trouble_datapts_idx])

    # Sort values based on x values
    df_clean_sorted = df_clean.sort_values(by=['x'])

    # pull true values and x to be used from cleaned dataframe
    y = np.array(df_clean_sorted["y"])
    x = np.array(df_clean_sorted["x"])

    # Time scale (duration) inputs
    using_duration_scale_argument = duration_scale != None

    # Prioritizing time scale (duration) inputs
    if using_duration_scale_argument:
        scale = duration_scale

        if verbose:
            print ("Using duration_scale argument.")

    elif not using_duration_scale_argument:
        scale = getScale(x=x, **duration_args)

        if verbose:
            print ("Using duration_args argument.")
    else:
        raise ("Time scale must be defined by duration_scale or within duration_args")

    # calculation starttimes (in seconds) for all tone events
    DurScale = DurationsScale(scale)
    starttimes = DurScale.getDurations(x=x)

    # creating multitone
    Tone = MultiTone(values=y, starttimes=starttimes, bliplength=bliplength, verbose=verbose,
                     alertMultitoneCreated=alertMultitoneCreated, **frequency_args)

    # Passing necisssary details to the Tone namespace
    # durations
    keys_to_pass = DurScale.__dict__.keys()
    for key in DurScale.__dict__.keys():
        setattr(Tone, 'x_' + key, DurScale[key])

    Tone.x = x
    del Tone.x_x

    # frequency
    Tone.y = y
    Tone.y_input_frequency_args = frequency_args

    Tone.keys = sorted(Tone.__dict__.keys())

    return Tone
