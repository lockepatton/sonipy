"""This file contains code used in "Think DSP",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: MIT License (https://opensource.org/licenses/MIT)
"""

import copy
import math

import numpy as np
import random
import scipy
import scipy.stats
import scipy.fftpack
import subprocess
import warnings

from wave import open as open_wave
from scipy.io import wavfile

import matplotlib.pyplot as plt

try:
    from math import gcd
except ImportError:
    from fractions import gcd

try:
    from IPython.display import Audio
except:
    warnings.warn(
        "Can't import Audio from IPython.display; " "Wave.make_audio() will not work."
    )

PI2 = math.pi * 2

def random_seed(x):
    """Initialize the random and np.random generators.

    x: int seed
    """
    random.seed(x)
    np.random.seed(x)


class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""


class WavFileWriter:
    """Writes wav files."""

    def __init__(self, filename="sound.wav", framerate=11025):
        """Opens the file and sets parameters.

        filename: string
        framerate: samples per second
        """
        self.filename = filename
        self.framerate = framerate
        self.nchannels = 1
        self.sampwidth = 2
        self.bits = self.sampwidth * 8
        self.bound = 2 ** (self.bits - 1) - 1

        self.fmt = "h"
        self.dtype = np.uint16

        self.fp = open_wave(self.filename, "w")
        self.fp.setnchannels(self.nchannels)
        self.fp.setsampwidth(self.sampwidth)
        self.fp.setframerate(self.framerate)
#
    def write(self, wave):
        """Writes a wave.

        wave: Wave
        """
        zs = wave.quantize(self.bound, self.dtype)
        self.fp.writeframes(zs.tostring())

    def close(self, duration=0):
        """Closes the file.

        duration: how many seconds of silence to append
        """
        if duration:
            self.write(rest(duration))

        self.fp.close()

def play_wave(filename="sound.wav", player="aplay"):
    """Plays a wave file.

    filename: string
    player: string name of executable that plays wav files
    """
    cmd = "%s %s" % (player, filename)
    popen = subprocess.Popen(cmd, shell=True)
    popen.communicate()


def find_index(x, xs):
    """Find the index corresponding to a given value in an array."""
    n = len(xs)
    start = xs[0]
    end = xs[-1]
    i = round((n - 1) * (x - start) / (end - start))
    return int(i)


class _SpectrumParent:
    """Contains code common to Spectrum and DCT.
    """

    def __init__(self, hs, fs, framerate, full=False):
        """Initializes a spectrum.

        hs: array of amplitudes (real or complex)
        fs: array of frequencies
        framerate: frames per second
        full: boolean to indicate full or real FFT
        """
        self.hs = np.asanyarray(hs)
        self.fs = np.asanyarray(fs)
        self.framerate = framerate
        self.full = full

    @property
    def max_freq(self):
        """Returns the Nyquist frequency for this spectrum."""
        return self.framerate / 2

    @property
    def amps(self):
        """Returns a sequence of amplitudes (read-only property)."""
        return np.absolute(self.hs)

    @property
    def power(self):
        """Returns a sequence of powers (read-only property)."""
        return self.amps ** 2

    def copy(self):
        """Makes a copy.

        Returns: new Spectrum
        """
        return copy.deepcopy(self)
    def plot(self, high=None, **options):
        """Plots amplitude vs frequency.

        Note: if this is a full spectrum, it ignores low and high

        high: frequency to cut off at
        """
        if self.full:
            fs, amps = self.render_full(high)
            plt.plot(fs, amps, **options)
        else:
            i = None if high is None else find_index(high, self.fs)
            plt.plot(self.fs[:i], self.amps[:i], **options)

    def plot_power(self, high=None, **options):
        """Plots power vs frequency.

        high: frequency to cut off at
        """
        if self.full:
            fs, amps = self.render_full(high)
            plt.plot(fs, amps ** 2, **options)
        else:
            i = None if high is None else find_index(high, self.fs)
            plt.plot(self.fs[:i], self.power[:i], **options)

class Spectrum(_SpectrumParent):
    """Represents the spectrum of a signal."""

    def __len__(self):
        """Length of the spectrum."""
        return len(self.hs)

    def __add__(self, other):
        """Adds two spectrums elementwise.

        other: Spectrum

        returns: new Spectrum
        """
        if other == 0:
            return self.copy()

        assert all(self.fs == other.fs)
        hs = self.hs + other.hs
        return Spectrum(hs, self.fs, self.framerate, self.full)

    __radd__ = __add__

    def __mul__(self, other):
        """Multiplies two spectrums elementwise.

        other: Spectrum

        returns: new Spectrum
        """
        assert all(self.fs == other.fs)
        hs = self.hs * other.hs
        return Spectrum(hs, self.fs, self.framerate, self.full)

    def make_wave(self):
        """Transforms to the time domain.

        returns: Wave
        """
        if self.full:
            ys = np.fft.ifft(self.hs)
        else:
            ys = np.fft.irfft(self.hs)

        # NOTE: whatever the start time was, we lose it when
        # we transform back; we could fix that by saving start
        # time in the Spectrum
        # ts = self.start + np.arange(len(ys)) / self.framerate
        return Wave(ys, framerate=self.framerate)

class Spectrogram:
    """Represents the spectrum of a signal."""

    def __init__(self, spec_map, seg_length):
        """Initialize the spectrogram.

        spec_map: map from float time to Spectrum
        seg_length: number of samples in each segment
        """
        self.spec_map = spec_map
        self.seg_length = seg_length

    def any_spectrum(self):
        """Returns an arbitrary spectrum from the spectrogram."""
        index = next(iter(self.spec_map))
        return self.spec_map[index]

    @property
    def time_res(self):
        """Time resolution in seconds."""
        spectrum = self.any_spectrum()
        return float(self.seg_length) / spectrum.framerate

    @property
    def freq_res(self):
        """Frequency resolution in Hz."""
        return self.any_spectrum().freq_res

    def times(self):
        """Sorted sequence of times.

        returns: sequence of float times in seconds
        """
        ts = sorted(iter(self.spec_map))
        return ts

    def frequencies(self):
        """Sequence of frequencies.

        returns: sequence of float freqencies in Hz.
        """
        fs = self.any_spectrum().fs
        return fs

    def plot(self, high=None, **options):
        """Make a pseudocolor plot.

        high: highest frequency component to plot
        """
        fs = self.frequencies()
        i = None if high is None else find_index(high, fs)
        fs = fs[:i]
        ts = self.times()

        # make the array
        size = len(fs), len(ts)
        array = np.zeros(size, dtype=float)

        # copy amplitude from each spectrum into a column of the array
        for j, t in enumerate(ts):
            spectrum = self.spec_map[t]
            array[:, j] = spectrum.amps[:i]

        underride(options, cmap='inferno_r')
        plt.pcolor(ts, fs, array, **options)

    def get_data(self, high=None, **options):
        """Returns spectogram as 2D numpy array

        high: highest frequency component to return
        """
        fs = self.frequencies()
        i = None if high is None else find_index(high, fs)
        fs = fs[:i]
        ts = self.times()

        # make the array
        size = len(fs), len(ts)
        array = np.zeros(size, dtype=float)

        # copy amplitude from each spectrum into a column of the array
        for j, t in enumerate(ts):
            spectrum = self.spec_map[t]
            array[:, j] = spectrum.amps[:i]

        return array

    def make_wave(self):
        """Inverts the spectrogram and returns a Wave.

        returns: Wave
        """
        res = []
        for t, spectrum in sorted(self.spec_map.items()):
            wave = spectrum.make_wave()
            n = len(wave)

            window = 1 / np.hamming(n)
            wave.window(window)

            i = wave.find_index(t)
            start = i - n // 2
            end = start + n
            res.append((start, end, wave))

        starts, ends, waves = zip(*res)
        low = min(starts)
        high = max(ends)

        ys = np.zeros(high - low, float)
        for start, end, wave in res:
            ys[start:end] = wave.ys

        # ts = np.arange(len(ys)) / self.framerate
        return Wave(ys, framerate=wave.framerate)


class Wave:
    """Represents a discrete-time waveform.

    """

    def __init__(self, ys, ts=None, framerate=None):
        """Initializes the wave.

        ys: wave array
        ts: array of times
        framerate: samples per second
        """
        self.ys = np.asanyarray(ys)
        self.framerate = framerate if framerate is not None else 11025

        if ts is None:
            self.ts = np.arange(len(ys)) / self.framerate
        else:
            self.ts = np.asanyarray(ts)

    def copy(self):
        """Makes a copy.

        Returns: new Wave
        """
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.ys)

    @property
    def start(self):
        return self.ts[0]

    @property
    def end(self):
        return self.ts[-1]

    @property
    def duration(self):
        """Duration (property).

        returns: float duration in seconds
        """
        return len(self.ys) / self.framerate

    def __add__(self, other):
        """Adds two waves elementwise.

        other: Wave

        returns: new Wave
        """
        if other == 0:
            return self

        assert self.framerate == other.framerate

        # make an array of times that covers both waves
        start = min(self.start, other.start)
        end = max(self.end, other.end)
        n = int(round((end - start) * self.framerate)) + 1
        ys = np.zeros(n)
        ts = start + np.arange(n) / self.framerate

        def add_ys(wave):
            i = find_index(wave.start, ts)

            # make sure the arrays line up reasonably well
            diff = ts[i] - wave.start
            dt = 1 / wave.framerate
            if (diff / dt) > 0.1:
                warnings.warn(
                    "Can't add these waveforms; their " "time arrays don't line up."
                )

            j = i + len(wave)
            ys[i:j] += wave.ys

        add_ys(self)
        add_ys(other)

        return Wave(ys, ts, self.framerate)

    __radd__ = __add__

    def quantize(self, bound, dtype):
        """Maps the waveform to quanta.

        bound: maximum amplitude
        dtype: numpy data type or string

        returns: quantized signal
        """
        return quantize(self.ys, bound, dtype)

    def window(self, window):
        """Apply a window to the wave.

        window: sequence of multipliers, same length as self.ys
        """
        self.ys *= window

    def normalize(self, amp=1.0):
        """Normalizes the signal to the given amplitude.

        amp: float amplitude
        """
        self.ys = normalize(self.ys, amp=amp)

    def slice(self, i, j):
        """Makes a slice from a Wave.

        i: first slice index
        j: second slice index
        """
        ys = self.ys[i:j].copy()
        ts = self.ts[i:j].copy()
        return Wave(ys, ts, self.framerate)
    #
    def make_spectrum(self, full=False):
        """Computes the spectrum using FFT.

        full: boolean, whethere to compute a full FFT
              (as opposed to a real FFT)

        returns: Spectrum
        """
        n = len(self.ys)
        d = 1 / self.framerate

        if full:
            hs = np.fft.fft(self.ys)
            fs = np.fft.fftfreq(n, d)
        else:
            hs = np.fft.rfft(self.ys)
            fs = np.fft.rfftfreq(n, d)

        return Spectrum(hs, fs, self.framerate, full)

    def make_spectrogram(self, seg_length, win_flag=True):
        """Computes the spectrogram of the wave.

        seg_length: number of samples in each segment
        win_flag: boolean, whether to apply hamming window to each segment

        returns: Spectrogram
        """
        if win_flag:
            window = np.hamming(seg_length)
        i, j = 0, seg_length
        step = int(seg_length // 2)

        # map from time to Spectrum
        spec_map = {}

        while j < len(self.ys):
            segment = self.slice(i, j)
            if win_flag:
                segment.window(window)

            # the nominal time for this segment is the midpoint
            t = (segment.start + segment.end) / 2
            spec_map[t] = segment.make_spectrum()

            i += step
            j += step

        return Spectrogram(spec_map, seg_length)

    def get_xfactor(self, options):
        try:
            xfactor = options["xfactor"]
            options.pop("xfactor")
        except KeyError:
            xfactor = 1
        return xfactor
    #
    def plot(self, **options):
        """Plots the wave.

        If the ys are complex, plots the real part.

        """
        xfactor = self.get_xfactor(options)
        plt.plot(self.ts * xfactor, np.real(self.ys), **options)

    def plot_vlines(self, **options):
        """Plots the wave with vertical lines for samples.

        """
        xfactor = self.get_xfactor(options)
        plt.vlines(self.ts * xfactor, 0, self.ys, **options)

    def write(self, filename="sound.wav"):
        """Write a wave file.

        filename: string
        """
        print("Writing", filename)
        wfile = WavFileWriter(filename, self.framerate)
        wfile.write(self)
        wfile.close()

    def play(self, filename="sound.wav"):
        """Plays a wave file.

        filename: string
        """
        self.write(filename)
        play_wave(filename)

    def make_audio(self):
        """Makes an IPython Audio object.
        """
        audio = Audio(data=self.ys.real, rate=self.framerate)
        return audio

def normalize(ys, amp=1.0):
    """Normalizes a wave array so the maximum amplitude is +amp or -amp.

    ys: wave array
    amp: max amplitude (pos or neg) in result

    returns: wave array
    """
    high, low = abs(max(ys)), abs(min(ys))
    return amp * ys / max(high, low)


def quantize(ys, bound, dtype):
    """Maps the waveform to quanta.

    ys: wave array
    bound: maximum amplitude
    dtype: numpy data type of the result

    returns: quantized signal
    """
    if max(ys) > 1 or min(ys) < -1:
        warnings.warn("Warning: normalizing before quantizing.")
        ys = normalize(ys)

    zs = (ys * bound).astype(dtype)
    return zs


class Signal:
    """Represents a time-varying signal."""

    def __add__(self, other):
        """Adds two signals.

        other: Signal

        returns: Signal
        """
        if other == 0:
            return self
        return SumSignal(self, other)

    __radd__ = __add__

    @property
    def period(self):
        """Period of the signal in seconds (property).

        Since this is used primarily for purposes of plotting,
        the default behavior is to return a value, 0.1 seconds,
        that is reasonable for many signals.

        returns: float seconds
        """
        return 0.1

    def plot(self, framerate=11025):
        """Plots the signal.

        The default behavior is to plot three periods.

        framerate: samples per second
        """
        duration = self.period * 3
        wave = self.make_wave(duration, start=0, framerate=framerate)
        wave.plot()

    def make_wave(self, duration=1, start=0, framerate=11025):
        """Makes a Wave object.

        duration: float seconds
        start: float seconds
        framerate: int frames per second

        returns: Wave
        """
        n = round(duration * framerate)
        ts = start + np.arange(n) / framerate
        ys = self.evaluate(ts)
        return Wave(ys, ts, framerate=framerate)

class Sinusoid(Signal):
    """Represents a sinusoidal signal."""

    def __init__(self, freq=440, amp=1.0, offset=0, func=np.sin):
        """Initializes a sinusoidal signal.

        freq: float frequency in Hz
        amp: float amplitude, 1.0 is nominal max
        offset: float phase offset in radians
        func: function that maps phase to amplitude
        """
        self.freq = freq
        self.amp = amp
        self.offset = offset
        self.func = func

    @property
    def period(self):
        """Period of the signal in seconds.

        returns: float seconds
        """
        return 1.0 / self.freq

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        phases = PI2 * self.freq * ts + self.offset
        ys = self.amp * self.func(phases)
        return ys


def CosSignal(freq=440, amp=1.0, offset=0):
    """Makes a cosine Sinusoid.

    freq: float frequency in Hz
    amp: float amplitude, 1.0 is nominal max
    offset: float phase offset in radians

    returns: Sinusoid object
    """
    return Sinusoid(freq, amp, offset, func=np.cos)


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    If d is None, create a new dictionary.

    d: dictionary
    options: keyword args to add to d
    """
    if d is None:
        d = {}

    for key, val in options.items():
        d.setdefault(key, val)

    return d
