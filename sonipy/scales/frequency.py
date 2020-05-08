from __future__ import print_function
import warnings
import numpy as np

C4 = 261.6  # Hz
piano_max = 4186.01  # Hz
piano_min = 27.5000  # Hz - not audible

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
    step = 1200 * np.log2(f_max / f_min) / (v_max - v_min)
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
    f_min = f_max / (2 ** ((v_max - v_min) * cents_per_value / 1200))
    return f_min


def get_f_max(f_min, cents_per_value, v_min, v_max):
    """
    This function takes in a y value max and min, a minimum frequency and a y scale parameter in units of cents/y value, and returns the maximum frequency that fits to such a scale.
    Cents are a logarithmic unit of tone intervals (https://en.wikipedia.org/wiki/Cent_(music)).

    Parameters
    ----------
    f_min : float
        Minimum frequency.
    cents_per_value : float
        A y scale parameter in units of cents/y value.
    v_min : float
        Minimum y value.
    v_max : float
        Maximum y value.

    Returns
    -------
    float
        Maximum frequency.

    """
    f_max = f_min * (2 ** ((v_max - v_min) * cents_per_value / 1200))
    return f_max


class FrequencyScale(object):
    """
    This class builds a frequency scale and populates the namespace of frequency objects based on the given inputs from the following combos:
        - frequency_min, frequency_max, y value min and y value max
        - frequency_max, cents_per_value, y value min and y value max
        - frequency_min, cents_per_value, y value min and y value max
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

    def __init__(self, value_min, value_max,
                 frequency_min=None, frequency_max=None, cents_per_value=None,
                 verbose=False):

        if verbose:
            print('initial vals (fmin, fmax, vmin, vmax):',
                  frequency_min, frequency_max, value_min, value_max)

        # checking for which inputs were given
        self.y_inputs = []
        if frequency_min != None:
            self.y_inputs.append('frequency_min')
        if frequency_max != None:
            self.y_inputs.append('frequency_max')
        if cents_per_value != None:
            self.y_inputs.append('cents_per_value')
        self.y_n_inputs = len(self.y_inputs)

        # raising exception if anything other than two inputs were given
        if self.y_n_inputs != 2:
            raise Exception('Frequency takes 2 of the frequency_min, frequency_max, and cents_per_value inputs. You inputted {} inputs, which were {}.'.format(
                self.y_n_inputs, self.y_inputs))

        # frequency_min and frequency_max input case
        if (cents_per_value == None):
            cents_per_value = cent_per_value(frequency_min, frequency_max,
                                             value_min, value_max)

        # cents_per_value and frequency_max input case
        if (frequency_min == None):
            frequency_min = get_f_min(frequency_max, cents_per_value,
                                      value_min, value_max)

        # cents_per_value and frequency_min input case
        if (frequency_max == None):
            frequency_max = get_f_max(frequency_min, cents_per_value,
                                      value_min, value_max)

        self.y_value_min = value_min
        self.y_value_max = value_max
        self.y_frequency_max = frequency_max
        self.y_frequency_min = frequency_min
        self.y_cents_per_value = cents_per_value

        if self.y_frequency_max > piano_max:
            warnings.warn('Your maximum frequency of {} Hz is above a pianos maximum of {} Hz.'.format(
                np.round(self.y_frequency_max, 2), piano_max))
        if self.y_frequency_min < piano_min:
            warnings.warn('Your minimum frequency of {} Hz is below a pianos minimum of {} Hz.'.format(
                np.round(self.y_frequency_min, 2), piano_min))
        if self.y_value_min > self.y_value_max:
            warnings.warn('Min y value is greater than max y value.')

        if verbose:
            print('initial vals (f_min, f_max, y_min, y_max):', self.y_frequency_min,
                  self.y_frequency_max, self.y_value_min, self.y_value_max)

        def freq(v): return self.y_frequency_min * \
            2 ** ((v - self.y_value_min) * self.y_cents_per_value / 1200)
        self.y_freq_translate_to_range = lambda array: list(map(freq, array))

        if verbose:
            print('Frequency Scale Built')
