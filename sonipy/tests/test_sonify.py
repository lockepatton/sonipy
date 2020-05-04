from unittest import TestCase
import pytest
import sonipy
import numpy as np
import thinkdsp

from sonipy.sonify import MultiTone, SonifyTool
from sonipy.scales.durations import DurationsScale, getScale

# linear sample scatterplot data
x = np.linspace(-10, 2, 10)
y = np.linspace(-1, 1, 10)

# non-linear sample scatterplot data
x_nonlin = np.linspace(0, 1, 10)
y_nonlin = np.sin(x * np.pi * 2)

# random sample scatterplot data, and those random values ordered by x value order
x_random = np.random.random(10) * 2
y_random = np.random.random(10) * 2
x_random_sorted_idx = x_random.argsort()
x_random_ordered = x_random[x_random_sorted_idx]
y_random_ordered = y_random[x_random_sorted_idx]

C4 = 261.6  # Hz
time_total = 2000  # ms
frequency_args = {"frequency_min": C4, "frequency_max": 4 * C4}

robustness_diff = 0.00001
robustness_round_tolerence = 4
s_to_ms = 1000.

def createTone(times, values, length=.1, time_total=time_total, frequency_args=frequency_args):
    """Runs code to create a multitone and calculate the scale"""

    scale = getScale(times=times, time_total=time_total)
    DurScale = DurationsScale(scale)
    starttimes = DurScale.getDurations(times=times)

    return scale, MultiTone(values=values, starttimes=starttimes, length=length,
                            alertMultitoneCreated=False, **frequency_args)


def starttimes_match_to_x_via_scale(self, times, values):
    """Test to see if the total starttimes scales to match x values - min(x value)"""

    scale, Tone = createTone(times, values)

    starttimes_in_xvalue = Tone.starttimes * scale * s_to_ms
    times_minus_min_times = times - min(times)

    self.assertTrue(np.sum(starttimes_in_xvalue -
                           times_minus_min_times) <= robustness_diff)


def tonelength_equals_total_time_plus_bliptime(self, times, values, length=.5):
    """Test to see total time +  blip length ~ length of tone"""

    scale, Tone = createTone(times, values, length=length)

    self.assertEqual(np.round(Tone.multitone.duration,
                              robustness_round_tolerence), time_total / s_to_ms + length)


class TestMultiTone(TestCase):

    def test_multitones_is_thinkdsp_instance(self):
        # testing to multitone was created and is an instance of thinkdsp.Wave
        times, values = x, y
        scale, Tone = createTone(times, values)
        assert isinstance(Tone.multitone, thinkdsp.Wave)

    def test_setup_starttimes_match_to_x_via_scale(self):
        # testing to see if the total starttimes scales to match x values - min(x value)

        # linear x, linear y
        times, values = x, y
        starttimes_match_to_x_via_scale(self, times, values)

        # linear x, y = sin(x)
        times, values = x_nonlin, y_nonlin
        starttimes_match_to_x_via_scale(self, times, values)

        # random x, random y
        times, values = x_random, y_random
        starttimes_match_to_x_via_scale(self, times, values)

        # ordered random x, random y
        times, values = x_random_ordered, y_random_ordered
        starttimes_match_to_x_via_scale(self, times, values)

    def test_tonelength_equals_total_time_plus_bliptime(self):
        # total time +  blip length ~ length of tone

        # linear x and y
        times, values = x, y
        tonelength_equals_total_time_plus_bliptime(
            self, times, values, length=.5)

        # ordered random x, random y
        times, values = x_random_ordered, y_random_ordered
        tonelength_equals_total_time_plus_bliptime(
            self, times, values, length=.5)

    def test_vmin_vmax_not_finite_input(self):
        # test when value_max is given as -inf, nan

        times, values = x, y

        def value_max_input_is_None(self):
            frequency_args = {"frequency_min": C4,
                              "frequency_max": 4 * C4, "value_max": -np.inf}
            scale, Tone = createTone(
                times, values, frequency_args=frequency_args)

        self.assertRaises(Exception, value_max_input_is_None)

    def test_vmin_vmax_equal_None_input(self):
        times, values = x, y

        def value_max_input_is_None(self):
            frequency_args = {"frequency_min": C4,
                              "frequency_max": 4 * C4, "value_max": None}
            scale, Tone = createTone(
                times, values, frequency_args=frequency_args)

    def test_fade_equals_false_argument_in_Multitone(self):
        # testing fade=False argument runs smoothly
        times, values = x, y
        scale = getScale(times=times, time_total=time_total)
        DurScale = DurationsScale(scale)
        starttimes = DurScale.getDurations(times=times)

        MultiTone(values=values, starttimes=starttimes, length=0.3,
                  alertMultitoneCreated=False, fade=False, **frequency_args)

    def test_play(self):
        # check to make sure .play(), the wrapper of ThinkDSP .make_audio(), runs
        scale, Tone = createTone(x, y)
        Tone.play()


class TestSonifyTool(TestCase):

    def test_values_and_times_differ_in_length_must_raise_exception(self):

        def values_doesnt_equal_times_length():
            values = np.linspace(0, 1, 2)
            times = np.linspace(0, 1, 3)

            Tone = SonifyTool(values, times)

        self.assertRaises(Exception, values_doesnt_equal_times_length)

    def test_unordered_XY_correctly_orders_based_on_xvalues(self):
        # having times be in reverse order
        values = np.linspace(0, 1, 3)
        times = np.linspace(0, 1, 3)[::-1]

        Tone = SonifyTool(values, times, alertMultitoneCreated=False)

        # testing to make sure the times have been ordered correctly (to within the robustness_diff)
        self.assertTrue(np.sum(Tone.times - times[::-1]) < robustness_diff)

    def test_ints_entered_instead_of_arrays_causes_pandas_dataframe_exception(self):
        def entered_values_and_times_are_single_ints():
            values = 1
            times = 1
            Tone = SonifyTool(values, times, alertMultitoneCreated=False)

        self.assertRaises(Exception, entered_values_and_times_are_single_ints)

    def test_ints(self):
        values = np.array([1, 2], dtype='int')
        times = [1, 3]
        Tone = SonifyTool(values, times, alertMultitoneCreated=False)

    def test_lists(self):
        values = [1, 2]
        times = [1, 3]
        Tone = SonifyTool(values, times, alertMultitoneCreated=False)

    def test_valuemin_and_maxes_outside_true_min_max(self):
        values = [1, 2]
        times = [1, 3]
        frequency_args = {
            "frequency_min": C4,
            "frequency_max": 4 * C4,
            "value_min": 0.,
            "value_max": 1.,
        }

        Tone = SonifyTool(values, times, frequency_args=frequency_args, alertMultitoneCreated=False)

    def test_warning_when_value_max_less_than_value_min(self):
        values = [1, 2]
        times = [1, 3]
        frequency_args = {
            "frequency_min": C4,
            "frequency_max": 4 * C4,
            "value_min": 10,
            "value_max": 0.,
        }
        with pytest.warns(UserWarning) as record:
            Tone = SonifyTool(values, times, frequency_args=frequency_args, alertMultitoneCreated=False)

        self.assertTrue(len(record) == 1)
        self.assertTrue("greater than" in str(record[0].message))

    def test_input_states(self):
        values = [1, 2]
        times = [1, 2]

        # 1 second total
        duration_scale = 1. / s_to_ms #x value / time (ms)

        # 2 second total
        duration_args={"time_total": 2000.}
        duration_scale_to_match_dur_args = 1. / (2*s_to_ms) #x value / time (ms)

        # duration scale - should be 1 second total
        Tone = SonifyTool(values, times, frequency_args=frequency_args,
                          duration_scale=duration_scale, duration_args=duration_args,
                          alertMultitoneCreated=False)
        self.assertEqual(Tone.duration_scale, duration_scale)

        # duration arguments - should be 2 seconds total
        Tone = SonifyTool(values, times, frequency_args=frequency_args,
                          duration_args=duration_args,
                          alertMultitoneCreated=False)
        self.assertEqual(Tone.duration_scale, duration_scale_to_match_dur_args)


if __name__ == '__main__':
    unittest.main()
