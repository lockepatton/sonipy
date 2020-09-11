import numpy as np
from sonipy.scales.frequency import *

import warnings
import pytest
from unittest import TestCase


C4 = 261.6  # Hz

# EXPECTED VALUES
frequency_min = C4
frequency_max = 4 * C4

# Case 1: Vmin Vmax 0 and 1 (defaults are None)
# cents_per_value w\ default vmin and vmax
cents_per_value = 2400.0
vmin = 0
vmax = 1

# Case 2: Vmin Vmax Defaults
# cents_per_value w\ alternate input vmin and vmax
cents_per_value2 = 12.0
vmin2 = -100
vmax2 = 100

# FREQUENCY RANGE WARNINGS
frequency_min_outofrange = 20  # Hz
frequency_max_outofrange = 45000  # Hz

class TestFrequencyScale(TestCase):

    def test_cent_per_value(self):
        cents_per_value_defaulttest = cent_per_value(
            frequency_min, frequency_max, vmin, vmax)
        self.assertEqual(cents_per_value_defaulttest, cents_per_value)

        cents_per_value_nondefaulttest = cent_per_value(
            frequency_min, frequency_max, vmin2, vmax2)
        self.assertEqual(cents_per_value_nondefaulttest, cents_per_value2)

    def test_get_f_min(self):
        frequency_min_defaulttest = get_f_min(
            frequency_max, cents_per_value, vmin, vmax)
        self.assertEqual(frequency_min_defaulttest, frequency_min)

        frequency_min_nondefaulttest = get_f_min(
            frequency_max, cents_per_value2, vmin2, vmax2)
        self.assertEqual(frequency_min_nondefaulttest, frequency_min)

    def test_get_f_max(self):
        frequency_max_defaulttest = get_f_max(
            frequency_min, cents_per_value, vmin, vmax)
        self.assertEqual(frequency_max_defaulttest, frequency_max)

        frequency_max_nondefaulttest = get_f_max(
            frequency_min, cents_per_value2, vmin2, vmax2)
        self.assertEqual(frequency_max_nondefaulttest, frequency_max)

    def test_TooManyInputsFrequencyScale(self):

        # fails when you give frequency scale 3 inputs
        def tooManyInputsFunc():
            Scale = FrequencyScale(frequency_min=frequency_min,
                                   frequency_max=frequency_max,
                                   cents_per_value=cents_per_value)

        # fails when you give frequency scale 3 inputs
        self.assertRaises(Exception, tooManyInputsFunc)

    def test_TooFewInputsInFrequencyScale(self):
        # fails when you give frequency scale 1 input

        def only_frequency_min():
            Scale = FrequencyScale(frequency_min=frequency_min)

        self.assertRaises(Exception, only_frequency_min)

        def only_frequency_max():
            Scale = FrequencyScale(frequency_max=frequency_max)

        self.assertRaises(Exception, only_frequency_max)

        def only_cents_per_value():
            Scale = FrequencyScale(cents_per_value=cents_per_value)

        self.assertRaises(Exception, only_cents_per_value)

    def test_FrequencyRangeWarnings(self):
        # frequency min out of range
        with pytest.warns(UserWarning) as record:
            Scale = FrequencyScale(frequency_min=frequency_min_outofrange,
                                   frequency_max=frequency_max,
                                   value_min = vmin,
                                   value_max = vmax)

        self.assertTrue(len(record) == 1)
        self.assertTrue("minimum" in str(record[0].message))

        # frequency max out of range
        with pytest.warns(UserWarning) as record:
            Scale = FrequencyScale(frequency_min=frequency_min,
                                   frequency_max=frequency_max_outofrange,
                                   value_min = vmin,
                                   value_max = vmax)

        self.assertTrue(len(record) == 1)
        self.assertTrue("maximum" in str(record[0].message))

        # both frequency min and max out of range
        with pytest.warns(UserWarning) as record:
            Scale = FrequencyScale(frequency_min=frequency_min_outofrange,
                                   frequency_max=frequency_max_outofrange,
                                   value_min = vmin,
                                   value_max = vmax)

        self.assertTrue(len(record) == 2)
        self.assertTrue("maximum" in str(record[0].message))
        self.assertTrue("minimum" in str(record[1].message))

    def test_value_min_greater_than_value_max_warning(self):
        # warns when you put value_min > value_max

        with pytest.warns(UserWarning) as record:
            Scale = FrequencyScale(frequency_min=frequency_min,
                                   frequency_max=frequency_max,
                                   value_min = vmax,
                                   value_max = vmin)

        self.assertTrue(len(record) == 1)
        self.assertTrue("greater than" in str(record[0].message))

    def test_missingVminVmax(self):

        # fails when you give frequency scale 3 inputs
        def missingVminVax():
            Scale = FrequencyScale(frequency_max=frequency_max,
                                   cents_per_value=cents_per_value)

        # fails when missing vmin and vmax
        self.assertRaises(Exception, missingVminVax)


    def test_alternateVminVmax(self):
        # missing frequency_min
        Scale = FrequencyScale(frequency_max=frequency_max, cents_per_value=cents_per_value2,
                               value_min=vmin2, value_max=vmax2)
        self.assertEqual(Scale.y_frequency_min, frequency_min)

        # missing frequency_max
        Scale = FrequencyScale(frequency_min=frequency_min, cents_per_value=cents_per_value2,
                               value_min=vmin2, value_max=vmax2)
        self.assertEqual(Scale.y_frequency_max, frequency_max)

        # missing cents_per_value
        Scale = FrequencyScale(frequency_min=frequency_min, frequency_max=frequency_max,
                               value_min=vmin2, value_max=vmax2)
        self.assertEqual(Scale.y_cents_per_value, cents_per_value2)


if __name__ == '__main__':
    unittest.main()
