import numpy as np
import matplotlib
from sonipy.scales.durations import *

import warnings
import pytest
from unittest import TestCase

times = np.array([0,5,7])
time_total = 2000. # 2 seconds
time_min = .1
time_max = .5

robustness_diff = 0.0001
s_to_ms = 1000.

debug = False

class Test_getScale(TestCase):

    def test_TooManyInputs(self):
        def ThreeInputs():
            scale = getScale(times, time_total=time_total, time_min=time_min, time_max=time_max)

        self.assertRaises(Exception, ThreeInputs)

    def test_time_total_input(self):
        scale_true = from_total_ms(times, time_total)
        scale = getScale(times, time_total=time_total)

        self.assertEqual(scale_true, scale)

    def test_time_min_input(self):
        scale_true = (times[2]-times[1]) / time_min
        scale = getScale(times, time_min=time_min)

        self.assertEqual(scale_true, scale)

    def test_time_max_input(self):
        scale_true = (times[1]-times[0]) / time_max
        scale = getScale(times, time_max=time_max)

        self.assertEqual(scale_true, scale)

    def test_scale_is_0_returns_exception(self):
        def time_total_is_zero():
            scale = getScale(times=times, time_total=0.)
            print(scale)
        self.assertRaises(Exception, time_total_is_zero)

    def test_time_min_when_repeats(self):
        scale_true = 1 / time_min
        scale = getScale(times=[0.,0.,1.], time_min=time_min)

        self.assertEqual(scale_true, scale)

def getDurationsMainRun(self, times):
    time_total = 200. #ms
    scale = getScale(times, time_total=time_total)

    DurationsObj = DurationsScale(scale)
    starttimes = DurationsObj.getDurations(times)

    if debug == True:
        print "times\t", DurationsObj.times
        print "durations\t",DurationsObj.durations
        print "dt\t",DurationsObj.dt
        print "dt / scale", DurationsObj.dt/ DurationsObj.scale
        print "sum(dt)\t", np.sum(DurationsObj.dt)
        print "time_total\t", time_total, time_total / s_to_ms
        print "starttimes",starttimes
        print "starttimes[0]",starttimes[0],type(starttimes[0])

    # testing first durations argument
    first_duration_truth = (times[1]-times[0])/scale
    self.assertEqual(DurationsObj.durations[0], first_duration_truth)

    # testing total time matches sum of dt
    self.assertEqual(np.round(np.sum(DurationsObj.durations),6), np.round(time_total,6))

    # testing starttimes start at 0
    self.assertEqual(starttimes[0], 0.0)

    # and the max start time ends at the total time scales
    self.assertTrue(np.max(starttimes) - time_total/s_to_ms <= robustness_diff)

class Test_DurationsScale(TestCase):

    def test_initState(self):
        scale = getScale(times, time_total=time_total)
        DurationsObj = DurationsScale(scale)

        self.assertEqual(DurationsObj.scale, scale)

    def test_getDurations_linear(self):
        getDurationsMainRun(self,[0,1.,2])

    def test_getDurations_nonlinear(self):
        getDurationsMainRun(self,[0,.5,3])

    def test_getDurations_nonlinear2(self):
        getDurationsMainRun(self,[2,3,7])

    def test_get_Durations_randomInput(self):
        times = np.sort(np.random.random(10))
        getDurationsMainRun(self,times)

    def test_plotDurations(self):
        scale = getScale(times, time_total=time_total)

        DurationsObj = DurationsScale(scale)
        DurationsObj.getDurations(times)
        fig, ax = DurationsObj.plotDurations()

        assert isinstance(fig, matplotlib.figure.Figure)


if __name__ == '__main__':
    unittest.main()
