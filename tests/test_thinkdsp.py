from unittest import TestCase
import pytest

from sonipy.thinkdsp import *
import numpy as np

freq=440; amp=1.0; offset=0
ts = np.array([1]); ys = np.array([6.27613383e-14])
tolerance = 1e-8

class TestSignals(TestCase):

    def test_CosSignal(self):

        CosSignalObj = CosSignal(freq=freq, amp=amp, offset=offset)
        self.assertEqual(CosSignalObj.freq, freq)
        self.assertEqual(CosSignalObj.amp, amp)
        self.assertEqual(CosSignalObj.offset, offset)
        self.assertEqual(CosSignalObj.func, np.cos)

    def test_Sinusoid(self):

        SinusoidObj = Sinusoid(freq, amp, offset)
        self.assertEqual(SinusoidObj.freq, freq)
        self.assertEqual(SinusoidObj.amp, amp)
        self.assertEqual(SinusoidObj.offset, offset)
        self.assertEqual(SinusoidObj.func, np.sin)
        self.assertEqual(SinusoidObj.period, 1/freq)

        print(ts, ys)
        print(SinusoidObj.evaluate(ts)-ys)
        self.assertTrue((SinusoidObj.evaluate(ts)-ys) <= tolerance)

    def test_Signal(self):
        """testing basic functions run"""

        SignalObj = Signal()
        self.assertEqual(SignalObj.period, 0.1)

        def testplot(Signal):
            Signal.plot()

        def testmake_wave(Signal):
            Signal.make_wave()

        SinusoidObj = Sinusoid(freq, amp, offset)
        testplot(SinusoidObj)
        testmake_wave(SinusoidObj)

    def test_normalize(self):

        self.assertEqual(normalize(ys),[1.])


ys_long = [0,1,2,3,4.]
framerate = 11025.

class TestWave(TestCase):

    def test_Wave(self):

        WaveObj = Wave(ys_long)
        self.assertEqual(WaveObj.__len__(), len(ys_long))
        self.assertEqual(float(WaveObj.start), ys_long[0]/framerate)
        self.assertEqual(WaveObj.end, ys_long[-1]/framerate)


        def testcopy(WaveObj):
            WaveObj.copy()

        testcopy(WaveObj)






# class TestExample(TestCase):
#
#     def test_example1(self):
#         assert isinstance(int, int)
#
#     def test_example2(sefl):
#
#         def function_that_raises_exception():
#             raise Exception
#
#         self.assertRaises(Exception, function_that_raises_exception)
