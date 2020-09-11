from unittest import TestCase
import pytest

import numpy as np
from sonipy.sonify import SonifyTool
from sonipy.visualisation import plotData

# linear sample scatterplot data
x_lin = np.linspace(-10, 2, 10)
y_lin = np.linspace(-1, 1, 10)


class TestPlotData(TestCase):

    # simple "will it run"? test for plotting wrapper.
    def test_plotData(self):
        Tone = SonifyTool(x_lin, y_lin)
        plotData(Tone)
        plotData(Tone, percentedge=0.15)


if __name__ == '__main__':
    unittest.main()
