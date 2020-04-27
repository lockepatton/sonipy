# `sonipy` : A Scatterplot Sonification Package

A package to turn scatter plots into perceptually uniform sound files for use in science and to make science more accessible.

## What does `sonipy` do?

Our method of sonification takes in scatterplot data and produces audio clips that depict each datapoint as a short sound blip with a y value corresponding to pitch and an x value corresponding to arrival time.

![sonipy setup](./paper/images/Method2.png)

**Each data point has a corresponding short tone called a `blip`,** with a y value corresponding to its pitch and a x value corresponding to its arrival time. Higher y value data points have higher corresponding blip pitches.

## Installation

`sonipy` is pip-installable from command line:

``` bash
pip install sonipy
```

Alternately, you can clone the repository and install it yourself:

``` bash
git clone https://github.com/lockepatton/sonipy.git
cd sonipy
python setup.py install
```


## Example Easy Setup

``` Python
from sonify import *

C4 = 261.6 # Hz
args = {'frequency_min' : C4,
        'frequency_max' : C4*4}

SN = MultiTone(values=x, durations=y,
               length=0.5, **args)
SN.SaveTone()
```

## Extended Setup

``` Python
from sonify import *

C4 = 261.6 # Hz
frequency_args = {
  'frequency_min' : C4,
  'frequency_max' : C4*4
  # 'cents_per_value' : -680,
  # 'value_min' : 0,
  # 'value_max' : 1,
}

duration_args = {
  'time_total' : 2,
  # 'time_min' : None,
  # 'time_max' : None,
}
duration_scale = None

SN = SonifyTool(values=x, durations=y,
                frequency_args = frequency_args,
                duration_args = duration_args,
                # duration_scale = duration_scale,
                length=0.5, **args)
SN.SaveTone()
```

## Parameter Inputs

### Frequency Scale Parameters:

1. a minimum frequency <img src="https://render.githubusercontent.com/render/math?math=f_{min}"> and it's corresponding minimum y value <img src="https://render.githubusercontent.com/render/math?math=y_{min}">
2. a maximum frequency <img src="https://render.githubusercontent.com/render/math?math=f_{max}"> and it's corresponding maximum y value <img src="https://render.githubusercontent.com/render/math?math=y_{max}">
3. a change in pitch (measured in [cents](https://en.wikipedia.org/wiki/Cent_(music))) over change in y value parameter <img src="https://render.githubusercontent.com/render/math?math=\frac{dc}{dy}">

All frequency parameters are entered inside the args parameter.

### Time Scale Parameters:

By default, the sound files are 2 seconds.

1. a total time of the soundfile <img src="https://render.githubusercontent.com/render/math?math=t_{total}">
2. a change in time (measured in seconds) over change in x value parameter <img src="https://render.githubusercontent.com/render/math?math=\frac{dt}{dx}">

Time parameters are entered by simply by defining a duration_scale (in seconds per x value). Or alternately by passing a duration_args dictionary with some total time, smallest delta time between points or max delta time between points.

## Demos
Several Jupyter notebooks that demonstrate some use cases and examples of sonipy are found
[here](https://github.com/lockepatton/sonipy/tree/master/demos).
