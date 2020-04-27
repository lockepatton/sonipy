# `sonipy` : A Perceptually Uniform Scatterplot Sonification Package

A package to turn scatter plots into perceptually uniform sound files for use in science and to make science more accessible.

## What does `sonipy` do?

Our method of sonification takes in scatterplot data and produces audio clips that depict each datapoint as a short sound blip with a y value corresponding to pitch and an x value corresponding to arrival time.

![sonipy setup](/Users/lockepatton/sonipy/paper/images/Method2.png)


**Each data point has a corresponding short tone called a `blip`,** with a y value corresponding to its pitch and a x value corresponding to its arrival time.

### The pitch of the blip corresponds to its y values

A completely well-defined y frequency scale has the following parameters:

1. a minimum frequency $f_{min}$ and it's corresponding minimum y value $y_{min}$
2. a maximum frequency $f_{max}$ and it's corresponding maximum y value $y_{min}$
3. a change in pitch (measured in [cents](https://en.wikipedia.org/wiki/Cent_(music))) over change in y value parameter $\frac{dc}{dy}$

We relate any given y value to it's corresponding frequency via:

$$ f = \frac{f_{max}}{2^{\frac{dc}{dy} [y_{max} - y] ~/~ 1200}} $$


### The arrival time of the blip corresponds to its x value

A completely well-defined x time scale has the following parameters:

1. a minimum x value $x_{min}$
2. a maximum x value $x_{max}$
3. a total time of the soundfile $t_{total}$
4. a change in time (measured in seconds) over change in x value parameter $\frac{dt}{dx}$

$$ t = \frac{dt}{dx} [x - x_{min}]$$


## Installation

`pip install sonipy`
