---
title: 'sonipy: A Perceptually Uniform Sonification Package'
tags:
  - Python
  - Sonification
  - Astronomy
  - Supernova
authors:
  - name: Locke Patton^[locke.patton@cfa.harvard.edu]
    orcid: 0000-0002-7640-236X
    affiliation: "1, 2"
  - name: Emily Levesque^[emsque@uw.edu]
    affiliation: 1
affiliations:
 - name: University of Washington, Department of Astronomy,  Seattle, WA 98195 USA
   index: 1
 - name: Center for Astrophysics | Harvard and Smithsonian, 60 Garden St, Cambridge, MA 02138
   index: 2
date: 1 July 2020
bibliography: paper.bib
---

![Example sonification case: an exploding star's change in brightness is plotted against time. Each datapoint corresponds to a tone blip at a frequency specified by its y value and a time specified by its x value. As the sound file plays, it scans the plot left to right, with the brightest moments of the exploding star reaching pitches of 3 times middle C on the piano and the tail of the cooling supernovae remnant dropping into lower audible pitches. \label{fig:largeexample}](./images/Picture1-nobkgd.png)



Introduction
============

`Sonipy` moves beyond visual analyses by sonifying scatter-plot data,
producing audio files that depict variations in y as perceptually
uniform changes in pitch. Blips are sounded in time at intervals
corresponding to x values.

Understanding pitch
-------------------

The cent is a logarithmic unit of measure for pitch intervals where
$n \approx 3986\log(b/a)$ defines the number of cents between the pitch
frequencies a and b.

Human Pitch Sensitivity
-----------------------

The average person is capable of discerning independent subsequent
pitches with a difference of  10 cents (Kollmeier et al. 2008). The
human ear is most sensitive to frequencies between   500-4000 Hz,
similar to the range of a standard piano.

With these parameters, xy scatterplot data can be translated into audio
files that map y values to specific pitch frequencies, with the minimum
discernible $\Delta y$ corresponding to a 10 cent pitch difference.

The Case for Sonification
=========================

Why sonify lightcurves?
-----------------------

Thanks to the nature of human hearing, we can audibly discern pitch
differences of 10 cents. On a y scale ranging from 0 to 10, that
corresponds to hearing variations as small as $\Delta y~0.03$. This
simultaneous depth and range of pitch is unique to sound and incredibly
powerful as a tool for understanding nuances in data.

Furthermore, through our sonification efforts of periodically variable
astronomy sources, we have discovered that periodicity that is visually
indiscernible can be heard in our sonified data.

Finally and most importantly, this approach opens up science and citizen
science to participants who are visually impaired, and empowers BVI
individuals to explore their own data.

Our Sonification Technique
==========================

$$\begin{gathered}
\label{eq:kajiya}
L_o \left( \mathbf{x},\omega_o,\lambda,t \right) = L_e\left(\mathbf{x},\omega_o,\lambda,t \right) + \\
   \int_\Omega f_r \left(\mathbf{x},\omega_i,\omega_o,\lambda,t\right) L_i\left(\mathbf{x},\omega_i,\lambda,t\right) \left(\omega_i \cdot \mathbf{n}\right) d\omega_i\end{gathered}$$

\centering
![image](paper/images/Method1.png){width=".7\linewidth"}

1.  Each data point corresponds to a short tone in the sound file.

2.  X value determines the placement of the tone in time.

3.  The Y value determines the tone's pitch.

4.  As Y value decreases, the tone's pitch gets lower.

Why this method?
-----------------

Each datapoint corresponds to a tone blip at a frequency specified by
its y value and a time specified by its x value. As the sound file
plays, it scans the plot left to right, with higher y datapoints causing
higher pitched blips and vica verca.

Our method is tailored to the capabilities of the human ear and audio
equipment. It is flexible, applies to a broad variety of data inputs, is
fast to generate, and offers a unique means of classifying data.

We avoid methods that match changes in y to decibels, because our
perception of loudness is not a perceptually uniform space and
inconsistent across users. As our method is tailored to a science case,
a linear increase in y corresponds to a perceptually uniform and linear
increase in perceived pitch.

The Code
========

**Total File Duration:** Duration is set by a total time, a duration
scale, or by choosing the length of minimum or maximum time difference.

**Frequency scale:** Frequency is set by one of the following: a
frequency minimum and maximum, or a frequency maximum and a cents / y
value scale value.

\vspace{-1.25\baselineskip}
``` {#lst:code_direct .python language="Python" frame="lines" label="lst:code_direct" caption="A simple textured shader." basicstyle="\footnotesize"}
from sonify import *

C4 = 261.6 # Hz
args = {'frequency_min' : C4,
        'frequency_max' : C4*4,
        'value_min' : 0,
        'value_max' : 1}

SN = MultiTone(values=x, durations=y,
               length=0.5, **args)
SN.SaveTone()
```

Our Astronomy Case Study
========================

Citizen Science - Supernova Lightcurves
---------------------------------------

We're building TransientZoo, a citizen science program that will allow
participants, including the visually impaired, to classify supernova
lightcurves using sound.

Figure [\[fig:Ia\]](#fig:Ia){reference-type="ref" reference="fig:Ia"}
and [\[fig:IIb\]](#fig:IIb){reference-type="ref" reference="fig:IIb"}
are two examples of successfully sonified audio light curves, for a Type
IIb and Type Ia supernova. We find that linear and plateau supernova
light curves can be audibly differentiated. This approach offers a new
tool for citizen science lightcurve classification.

\centering
![A type Ia supernova
lightcurve.[]{label="fig:Ia"}](paper/images/Ia.png){#fig:Ia
width=".7\linewidth"}

\centering
![A type IIb supernova
lightcurve.[]{label="fig:IIb"}](paper/images/IIb.png){#fig:IIb
width=".7\linewidth"}

Other Variable Objects in Astronomy
-----------------------------------

We've also explored the sonification of other time-domain data, which
will eventually help TransientZoo expand into LightcurveZoo. To the right
are pictured examples of an eclipsing binary from Kepler's catalogue and
an RR Lyrae from our observations. LightcurveZoo will ultimately include
a collection of transients: supernovae, binaries, and variable stars.
