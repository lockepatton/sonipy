import matplotlib.pyplot as plt
import numpy as np

def plotData(Tone, percentedge=0.1):
    """
    Plots start times and x values against y values.

    Parameters
    ----------
    Tone : MultiTone
        Multitone object to be plotted.
    percentedge : float
        Percentage of total x limits to be plotted as edge space.
    """

    def calc_limits(values, percentedge=percentedge):
        delta = (max(values) - min(values)) * percentedge
        return min(values) - delta, max(values) + delta

    fig, ax = plt.subplots(1, 1)

    ax.scatter(Tone.x, Tone.y_values)
    ax.set_xlim(calc_limits(Tone.x))
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')

    ax2 = ax.twiny()
    ax2.scatter(Tone.x_starttimes, Tone.y_values)
    ax2.set_xlim(calc_limits(Tone.x_starttimes))
    ax2.set_xlabel('Time in Audio File (s)')

    return fig, [ax, ax2]


# def animate(Tone,):
#     pass
