from sonify import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpfuncs import *


cutoff = None
object = 'RRLyrae3733346'
time_total = 5000 #ms
period = .6818

All = pd.read_csv('./astr481/RRLyrae3733346.csv', delimiter=',')
print(All)

# mags, times = np.array(All['diffMag'])[:cutoff], np.array(All['T/Period'])[:cutoff]
mags = np.append(np.append(All['diffMag'], All['diffMag']), All['diffMag'])
times = np.append(np.append(All['T/Period'], period+All['T/Period']), 2*period+All['T/Period'])

fig, ax = plt.subplots(1)
# ax.plot(times, mags, alpha=.2)
ax.scatter(times, -mags, marker='x')
fig.suptitle(object)
ax.set_xlabel("Time within one period, .6818 days",)
ax.set_ylabel("Differential Magnitude, $m_{RR} - <m>_{background}$")
fig.savefig('./plots/'+object+'_time_vs_diff_mag.jpg')

times_adjusted = centerMagMax(times, mags)
times_logs, mags_logs = logTimes(times, mags)
times_logs = np.array(times_logs)

# Durations
duration_args = {
    # 'time_min' : 50, #ms
    # 'time_max' : 2000, #ms
    'time_total': time_total,  # ms
}

# finding duration scale
thisscale = getScale(times=times, **duration_args)
scale = thisscale

MJD_Scale = DurationsScale(scale)
MJD_Durations = MJD_Scale.getDurations(times=times)

# frequencies
mags, times
C4 = 261.6  # Hz
args = {
    'frequency_min': C4, #* (1 / 2),
    'frequency_max': C4 * 3,
    # 'cents_per_value' : -680,
    'value_min': max(mags),
    'value_max': min(mags),
    # 'value_min': min(mags),
    # 'value_max': max(mags),
}

# LINEAR TIME SCALE
print(scale)


SN_MultiTone = MultiTone(values=mags[:], durations=MJD_Durations[:], length=.1, **args)

fig, ax = plt.subplots(1)
ax.plot(mags, SN_MultiTone.frequencies,marker='x')
fig.savefig('./plots/mag_vs_freq'+object+'.jpg')

DataToSave = pd.DataFrame({'times' : times, 'diffmag' : mags, 'freq' : SN_MultiTone.frequencies})
DataToSave.to_csv('./astr481/RRLyrae3733346_times_mag_freq.csv')

SN_MultiTone.SaveTone(filename='multiscales/MultiTone_' + object + '.wav')
# SN_MultiTone.SaveTone(filename='./multiscales/MultiTone_' + type_all + '__' + SN + '.wav')
#
# # # SN_StringTone = StringTone(values=mags, durations=MJD_Durations, fadetime=10, **args)
# # # SN_StringTone.SaveTone(filename='StringTone_' + SN + '.mp4')
# # # SN_StringTone.SaveTone(filename='StringTone_' + type_all + '__' + SN + '.mp4')


print ('complete')
