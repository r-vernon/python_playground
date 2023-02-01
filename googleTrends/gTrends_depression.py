#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:11:15 2023

@author: richard
"""

# import necessary stuff
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy import signal

# set dpi for figures
plt.rcParams["figure.dpi"] = 300

# set directory
os.chdir('/home/richard/Documents/Python/googleTrends/')

#%% read in the data
# original data ('data') downloaded yearly
# new data ('data2') downloaded relative to 2014, which had highest peak
# - that ensures that all trend data has the same scale

all_dat = []

for currYear in range(2006,2023):
    
    # read in current year
    if currYear == 2014:
        # read in current year, skip 2 rows, 
        # **use colums 0+1**, treat row0 as header, rename cols, 
        # parse col1 as date
        df = (pd.read_csv('./data2/Depression_%d.csv' % (currYear),skiprows=2,
                          usecols=[0,1],header=0,names=['Date','Freq'],
                          parse_dates=[0]))
    else:
        # read in current year, skip 2 rows, 
        # **use colums 2+3**, treat row0 as header, rename cols, 
        # parse col1 as date
        df = (pd.read_csv('./data2/Depression_%d.csv' % (currYear),skiprows=2,
                          usecols=[2,3],header=0,names=['Date','Freq'],
                          parse_dates=[0]))
    
    # no longer need to standardize - although probably didn't need to detrend
    #  in first place!
    # detrend and standardize (z-score; demeaning again to be safe!)
    #df['Freq'] = signal.detrend(df['Freq'].values,type='linear')
    #df['Freq']=(df['Freq']-df['Freq'].mean())/df['Freq'].std()
    
    # add to list of all data
    all_dat.append(df)

# concatenate
df = pd.concat(all_dat, axis=0, ignore_index=True)

# check no missing dates (there aren't!)
# dateDiff = df['Date'].diff()
# dateDiff.plot()

#%% create detrended version of the data
# we have 16yrs of data, so will filter out up to 3 cycles (every 5yrs ish)
#   as looking for cyclical shifts in a year, not longer trends

# number of data points
N = len(df)

# time series (for sin/cos regressors)
t = np.linspace(0.0,2.0*np.pi,N+1)[0:-1]

# preallocate array (betas)
B = np.zeros((N,8))

# create a matrix, we'll use constant, lin. trend and 1-3 cycles (sin+cos)
B[:,0] = 1.0                    # constant
B[:,1] = np.linspace(0.0,1.0,N) # linear trend
B[:,2] = np.sin(t)              # single cycle
B[:,3] = np.cos(t)              # "
B[:,4] = np.sin(2.0*t)          # double cycle
B[:,5] = np.cos(2.0*t)          # "
B[:,6] = np.sin(3.0*t)          # triple cycle
B[:,7] = np.cos(3.0*t)          # "

# perform the regression
modelFit = np.linalg.lstsq(B,df['Freq'],rcond=None)

# calculate residuals (Freq detrend)
df['Freq_dt'] = df['Freq'] - (B @ modelFit[0])

# create a smoothed version to remove outliers (median filter)
df['Freq_sm'] = signal.medfilt(df['Freq_dt'],5)

# plot the comparison
ax1 = plt.subplot(211)
ax1.plot(df['Date'],df['Freq'],c='k',lw=0.75)
ax1.set_title('Original data')
ax1.set_ylim(20,120)
ax1.set_yticks(np.arange(20,140,20))

ax2 = plt.subplot(223)
ax2.plot(df['Date'],df['Freq_dt'],c='k',lw=0.75)
ax2.set_title('Detrended data')
ax2.set_ylim(-40,40)
ax2.set_yticks(np.arange(-40,60,20))
ax2.xaxis.set_major_formatter(DateFormatter('%y')) # convert 2006 to 06

ax3 = plt.subplot(224)
ax3.plot(df['Date'],df['Freq_sm'],c='k',lw=0.75)
ax3.set_title('Smoothed data')
ax3.set_ylim(-20,20)
ax3.set_yticks(np.arange(-20,30,10))
ax3.xaxis.set_major_formatter(DateFormatter('%y'))

plt.tight_layout()
plt.show()

#%% initial exploration - compare by month

# group by month (ignoring day, year)
df_byMth = df['Freq_dt'].groupby(by=df.Date.dt.month)

# create list of month names (alt: calander module, or pandas dt.strftime)
mthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# plot the result as a bar chart
f, ax = plt.subplots()
ax.bar(np.arange(12), df_byMth.mean(), yerr=df_byMth.std(), 
       align='center', color='dimgray', edgecolor='black', 
       ecolor='black', capsize=0, tick_label=mthNames)
ax.set_title('Depression Searches by Month')
ax.set_ylabel('Search Frequency (Arb. Units)')
ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1)
f.text(0.88, 0.15, 'Errorbars +/- 1SD', ha='right',size='x-small')
plt.show()

# NOTES:
#   Seems to show a trend rising in winter months, falling in summer
#   No rise for christmas though!
#   If we do fft, expect mag. peak around 16 cycles, aka 1 cycle per year

#%% perform an fft

# calculate number of years data we have
nYrs = ((df['Date'].iloc[-1] - df['Date'].iloc[0]).days)/365.25

# perform fft
#   applying hann window to reduce frequency leakage
#   upsampling (zero padding) to 2N points to increase frequency resolution
n = 2*N
w = np.fft.rfft(df['Freq_dt']*signal.get_window('hann',N),n=n)

# calculate amplitude and phase (as cosine phase, radians)
w_amp = 2.0/N * np.abs(w[:n//2 + 1])
w_ph  = np.angle(w[:n//2 + 1])

# get frequency range (in cycles per year)
freqs = np.fft.rfftfreq(n,d=nYrs/N)

# find the peaks in the amp. spectrum
#   using standard outlier def. q3+1.5IQR for min. of peak height
#   using half distance to one cycle as min. distance
#   using third of max peaks height for min. prominence
q25, q75 = np.percentile(w_amp, [25, 75])
mHeight = q75 + 1.5*(q75 - q25)
mDist = np.round(1.0/(2.0*freqs[1])) # freqs starts at 0, so freqs[1] gives incr.
mProm = w_amp.max()/3.0
peaks, pProp = signal.find_peaks(w_amp, height=mHeight, prominence=mProm,distance=mDist)

# plot the fourier spectrum
f, ax = plt.subplots()
ax.plot(freqs,w_amp,c='k',lw=0.75)
ax.set_xlim(0,16)
# ax.set_ylim(0,1.4)
ax.set_title('Fourier Spectrum of Depression Search Trends')
ax.set_xlabel('Cycles per year')
ax.set_ylabel('Amplitude')

# plot the peaks
h1, = ax.plot(freqs[peaks], w_amp[peaks], '.', c='r')
for cPk in peaks:
    ax.text(freqs[cPk]+0.3,w_amp[cPk],'%.1f' % (freqs[cPk]),va='center',size='x-small')

# create empty handle for legend (so can format it as desired)
h2, = ax.plot(0,0,visible=False)

# add a text box (legend) explaining peaks
tStr = ['Peaks identified using:',
    '\n'.join((
    '- height > %.2f' % (mHeight),
    '- distance between peaks > %.2f cycles/yr' % (freqs[round(mDist)]),
    '- height above neighbours (prominance) > %.2f' % (mProm)))]
ax.legend(handles=[h1,h2], labels=[tStr[0],tStr[1]], loc='upper right', 
          fontsize='x-small', facecolor='whitesmoke', labelspacing=0.1, 
          fancybox=False, edgecolor='black', handletextpad=0.0,
          borderpad=0.5, borderaxespad=1)

# show the thing
plt.show()

# found 3 peaks
#   first (1 cycle/yr) likely the dominant peak of interest
#   second (2 cycles/yr) likely just a harmonic
#   third (7 cycles/yr), no idea, interesting!

#%% explore the peaks (1+all)

# t = np.linspace(0.0,2*np.pi,n+1)[0:N]
p1 = w_amp[peaks[0]]*np.cos(peaks[0]*t + w_ph[peaks[0]])
p2 = w_amp[peaks[1]]*np.cos(peaks[1]*t + w_ph[peaks[1]])
p3 = w_amp[peaks[2]]*np.cos(peaks[2]*t + w_ph[peaks[2]])
p4 = p1 + p2 + p3

# plot the comparison
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(df['Date'],df['Freq_dt'],c='k',lw=0.75, alpha=0.75, clip_on=False)
ax1.plot(df['Date'],p1,c='r',lw=0.75)
ax1.set_ylim(-20,20)
ax1.set_yticks(np.arange(-20,30,10))

ax2.plot(df['Date'],df['Freq_dt'],c='k',lw=0.75, alpha=0.75, clip_on=False)
ax2.plot(df['Date'],p4,c='r',lw=0.75)
ax2.set_ylim(-20,20)
ax2.set_yticks(np.arange(-20,30,10))

plt.tight_layout()
plt.show()





























