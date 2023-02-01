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
from scipy.signal import get_window, find_peaks

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
t = np.linspace(0.0,2*np.pi,N+1)[0:-1]

# preallocate array (betas)
B = np.zeros((N,8))

# create a matrix, we'll use constant, lin. trend and 1-3 cycles (sin+cos)
B[:,0] = 1                  # constant
B[:,1] = np.linspace(0,1,N) # linear trend
B[:,2] = np.sin(t)          # single cycle
B[:,3] = np.cos(t)          # "
B[:,4] = np.sin(2*t)        # double cycle
B[:,5] = np.cos(2*t)        # "
B[:,6] = np.sin(3*t)        # triple cycle
B[:,7] = np.cos(3*t)        # "

# perform the regression
modelFit = np.linalg.lstsq(B,df['Freq'],rcond=None)

# calculate residuals (Freq detrend)
df['Freq_dt'] = df['Freq'] - (B @ modelFit[0])

# plot the comparison
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(df['Date'],df['Freq'],c='b',lw=0.5)
ax1.set_title('Original data')
ax2.plot(df['Date'],df['Freq_dt'],c='b',lw=0.5)
ax2.set_title('Detrended data')
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
#   If we do fft, expect mag. peak around 16 cycles, aka yearly

#%% perform an fft

# calculate number of years data we have
nYrs = ((df['Date'].iloc[-1] - df['Date'].iloc[0]).days)/365.25

# perform fft
#   applying hann window to reduce frequency leakage
#   upsampling (zero padding) to 4096 points to increase frequency resolution
n = 4096
w = np.fft.rfft(df['Freq_dt']*get_window('hann',N),n=n)

# calculate amplitude and phase (as cosine phase, radians)
w_amp = 2.0/n * np.abs(w[:n//2 + 1])
w_ph  = np.angle(w[:n//2 + 1])

# get frequency range (in cycles per year)
freqs = np.fft.rfftfreq(n,d=nYrs/N)

# find the peaks in the amp. spectrum
#   using standard outlier def. q3+1.5IQR for min. of peak height
#   using half distance to one cycle as min. distance
q25, q75 = np.percentile(w_amp, [25, 75])
iqr = q75 - q25
mHeight = q75 + 1.5*iqr
mDist = np.round(1/(2*freqs[1])) # freqs starts at 0, so freqs[1] gives incr.

#%%
peaks, _ = find_peaks(w_amp, height=mHeight, distance=mDist)

test = np.column_stack((peaks,freqs[peaks],peak_prominences(w_amp,peaks)[0]))

# plot
f, ax = plt.subplots()
ax.plot(freqs,w_amp)
plt.plot(freqs[peaks], w_amp[peaks], "x")
ax.set_xlim(0,16)
ax.set_ylim(0,0.6)
ax.set_xlabel('Cycles per year')
ax.set_ylabel('Amplitude')
plt.show()
