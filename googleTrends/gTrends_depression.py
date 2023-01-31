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
from scipy import signal
from scipy.fftpack import fft

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
#   as looking for cyclical shifts

# number of data points
N = len(df)

# time series (for sin/cos regressors)
t = np.linspace(0,2*np.pi,N+1)[0:-1]

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

#%% perform an fft

#fft_pad = 7*np.ceil((N*1.1)/7) - N
#N = N + fft_pad

sr = (df['Date'][1]-df['Date'][0]).days
T = N/sr
freq = np.arange(N)/T

dep_fft = fft(df['Freq'].values,N)

#%%

plt.stem(np.arange(N-1), np.abs(dep_fft[1:]), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Freq')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(1, 20)
plt.show()

#%%

plt.plot(df['Date'],df['Freq_dt'],c='b',lw=0.5)
plt.show()