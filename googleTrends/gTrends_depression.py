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

#%% perform an fft

N = len(df)
fft_pad = 7*np.ceil((N*1.1)/7) - N
N = N + fft_pad

sr = (df['Date'][1]-df['Date'][0]).days
T = N/sr
freq = np.arange(N)/T

dep_fft = fft(df['Freq'].values,n=980)

#%%

plt.stem(freq, np.abs(dep_fft), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Freq')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 2)
plt.show()

#%%

plt.plot(df['Date'],df['Freq'],c='b',lw=0.5)
plt.show()