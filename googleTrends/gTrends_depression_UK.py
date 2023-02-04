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

# set flag to save figure
saveF = 0

#%% read in the data
# data downloaded relative to 2014, which had highest peak
# - that ensures that all trend data has the same scale

all_dat = []

for currYear in range(2006,2023):
    
    # read in current year
    if currYear == 2014:
        # read in current year, skip 2 rows, 
        # **use colums 0+1**, treat row0 as header, rename cols, 
        # parse col1 as date
        df = (pd.read_csv('./data_UK/Depression_%d.csv' % (currYear),skiprows=2,
                          usecols=[0,1],header=0,names=['Date','Freq'],
                          parse_dates=[0]))
    else:
        # read in current year, skip 2 rows, 
        # **use colums 2+3**, treat row0 as header, rename cols, 
        # parse col1 as date
        df = (pd.read_csv('./data_UK/Depression_%d.csv' % (currYear),skiprows=2,
                          usecols=[2,3],header=0,names=['Date','Freq'],
                          parse_dates=[0]))
    
    # add to list of all data
    all_dat.append(df)

# concatenate
df = pd.concat(all_dat, axis=0, ignore_index=True)

# delete any data from 1970 as erroneous!
df.drop(df[(df['Date'].dt.year==1970)].index, inplace=True)

# delete all_dat now it's concatenated
del all_dat

# check no missing dates (there aren't!)
# dateDiff = df['Date'].diff()
# dateDiff.plot()

#%% create detrended version of the data
#   we have 17yrs of data, so will filter out up to 3 cycles (every 5yrs ish)
#   as looking for cyclical shifts in a year, not longer trends
#   (effectively crude high pass filter!)

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

#%% collapse the signal to represent 1yrs data

# copy over just columns we want and set date as index
df_us = df[['Date','Freq_dt']].copy()
df_us.set_index('Date', inplace=True)

# upsample so every day is represented
df_us = df_us.asfreq(freq='D')

# interpolate
df_us['Freq_dt'] = df_us.interpolate(method='time')

# find and drop leap year days (Feb 29)
df_us.drop(df_us[(df_us.index.day==29) & (df_us.index.month==2)].index, inplace=True)

# average across years
yrAvg = np.zeros((365,1))
yrDiv = np.zeros((365,1)) # custom divider as missing few days end of last year
for currYr in range(2006,2023):
    stDt  = '%d-01-01' % (currYr)
    endDt = '%d-12-31' % (currYr)
    currSig = df_us[(df_us.index >= stDt) & (df_us.index <= endDt)].values
    yrAvg[0:len(currSig)] += currSig
    yrDiv[0:len(currSig)] += 1
yrAvg = np.divide(yrAvg,yrDiv)

# delete df_us as no longer need it
del df_us

# create xticks and xtick labels for the '1Yr' graphs
# getting dates for a single year (2010, arbitrary as long as not leap year!)
datList = pd.date_range(start='2010-01-01',end='2010-12-31')
xt = pd.date_range(start='2010-01-01',end='2010-12-01', freq='MS')
xtl = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

#%% plot the various data sets (orig, detrend, smooth, single cycle)

# create figure and gridspec for the figure
f = plt.figure(constrained_layout=False)
gs = f.add_gridspec(2,7)

# plot the comparison
ax1 = f.add_subplot(gs[0,:])
ax1.plot(df['Date'],df['Freq'],c='k',lw=0.75)
ax1.set_title('Original data')
ax1.set_ylim(20,120)
# ax1.set_yticks(np.arange(20,140,20))
ax1.set_yticks([])

ax2 = f.add_subplot(gs[1,0:3])
ax2.plot(df['Date'],df['Freq_dt'],c='k',lw=0.5, clip_on=False)
ax2.set_title('Detrended + HPF', fontsize='small')
ax2.set_ylim(-25,30)
# ax2.set_yticks(np.arange(-40,60,20))
ax2.set_yticks([])
ax2.xaxis.set_major_formatter(DateFormatter('%y')) # convert 2006 to 06
ax2.tick_params(axis='both',labelsize='x-small')

ax3 = f.add_subplot(gs[1,3:5])
ax3.plot(df['Date'],df['Freq_sm'],c='k',lw=0.5)
ax3.set_title('Smoothed (Illustrative only)', fontsize='small')
ax3.set_ylim(-25,30)
# ax3.set_yticks(np.arange(-20,30,10))
ax3.set_yticks([])
ax3.xaxis.set_major_formatter(DateFormatter('%y'))
ax3.tick_params(axis='both',labelsize='x-small')

ax4 = f.add_subplot(gs[1,5:7])
ax4.plot(datList,yrAvg,c='k',lw=0.75)
ax4.set_title('Average Year', fontsize='small')
ax4.set_ylim(-25,30)
# ax4.set_yticks(np.arange(-10,15,5))
ax4.set_yticks([])
ax4.tick_params(axis='y',labelsize='x-small') # can only use set_yticks for size if pass labels
ax4.grid(axis='x')
ax4.set_xticks(xt, labels=xtl, fontsize='x-small', rotation=90.0, family='monospace')

# adjust whitespace
plt.subplots_adjust(wspace=0.2,hspace=0.4)

# save the figure and show
if saveF: plt.savefig('./Fig1_UK.png',dpi=150, pad_inches=0)
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

# save the figure and show
if saveF: plt.savefig('./Fig2_UK.png',dpi=150, pad_inches=0)
plt.show()

# NOTES:
#   Seems to show a trend rising in winter months, falling in summer
#   No rise for christmas though!
#   If we do fft, expect mag. peak around 17 cycles, aka 1 cycle per year

#%% perform an fft

# calculate number of years data we have
nDays = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
nYrs = nDays/365.25

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
q25, q75 = np.percentile(w_amp, [25, 75])
mHeight = q75 + 1.5*(q75 - q25)
mDist = np.round(1.0/(2.0*freqs[1])) # freqs starts at 0, so freqs[1] gives incr.
peaks, _ = signal.find_peaks(w_amp, height=mHeight ,distance=mDist)

# calc. each peaks prominence and sort peaks in that order (most prom. first)
peaksPr = signal.peak_prominences(w_amp, peaks)[0]
peaksSr = [x for (y,x) in sorted(zip(peaksPr, peaks), key=lambda pair:pair[0],reverse=True)]

# create a function to calculate R^2 (coeff determination) and adj. R^2
sst = np.sum(np.square(df['Freq_dt']))
def adjR2(pred,k):
    ssr = np.sum(np.square(df['Freq_dt']-pred))
    r2 = 1.0 - (ssr/sst)
    adj_r2 = 1.0 - (((1.0-r2)*(N-1.0)) / (N-k-1.0))
    return [r2, adj_r2]

# calculate R^2 and adj. R^2 as you add more freqs to model
k=0
t = np.linspace(0.0,2*np.pi,n+1)[0:N]
pred = np.zeros_like(t)
all_r2 = np.zeros((len(peaks),2))
for cPeak in peaksSr:
    k += 1
    pred = pred + w_amp[cPeak]*np.cos(cPeak*t + w_ph[cPeak])
    all_r2[k-1,:] = adjR2(pred,k)

# based on adj. R^2 - keeping 3 peaks
#   first alone is ~20% var., then add about 5-6% [1], then 3-4% [2]
#   after that it's about 2% and less adding more in
#   more formal ways of doing it (likelihood etc), but it'll do!
peaks = peaksSr[0:3]

# plot the fourier spectrum
f, ax = plt.subplots()
ax.plot(freqs,w_amp,c='k',lw=0.75)
ax.set_xlim(0,16)
ax.set_ylim(0,3.0)
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
    '- then used adj. $R^2$ to select peaks of interest'))]
ax.legend(handles=[h1,h2], labels=[tStr[0],tStr[1]], loc='upper right', 
          fontsize='x-small', facecolor='whitesmoke', labelspacing=0.1, 
          fancybox=False, edgecolor='black', handletextpad=0.0,
          borderpad=0.5, borderaxespad=1)

# save the figure and show
if saveF: plt.savefig('./Fig3_UK.png',dpi=150, pad_inches=0)
plt.show()

# the 3 peaks...
#   first (1 cycle/yr) likely the dominant peak of interest
#   second (2 cycles/yr) likely just a harmonic
#   third (7 cycles/yr), no idea, interesting!

#%% explore the peaks (1+all)

# create the frequencies of interest (recreating 't' from above for clarity)
t = np.linspace(0.0,2*np.pi,n+1)[0:N]
p1 = w_amp[peaks[0]]*np.cos(peaks[0]*t + w_ph[peaks[0]])
p2 = w_amp[peaks[1]]*np.cos(peaks[1]*t + w_ph[peaks[1]])
p3 = w_amp[peaks[2]]*np.cos(peaks[2]*t + w_ph[peaks[2]])
pCom = p1 + p2 + p3

# calculate adjust coefficient of determination (adj R^2)
p1_ar2 = adjR2(p1,1)[1]
pCom_ar2 = adjR2(pCom,4)[1]

# calculate the frequencies of interest for just one year
t_1Dint = t[-1]/nDays # t[end] is cycle over all days, so calc. interval for 1 day
t_1Y = np.arange(0,365*t_1Dint,t_1Dint)
p1_1Y = w_amp[peaks[0]]*np.cos(peaks[0]*t_1Y + w_ph[peaks[0]])
pCom_1Y = (p1_1Y
           + w_amp[peaks[1]]*np.cos(peaks[1]*t_1Y + w_ph[peaks[1]])
           + w_amp[peaks[2]]*np.cos(peaks[2]*t_1Y + w_ph[peaks[2]]))

# create a subplot
f, ax = plt.subplots(ncols=2, nrows=2, constrained_layout=True,
                     gridspec_kw={'width_ratios':[2,1]})
f.suptitle('Depression Search Trends vs FFT Freq.', fontsize='medium')

# plot the main frequency of interest
ax[0,0].plot(df['Date'],df['Freq_dt'],c='k',lw=0.75, alpha=0.75, clip_on=False)
ax[0,0].plot(df['Date'],p1,c='r',lw=0.75)
ax[0,0].text(0.98,0.9,'adj. $R^2$ = %.2f' % (p1_ar2),fontsize='x-small',
             ha='right',transform=ax[0,0].transAxes)
ax[0,0].set_title('1 Cycle/yr',fontdict={'fontsize':'small'},loc='left')
ax[0,0].set_ylim(-20,20)
ax[0,0].set_yticks(np.arange(-20,30,10))
ax[0,0].xaxis.set_major_formatter(DateFormatter('%y')) 

# plot all frequencies of interest
ax[1,0].plot(df['Date'],df['Freq_dt'],c='k',lw=0.75, alpha=0.75, clip_on=False)
ax[1,0].plot(df['Date'],pCom,c='r',lw=0.75)
ax[1,0].text(0.98,0.9,'adj. $R^2$ = %.2f' % (pCom_ar2),fontsize='x-small',
             ha='right',transform=ax[1,0].transAxes)
ax[1,0].set_title('1+2+7 Cycles/yr',fontdict={'fontsize':'small'},loc='left')
ax[1,0].set_ylim(-20,20)
ax[1,0].set_yticks(np.arange(-20,30,10))
ax[1,0].xaxis.set_major_formatter(DateFormatter('%y')) 

#---

# plot the main frequency of interest (1Y)
ax[0,1].plot(datList,yrAvg,c='k',lw=1.0, alpha=0.75)
ax[0,1].plot(datList,p1_1Y,c='r',lw=1.0)
ax[0,1].set_title('Single Year',fontdict={'fontsize':'small'},loc='left')
ax[0,1].grid(axis='x')
ax[0,1].set_ylim(-10,10)
ax[0,1].set_yticks(np.arange(-10,15,5))
ax[0,1].set_xticks(xt, labels=xtl, fontsize='small', rotation=90.0, family='monospace')

# plot all frequencies of interest (1Y)
ax[1,1].plot(datList,yrAvg,c='k',lw=1.0, alpha=0.75)
ax[1,1].plot(datList,pCom_1Y,c='r',lw=1.0)
ax[1,1].set_title('Single Year',fontdict={'fontsize':'small'},loc='left')
ax[1,1].grid(axis='x')
ax[1,1].set_ylim(-10,10)
ax[1,1].set_yticks(np.arange(-10,15,5))
ax[1,1].set_xticks(xt, labels=xtl, fontsize='small', rotation=90.0, family='monospace')

# save the figure and show
if saveF: plt.savefig('./Fig4_UK.png',dpi=150, pad_inches=0)
plt.show()





























