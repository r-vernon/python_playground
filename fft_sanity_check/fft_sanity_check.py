#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:22:34 2023

@author: richard
"""

# import relevant stuff
import heapq
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys

# set dpi for figures
plt.rcParams["figure.dpi"] = 300

#%%

# setup details for signal
nPts = 1024 # number of time points
n = 2*nPts  # n for fft (zero padding if n > nPts)
t = np.linspace(0.0,2.0*np.pi,nPts+1)[0:-1]

# setup details for sine waves
sAmp  = [0.2, 0.47, 0.32] # amplitude
sFreq = [12., 7.,   3.]   # frequency
sPh   = [27., 56.,  38.]  # phase (degrees)
sPh = np.deg2rad(sPh)

# get number of sine waves
nFreq = len(sFreq)

# check entered enough properties
if len(sAmp) != nFreq or len(sPh) != nFreq:
    sys.exit('length of sine wave properties not equal')  

# just make sure they're sorted (by frequency)
#   zip them together so sFreq[0] zipped to sAmp[0] & sPh[0] as tuple
#   sort them (by default first in tuple, sFreq)
#   unpack them (*Fcn) and rezip, so sFreq[0] now zipped to sFreq[1]:SFreq[n]
#   then convert back to lists, rather than tuples
sFreq, sAmp, sPh = [list(x) for x in zip(*sorted(zip(sFreq, sAmp, sPh)))]

# calculate expected frequency bins based on zero padding
eFreq = [np.round((n/nPts) * x) for x in sFreq]

# create signal
s = np.zeros_like(t)
for inc in range(nFreq):
    s = s + (sAmp[inc]*np.sin(sFreq[inc]*t + sPh[inc]))

# perform the fft, get amp. and phase
w = np.fft.rfft(s,n=n)
w_amp = 2.0/nPts * np.abs(w[:n//2 +1])
w_ph  = np.angle(w[:n//2] +1)

# use outlier heuristic to set max peak height
# base minDist on minimum expected distance from eFreq
q25, q75 = np.percentile(w_amp, [25, 75])
mHeight = q75 + 1.5*(q75 - q25)
mDist = min(np.diff(eFreq))/2

# find the peaks
p,pProp = signal.find_peaks(w_amp,height=mHeight,distance=mDist)
pHeights = pProp['peak_heights']

# find the n biggest (target) peaks
tPeaks = heapq.nlargest(len(sFreq), zip(pHeights, range(len(pHeights))))
tPeaks = sorted([x[1] for x in tPeaks])

# get the properties of the frequencies of those peaks
tFreq = p[tPeaks]
tAmp  = w_amp[tFreq]
tPh   = w_ph[tFreq]

# create data for table
# converting frequency in case of zero padding, and converting phase from cos to sin
cols = ['Freq.','Est. Freq.', 'Amp.', 'Est. Amp.', 'Phase', 'Est. Phase']
rows = np.column_stack((sFreq,tFreq/(n/nPts),sAmp,tAmp,sPh,tPh+(np.pi/2)))
cell_text = []
for inc in range(nFreq):
    cell_text.append(['%1.2f' % (x) for x in rows[inc,:]])

# plot the fourier spectrum
f, ax = plt.subplots()
ax.set_title('FFT Amplitude Spectrum')
ax.set_xlabel('Frequency')
ax.set_ylabel('Amplitude')
ax.plot(np.arange(0,len(w_amp)),w_amp,'k')
h = ax.plot(tFreq,tAmp,'r.',label='Detected Peaks')
ax.legend(handles=h, loc='upper right', fontsize='x-small')
ax.set_xlim(0,np.ceil(max(eFreq)*1.2))
ax.set_ylim(0,None)
plt.show()

# recreate signal (using t based on n, not nPts)
t2 = np.linspace(0.0,2.0*np.pi,n+1)[0:nPts]
s2 = np.zeros_like(t2)
for inc in range(nFreq):
    s2 = s2 + (tAmp[inc]*np.cos(tFreq[inc]*t2 + tPh[inc]))

# compare the signals
f, ax = plt.subplots()
ax.set_title('Original vs. Estimated Signal')
ax.plot(t,s,'k-',label='Original')
ax.plot(t,s2,'r--',label='Estimated')
ax.legend(loc='upper right', fontsize='x-small')
ax.set_xticks([])
tbl = ax.table(cellText=cell_text, cellLoc='center',colLabels=cols)
plt.subplots_adjust(bottom=0.2)
plt.show()



