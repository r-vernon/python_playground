 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:18:13 2023

@author: richard
"""

# import necessary stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# set dpi for figures
plt.rcParams["figure.dpi"] = 300

#%% import file

# import the file
weight_dat = pd.read_csv('/home/richard/Documents/Python/weightGraph/weight.csv', \
                         usecols=['Date','Weight (kg)'])
    
# rename columns
weight_dat.rename(columns={'Weight (kg)':'Weight'},inplace=True)

# make sure dates are set as dates!
weight_dat["Date"] = pd.to_datetime(weight_dat["Date"])

# reverse ordering so chronological
weight_dat.sort_values(by=["Date"],inplace=True)

# delete rows before 25/05/16 as likely erroneous
weight_dat = weight_dat[~(weight_dat['Date'] < '2016-05-25')]

# normalise the dates (remove times) as we don't care what time we weighed ourself
# helps duplicate check later
weight_dat['Date'] = weight_dat['Date'].dt.normalize()

# remove duplicates (same day, same weight) - below does equivelant
# weight_dat.drop_duplicates(inplace=True)

# for any duplicates (same day, multiple weights) average
# also sets date as index
weight_dat = weight_dat.groupby('Date').mean()

#%% smooth the weight data

# upsample so every day is represented
weight_dat = weight_dat.asfreq(freq='D')

# # just demonstrate vast majority of cases it's one missing day at most
# # interpolation should be fine!
# nullIdx = np.arange(0,len(weight_dat))
# nullIdx = np.diff(nullIdx[weight_dat['Weight'].isnull()])
# plt.hist(nullIdx,bins=np.arange(1,10))

# interpolate missing days (sci-pi cubic spline interpolation)
weight_dat['iWeight'] = weight_dat.interpolate(method='spline', order=3)

# set window size and calculate SD (based on FWHM) 
winSz = 7
winSD = winSz/(2*np.sqrt(2*np.log(2)))

# smooth the data
weight_dat['sWeight'] = weight_dat['iWeight'].rolling(window=winSz, min_periods=1, \
                                     win_type='gaussian',center=True).mean(std=winSD)

#%% set some key dates to flag

# time on elvanse
elvDate = (datetime(2017,12,5),datetime(2020,1,31))

# time on sertraline
sertDate = (datetime(2020,2,14),datetime(2021,2,20))

# # time on trazadone not including as so minimal
# trazDate = (datetime(2021,8,18),datetime(2021,9,15))

# time on fluoxetine
fluoxDate = (datetime(2021,9,20),datetime(2022,3,18))

# time on concerta
concDate = (datetime(2022,9,28),weight_dat.index.max())

# sleep apnea diagnosed
apneaDate = (datetime(2020,10,19))

#%% plot the data

fig, ax = plt.subplots()

# plot the dots using actual data 
plt.plot(weight_dat.index,weight_dat['Weight'],'o',markersize=1, color='black')

# plot the line on top
plt.plot(weight_dat.index,weight_dat['sWeight'],'-', alpha=0.8)

# set axis lables
plt.xlabel('Date')
plt.ylabel('Weight (Kgs)')

# set y axis limits
plt.ylim(70,150)

# fill areas of interest
ax.fill_betweenx((70,150), elvDate[0],   elvDate[1],   alpha=0.2, color='steelblue')
ax.fill_betweenx((70,150), sertDate[0],  sertDate[1],  alpha=0.2, color='tomato')
# ax.fill_betweenx((70,150), trazDate[0],  trazDate[1],  alpha=0.2, color='orange')
ax.fill_betweenx((70,150), fluoxDate[0], fluoxDate[1], alpha=0.2, color='gold')
ax.fill_betweenx((70,150), concDate[0],  concDate[1],  alpha=0.2, color='cadetblue')

# # add a dashed line for sleep apnea treatment
# plt.plot((apneaDate,apneaDate),(70,150),'k--')

# show the glory
plt.show()