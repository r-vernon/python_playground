 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:18:13 2023

@author: richard

yep I know this is public - but I think it's interesting so don't care :p
decided to plot my weight coinciding with various medications:
    2x antidepressants (Zoloft + Prozac)
    2x ADHD meds (Elvanse + Concerta)
I do have ADHD, and did have an eating disorder which ADHD meds *mostly* address

"""

# import necessary stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# set dpi for figures
plt.rcParams["figure.dpi"] = 300

# create functions to calculate bmi from weight (in kgs) and vice versa
# hardcoded to my height of 1.8288m (6ft)
def kg2bmi(x):
    return x/(1.8288**2)
def bmi2kg(x):
    return x*(1.8288**2)

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

# interpolate missing days
weight_dat['iWeight'] = weight_dat.interpolate(method='time')

# set window size for gaussian window and calculate SD (based on FWHM) 
winSz = 30
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

# add secondary y-axis for bmi
secax = ax.secondary_yaxis('right', functions=(kg2bmi, bmi2kg))

# set axis lables
plt.xlabel('Date')
plt.ylabel('Weight (Kg)')
secax.set_ylabel('BMI')

# set main y axis limits
yL = 66
yU = 151
plt.ylim(yL,yU)

# fill areas of interest
ax.fill_betweenx((yL,yU), elvDate[0],   elvDate[1],   alpha=0.2, color='steelblue')
ax.fill_betweenx((yL,yU), sertDate[0],  sertDate[1],  alpha=0.2, color='tomato')
# ax.fill_betweenx((yL,yU), trazDate[0],  trazDate[1],  alpha=0.2, color='orange')
ax.fill_betweenx((yL,yU), fluoxDate[0], fluoxDate[1], alpha=0.2, color='gold')
ax.fill_betweenx((yL,yU), concDate[0],  concDate[1],  alpha=0.2, color='cadetblue')

# # add a dashed line for sleep apnea treatment
# plt.plot((apneaDate,apneaDate),(yL,yU),'k--')

# add text labels
ax.text(elvDate[0]+((elvDate[1]-elvDate[0])/2),yL+2,'Elvanse',ha='center',size='small')
ax.text(sertDate[0]+((sertDate[1]-sertDate[0])/2),yL+2,'Zoloft',ha='center',size='small')
ax.text(fluoxDate[0]+((fluoxDate[1]-fluoxDate[0])/2),yL+2,'Prozac',ha='center',size='small')
ax.text(concDate[0]+((concDate[1]-concDate[0])/2),yL+2,'Concerta',ha='center',size='small')

# add max weight
maxW = weight_dat['Weight'].agg(['idxmax','max'])
ax.text(maxW[0],maxW[1]+2,'Max %.1fKg' % (maxW[1]),ha='center',size='x-small')

# add start and current weight
ax.text(weight_dat.index[0],weight_dat.iat[0,0]-6,'   %.1fKg' % (weight_dat.iat[0,0]),ha='center',size='x-small')
ax.text(weight_dat.index[-1],weight_dat.iat[-1,0]-4,'%.1fKg   ' % (weight_dat.iat[-1,0]),ha='center',size='x-small')

# save the figure
plt.savefig('/home/richard/Documents/Python/weightGraph/weightGraph.png',dpi=150, pad_inches=0)

# show the glory
plt.show()

#%% plot a subset

# elvDat = weight_dat[elvDate[0]:elvDate[1]]

# fig, ax = plt.subplots()

# # plot the dots using actual data 
# plt.plot(elvDat.index,elvDat['Weight'],'o',markersize=1, color='black')

# # plot the line on top
# plt.plot(elvDat.index,elvDat['sWeight'],'-', alpha=0.8)

# # set axis lables
# plt.xlabel('Date')
# plt.ylabel('Weight (Kgs)')

# # set y axis limits
# plt.ylim(80,110)

# # show the glory
# plt.show()