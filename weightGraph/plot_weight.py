#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:18:13 2023

@author: richard

yep I know this is public - but I think it's interesting so don't care :p
decided to plot my weight coinciding with various medications:
    2x antidepressants (Zoloft + Prozac)
    2x ADHD meds (Elvanse + Concerta)

"""

# import necessary stuff
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
from datetime import datetime, timedelta

# set dpi for figures
plt.rcParams["figure.dpi"] = 300

# change directory to location of script
os.chdir(os.path.dirname(__file__))


# create functions to calculate bmi from weight (in kgs) and vice versa
# hardcoded to my height of 1.8288m (6ft)
def kg2bmi(x):
    return x / (1.8288 ** 2)


def bmi2kg(x):
    return x * (1.8288 ** 2)


# ------------------------------------------------------------------------------
# import file

# import the file
weight_dat = pd.read_csv('./weight.csv', usecols=['Date', 'Weight (kg)'])

# rename columns
weight_dat.rename(columns={'Weight (kg)': 'Weight'}, inplace=True)

# make sure dates are set as dates!
weight_dat["Date"] = pd.to_datetime(weight_dat["Date"])

# reverse ordering so chronological
weight_dat.sort_values(by=["Date"], inplace=True)

# delete rows before 25/05/16 as likely erroneous
weight_dat = weight_dat[~(weight_dat['Date'] < '2016-05-25')]

# normalize the dates (remove times) as we don't care what time we weighed ourself
# helps duplicate check later
weight_dat['Date'] = weight_dat['Date'].dt.normalize()

# for any duplicates (same day, multiple weights) average
# also sets date as index
weight_dat = weight_dat.groupby('Date').mean()

# ------------------------------------------------------------------------------
# smooth the weight data

# up-sample so every day is represented
weight_dat = weight_dat.asfreq(freq='D')

# just demonstrate in the vast majority of cases it's one missing day at most
# interpolation should be fine!
# nullIdx = np.arange(0,len(weight_dat))
# nullIdx = np.diff(nullIdx[weight_dat['Weight'].isnull()])
# plt.hist(nullIdx,bins=np.arange(1,10))

# interpolate missing days
weight_dat['iWeight'] = weight_dat.interpolate(method='time')

# set window size for gaussian window and calculate SD (based on FWHM) 
winSz = 30
winSD = winSz / (2 * np.sqrt(2 * np.log(2)))

# smooth the data
weight_dat['sWeight'] = weight_dat['iWeight'].rolling(
    window=winSz, min_periods=1, win_type='gaussian', center=True
    ).mean(std=winSD)

# ------------------------------------------------------------------------------
# set some key dates to flag

# time on elvanse
elvDate = (datetime(2017, 12, 5), datetime(2020, 1, 31))

# time on sertraline
sertDate = (datetime(2020, 2, 14), datetime(2021, 2, 20))

# # time on trazadone not including as so minimal
# trazDate = (datetime(2021,8,18),datetime(2021,9,15))

# time on fluoxetine
fluoxDate = (datetime(2021, 9, 20), datetime(2022, 3, 18))

# time on concerta
concDate = (datetime(2022, 9, 28), weight_dat.index.max())

# sleep apnea diagnosed
apneaDate = (datetime(2020, 10, 19))

# work stress started (day email sent about mrMeshPy mesh issue)
# (get weight on this day too, plotting purposes)
wStressDate = (datetime(2018, 4, 10))
wStressWt = weight_dat.loc[wStressDate, 'iWeight']

# ------------------------------------------------------------------------------
# calculate rates of change
#
# for ADHD meds, assuming instant impact
# for antidepressants, assuming 3wk lag (onset+offset)
#
# note: date2num is days since epoch, so gradient of polyfit will be expected
#       change per day

# set 1day and 21 day time deltas
d1 = timedelta(days=1)
d21 = timedelta(days=21)


# create function to calculate gain per year from data
def calc_gain(weight_vals):
    x = md.date2num(weight_vals.index)
    z = np.polyfit(x, weight_vals.values, 1)
    return z[0] * 365.25


# normal weight gain (before elvanse)
y = weight_dat.loc[:elvDate[0] - d1, 'Weight'].dropna()
nGain = calc_gain(y)

# elvanse weight gain
y = weight_dat.loc[elvDate[0]:elvDate[1], 'Weight'].dropna()
eGain = calc_gain(y)

# sertraline weight gain
y = weight_dat.loc[sertDate[0] + d21:sertDate[1] + d21, 'Weight'].dropna()
sGain = calc_gain(y)

# fluoxetine weight gain
y = weight_dat.loc[fluoxDate[0] + d21:fluoxDate[1] + d21, 'Weight'].dropna()
fGain = calc_gain(y)

# concerta weight gain
y = weight_dat.loc[concDate[0]:concDate[1], 'Weight'].dropna()
cGain = calc_gain(y)

# ------------------------------------------------------------------------------
# plot the data

fig, ax = plt.subplots()

# plot the dots using actual data 
plt.plot(weight_dat.index, weight_dat['Weight'], 'o',
         markersize=1, color='black')

# plot the line on top
plt.plot(weight_dat.index, weight_dat['sWeight'], '-', alpha=0.8)

# add x-axis minor tick marks for months
ax.xaxis.set_minor_locator(md.MonthLocator())

# add secondary y-axis for bmi
secax = ax.secondary_yaxis('right', functions=(kg2bmi, bmi2kg))

# add axis gridlines 
ax.grid(axis='both', which='major', alpha=0.4, linewidth=0.6)
ax.grid(axis='x', which='minor', alpha=0.2, linewidth=0.3)

# set axis lables
plt.xlabel('Date')
plt.ylabel('Weight (Kg)')
secax.set_ylabel('BMI')

# set main y axis limits
yL = 66
yU = 151
plt.ylim(yL, yU)

# fill areas of interest
ax.fill_betweenx((yL, yU), elvDate[0], elvDate[1], alpha=0.2, fc='steelblue')
ax.fill_betweenx((yL, yU), sertDate[0], sertDate[1], alpha=0.2, fc='tomato')
# ax.fill_betweenx((yL,yU), trazDate[0],  trazDate[1],  alpha=0.2, fc='orange')
ax.fill_betweenx((yL, yU), fluoxDate[0], fluoxDate[1], alpha=0.2, fc='gold')
ax.fill_betweenx((yL, yU), concDate[0], concDate[1], alpha=0.2, fc='cadetblue')

# # add an annotation for work stress date start
ax.annotate('work stress started', xy=(wStressDate, wStressWt),
            size='xx-small', ha='left', va='top', style='italic',
            xytext=(wStressDate + timedelta(days=100), wStressWt - 4.2),
            arrowprops={'color': 'orangered', 'arrowstyle': '-|>',
                        'connectionstyle': 'angle3,angleA=0,angleB=90',
                        'relpos': (0., 0.5)})

# add a dashed line for sleep apnea treatment
# plt.plot((apneaDate,apneaDate),(yL,yU),'k--')

# add text labels
ax.text(weight_dat.index[0] + ((elvDate[0] - d1 - weight_dat.index[0]) / 2),
        yL + 1.4, f"'Normal'\n({nGain:.1f}Kg/yr)", ha='center', size='x-small')
ax.text(elvDate[0] + ((elvDate[1] - elvDate[0]) / 2),
        yL + 1.4, f'Elvanse\n({eGain:.1f}Kg/yr)', ha='center', size='x-small')
ax.text(sertDate[0] + ((sertDate[1] - sertDate[0]) / 2),
        yL + 1.4, f'Zoloft\n({sGain:.1f}Kg/yr*)', ha='center', size='x-small')
ax.text(fluoxDate[0] + ((fluoxDate[1] - fluoxDate[0]) / 2),
        yL + 1.4, f'Prozac\n({fGain:.1f}Kg/yr*)', ha='center', size='x-small')
ax.text(concDate[0] + ((concDate[1] - concDate[0]) / 2),
        yL + 1.4, f'Concerta\n({cGain:.1f}Kg/yr)', ha='center', size='x-small')
plt.gcf().text(0.95, 0.02, '*assuming 3wk lag', ha='right', size='x-small')

# add max weight
maxW = weight_dat['Weight'].agg(['idxmax', 'max'])
ax.text(maxW[0], maxW[1] + 2, f'Max {maxW[1]:.1f}Kg', ha='center', size='x-small')

# add start and current weight
ax.text(weight_dat.index[0], weight_dat.iat[0, 0] - 5,
        f'   {weight_dat.iat[0, 0]:.1f}Kg', ha='center', size='x-small')
ax.text(weight_dat.index[-1], weight_dat.iat[-1, 0] - 4,
        f'{weight_dat.iat[-1, 0]:.1f}Kg   ', ha='center', size='x-small')

# save the figure
# plt.savefig('./weightGraph.png', dpi=150, pad_inches=0)

# show the glory
plt.show()

# ------------------------------------------------------------------------------
# plot a subset

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
