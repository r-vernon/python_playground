 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:18:13 2023

@author: richard
"""

# import necessary stuff
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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

# remove duplicates
weight_dat.drop_duplicates(inplace=True)

#%% set some key dates to flag

# time on elvanse
elvDate = (datetime(2017,12,5),datetime(2020,2,6))

# time on sertraline
sertDate = (datetime(2020,2,14),datetime(2021,2,20))

# time on trazadone
trazDate = (datetime(2021,8,18),datetime(2021,9,15))

# time on fluoxetine
fluoxDate = (datetime(2021,9,20),datetime(2022,3,18))

# time on concerta
concDate = (datetime(2022,9,28),weight_dat['Date'].iloc[-1])

# sleep apnea diagnosed
apneaDate = (datetime(2020,10,19))

#%% plot the data

fig, ax = plt.subplots()

# plot the dots
#plt.plot(weight_dat['Date'],weight_dat['Weight'],'o',markersize=2, color='black')

# plot the line on top
plt.plot(weight_dat['Date'],weight_dat['Weight'],'-')

# set axis lables
plt.xlabel('Date')
plt.ylabel('Weight (Kgs)')

# set y axis limits
plt.ylim(70,150)

# fill areas of interest
ax.fill_betweenx((70,150), elvDate[0],   elvDate[1],   alpha=0.2)
ax.fill_betweenx((70,150), sertDate[0],  sertDate[1],  alpha=0.2)
ax.fill_betweenx((70,150), trazDate[0],  trazDate[1],  alpha=0.2)
ax.fill_betweenx((70,150), fluoxDate[0], fluoxDate[1], alpha=0.2)
ax.fill_betweenx((70,150), concDate[0],  concDate[1],  alpha=0.2)

# show the glory
plt.show()