#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:52:11 2023

@author: richard
"""

# need pytrends for this little experiment
# !pip install pytrends

# import necessary stuff
from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt

#%% get the data

# connect to google (timezone is 0 for GMT)
pytrends = TrendReq(hl='en', tz=0)

# set keywords we're interested in
kw_list = ['ADHD', 'depression', 'anxiety', 'stress']

# optional - get suggested keywords and store as dataframe
# suggKW = pd.DataFrame(pytrends.suggestions(keyword=kw_list[0]))

# set timeframe and location to search in
# can find location (geo) codes from URL on google trends website
pytrends.build_payload(kw_list, cat=0, timeframe='2018-01-01 2023-01-01', geo='GB', gprop='')


#%% explore the data

# interest over time
dat = pytrends.interest_over_time()
dat.reset_index()

#%% plot

plt.plot(dat.index,dat['ADHD'],color='b')
plt.plot(dat.index,dat['depression'],color='r')
plt.plot(dat.index,dat['anxiety'],color='g')
plt.plot(dat.index,dat['stress'],color='k')

# show the glory
plt.show()