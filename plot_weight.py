#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:18:13 2023

@author: richard
"""

# import necessary stuff
import pandas as pd
import matplotlib.pyplot as plt

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

# remove duplicates (as some erroneous readings at the very start)
# weight_dat['isDup'] = weight_dat.duplicated(subset=['Weight'], keep='last')
test = weight_dat['Weight'].loc[weight_dat['Weight'].shift() != weight_dat['Weight']]