#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:11:15 2023

@author: richard
"""


# import necessary stuff
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# set directory
os.chdir('/home/richard/Documents/Python/googleTrends/')

#%% read in the data

# skip 2 rows, treat next (row 0) as header, rename cols, parse 1st col as date
df = pd.read_csv('./data/Depression_2006.csv',skiprows=2,\
                 header=0,names=['Date','Freq'],parse_dates=[0])

