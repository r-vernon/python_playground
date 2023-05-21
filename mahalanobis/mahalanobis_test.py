#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:14:28 2022

@author: richard
"""

# import shiz
import numpy as np
import matplotlib.pyplot as plt

#%% first lets create some correlated data
# corr(x,y) = cov(x,y)/(sqrt(var(x)*var(y)))

# set random seed for consistency
np.random.seed(42)

# set target correlation
targCorr = 0.6 

# set properties of data
nSamples = 1000 # Number of samples
xy_M = (6,6)    # Mean (centre) of data
xy_SD = (2,1)   # SD (spread) of data

# calculate required covariance
xy_cov = targCorr*(xy_SD[0]*xy_SD[1])

# create target covariance matrix (probably better ways to do this!)
covMat = np.full((2,2),xy_cov)
np.fill_diagonal(covMat,np.square(xy_SD))

# perform cholesky decomposition
# decomposes Cov into lower triangle matrix L, where L.L' = Cov
L = np.linalg.cholesky(covMat)

# why this works:
# - let C be target Cov, and L.L' = C
# - let X be uncorrelated variables so X.X' = I
# - let Y = LX (as we're doing below)
# - (note that (AB)' = B'A'
# - Y.Y' = (LX).(LX)' = LXX'L' = LIL' = LL' = C
# - therefore LX gives variables with target CovMat

# create uncorrelated normal data
# multiply by decomposition
# shift centre
xy = np.random.standard_normal((2,nSamples))
xy = np.dot(L,xy)
xy = xy + np.array(xy_M).reshape(2,1)

# correlate to check
xy_r = np.corrcoef(xy[0,:],xy[1,:])[0,1]

# calculate actual centre (not target)
xy_cent = np.mean(xy,1)

#%% plot that shit

fig, ax = plt.subplots()

ax.scatter(xy[0,:],xy[1,:],s=48,c='black',marker='.',alpha=0.6)
ax.scatter(xy_cent[0],xy_cent[1],s=40,c='red',marker='x')

ax.set(title='Correlated Data', \
       xlim=(-2,14), xticks=np.arange(-2,15,2), xlabel='X Data', \
       ylim=(-2,14),  ylabel='Y Data')
ax.set_aspect('equal', adjustable='box')

plt.show()

#%% create two potential data points
# one along trendline at reasonable (2SD) distance
# one at same dist., rotated 90deg CCW - outlier!

# line of best fit for data
m1,c1 = np.polyfit(xy[0,:], xy[1,:], 1)

# calculate lines orthogonal to that one
m2 = -1/m1                # when orth. m1*m2=-1
c2 = xy[1,:] - m2*xy[0,:] # y = mx+c, so c = y-mx

# calculate intercept points for each of those lines (basic algebra)
icpt_xy = np.empty_like(xy)
icpt_xy[0,:] = (c1-c2) / (m2-m1)    # x
icpt_xy[1,:] = m1*icpt_xy[0,:] + c1 # y

# calculate distance of each intercept point from centre, along trendline
# transpose for subtraction as trailing axes apparently need same dim. 
# - xy_cent is (2,) so icpt_xy must be (*,2)
icpt_dist = (icpt_xy.T - xy_cent).T # x, y deltas from centre
icpt_dist = np.sqrt(np.sum(np.square(icpt_dist),axis=0)) # calc. hypot

# preallocate potential new points
pts = np.empty((2,2))

# calculate point two stdevs along trendline from centre
targDist = 2*np.std(icpt_dist) # target distance
m1_ang = np.arctan(m1)         # angle between trendline and x-axis
pts[0,:] = xy_cent + targDist*np.array((np.cos(m1_ang), np.sin(m1_ang)))

# create rotated point
th = np.deg2rad(-90)
c, s = np.cos(th), np.sin(th)
rotMat = np.array(((c, -s), (s, c)))
pts[1,:] = xy_cent + ((pts[0,:]-xy_cent) @ rotMat)

#%% plot those markers

fig, ax = plt.subplots()

ax.scatter(xy[0,:],xy[1,:],s=48,c='black',marker='.',alpha=0.6)
#ax.plot(xy[0,:],m1*xy[0,:]+c1)
ax.scatter(xy_cent[0],xy_cent[1],s=40,c='red',marker='x')
ax.scatter(pts[0,0],pts[0,1],s=40,c='blue',marker='o')
ax.scatter(pts[1,0],pts[1,1],s=40,c='green',marker='o')

ax.set(title='Correlated Data', \
       xlim=(-2,14), xticks=np.arange(-2,15,2), xlabel='X Data', \
       ylim=(-2,14),  ylabel='Y Data')
ax.set_aspect('equal', adjustable='box')

plt.show()

#%% create a euclidian distance plot

# create meshgrid based on axis limits above
xlin = np.linspace(-2,14,100)
ylin = np.linspace(-2,14,100)
mgridX,mgridY = np.meshgrid(xlin,ylin)

# calculate euclidian distance
eucDist = np.sqrt(np.square(mgridX-xy_cent[0]) + np.square(mgridY-xy_cent[1]))

#%% plot that shit again

fig, ax = plt.subplots()

ax.scatter(xy[0,:],xy[1,:],s=48,c='black',marker='.',alpha=0.6)
ax.contourf(mgridX,mgridY,eucDist,1000,cmap=plt.cm.plasma_r,alpha=0.8,antialiased='true')
ax.scatter(xy_cent[0],xy_cent[1],s=40,c='red',marker='x')

ax.set(title='Euclidian Distance Map', \
       xlim=(-2,14), xticks=np.arange(-2,15,2), xlabel='X Data', \
       ylim=(-2,14),  ylabel='Y Data')
ax.set_aspect('equal', adjustable='box')

plt.show()

#%% create a mahalanobis distance plot

# calculate inverse of the covariance matrix
inv_covMat = np.linalg.inv(np.cov(xy))

# vectorise and demean earlier meshgrid
mgridXY = np.vstack((np.matrix.flatten(mgridX-xy_cent[0]),\
                     np.matrix.flatten(mgridY-xy_cent[1])))

mahaDist = np.diagonal(mgridXY.T @ inv_covMat @ mgridXY)
mahaDist = np.reshape(np.sqrt(mahaDist),(100,100))

#%% plot that shit once more

fig, ax = plt.subplots()

ax.scatter(xy[0,:],xy[1,:],s=48,c='black',marker='.',alpha=0.6)
ax.contourf(mgridX,mgridY,mahaDist,1000,cmap=plt.cm.plasma_r,alpha=0.8,antialiased='true')
ax.scatter(xy_cent[0],xy_cent[1],s=40,c='red',marker='x')

ax.set(title='Mahalanobis Distance Map', \
       xlim=(-2,14), xticks=np.arange(-2,15,2), xlabel='X Data', \
       ylim=(-2,14),  ylabel='Y Data')
ax.set_aspect('equal', adjustable='box')

plt.show()
