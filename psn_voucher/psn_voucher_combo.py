#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:18:57 2023

@author: richard
"""

# import necessary stuff
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint

# set dpi for figures
plt.rcParams["figure.dpi"] = 300

# change directory to location of script
os.chdir(os.path.dirname(__file__))

#%%----------------------------------------------------------------------------
# for now let's just hardcode pricing

# set voucher options (manually for now)
vOpts = np.array([[05.,04.85], [10.,09.85], [15.,13.85], [20.,18.85], 
                  [25.,22.85], [30.,26.85], [32.,27.85], [35.,30.85], 
                  [40.,35.85], [40.,34.85], [45.,37.85], [50.,42.85], 
                  [80.,66.85], [84.,69.85], [90.,75.85], [100.,84.85]])
                              
# add card option (so don't use vouchers alone)
useCard = 0
if useCard:
    vOpts = np.insert(vOpts, 0, [1.,1.], axis=0)

#%%----------------------------------------------------------------------------
# build model to find 'x'
# see: 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html
# https://docs.scipy.org/doc/scipy/tutorial/optimize.html#tutorial-optimize-milp

# set goal price
targSpend = 83

# split voucher options for readability
val = vOpts[:,0] # value of voucher
cst = vOpts[:,1] # cost of voucher

# set integrality constraint (all x vals need to be integers)
integrality = np.full_like(val, True)
if useCard: integrality[0] = False

# don't need to set bounds as default bounds are 0 and inf, which we want!

# set constraints (such that lb <= A.dot(x) <= ub)
constraints = LinearConstraint(A=val, lb=targSpend, ub=np.inf)

# run model
res = milp(c=cst, integrality=integrality, constraints=constraints)

# check result status
if res.success: 
    
    print('Optimal solution found:')
    
    print('Target: £{targSpend:.2f}, Cost: £{t_cst:.2f}, Value: £{t_val:.2f}'.format(
        targSpend=targSpend, t_cst=res.fun, t_val=val@res.x))
    
    # show solution
    st = 0
    txt = ''
    if useCard:
        st += 1
        if res.x[0] > 0:
            txt += '£{amt:.2f} on card, '.format(amt=res.x[0])
    for inc in range(st,len(val)):
        if res.x[inc] > 0:
            txt += '{x:.0f}x£{v:.2f} (£{c:.2f}), '.format(
                x=res.x[inc], v=val[inc], c=res.x[inc]*cst[inc])
    txt = txt.rstrip(', ')
    print(txt)
else:
    print('No solution found')

#%%----------------------------------------------------------------------------
# build a graph too

trg = range(5,151,1)
trg_cst = np.zeros(len(trg))
trg_val = np.zeros(len(trg))
inc = 0

for curr_trg in trg:
    constraints = LinearConstraint(A=val, lb=curr_trg, ub=np.inf)
    res = milp(c=cst, integrality=integrality, constraints=constraints)
    trg_cst[inc] = res.fun
    trg_val[inc] = val@res.x
    inc += 1

# calculate saving
trg_sav = (trg_cst/trg_val)

# plot the result
plt.plot(trg,trg_val,'r:')
plt.plot(trg,trg_cst,'k')
plt.show()
plt.plot(trg,trg_sav,'k')
plt.show()
