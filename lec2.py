# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:33:42 2026

@author: sschmidt
"""
from system1 import system1 

import matplotlib.pyplot as plt


# make the unit sample
u = [0] * 100
u[10]=1
plt.plot(u)


h=system1(u)

plt.plot(h)


