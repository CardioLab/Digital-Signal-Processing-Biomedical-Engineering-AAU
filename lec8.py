# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:33:42 2026

@author: sschmidt
"""
from system1 import system1 

import numpy as np
import matplotlib.pyplot as plt


# make the unit sample
u = [0] * 100
u[10]=1
plt.plot(u)


h=system1(u)

plt.plot(h)



#%% FFT of the impluse respon, the system function 

H = np.fft.fft(h)



N=len(h);
n = np.arange(N)
plt.figure()
plt.stem( n/N*np.pi*2,np.abs(H))
plt.title('FFT')
plt.xlabel('Freqency (randians/sample)')
