import numpy as np
from scipy.signal import lfilter

def system1(x):
    a = 1
    b = np.array([1, 1, 1]) / 3
    y = lfilter(b, a, x)
    return y