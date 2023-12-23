# file: symmetries.py

import numpy as np

def lvsym(x, r, k, p=None):
    if p is None:
        p = 1.
    return k * np.exp(p) * x / (k - x + np.exp(p) * x)


