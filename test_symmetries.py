# file: test_symmetries.py

from symmetries import *
from stochastic_systems import *
import numpy as np
import pdb


def identity_func(x, r, k):
    return x

def test_logistic_sym():
    r = np.array([1.])
    k = np.array([1000.])
    alpha = ([[1.]])
    sigma = np.array([0.])
    gamma = 1.
    params = [r, k, alpha, sigma, gamma]
    t_max = 2.
    num_times = 10
    init_x = np.array([5.])
    
    data = np.round(random_time_intervention(init_x, params, t_max, identity_func,
                                          num_times=num_times), 2)
    assert data[0,1] == data[-1,1]
    assert np.round(np.mean(data[:,1]), 2) == data[0,1]
    
    data = np.round(random_time_intervention(init_x, params, t_max, lvsym,
                                          num_times=num_times), 2)
    assert data[0,1] == data[-1,1]
    assert np.round(np.mean(data[:,1]), 2) == data[0,1]
