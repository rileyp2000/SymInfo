# file: test_stochastic_system.py

from stochastic_systems import *
import numpy as np
import matplotlib.pyplot as plt
import pdb

def test_point_sample():
    r = np.array([1., 1.])
    k = np.array([100., 100.])
    alpha = ([[1., -0.2], [-0.2, 1.]])
    sigma = np.array([0.1, 0.1])
    gamma = 1.

    # TEST 1
    init_x = [np.array([5., 5.]), np.array([40., 40.])]
    times = np.linspace(0., 6., 50)
    samples1 = point_sample(init_x, [r, k, alpha, sigma, gamma], times)

    for ss in samples1:
        plt.plot(times, ss[:,0], 'b.')
        plt.plot(times, ss[:,1], 'g.')

    # TEST 7
    init_x = [np.array([5., 5.]), np.array([40., 40.])]
    times = np.linspace(0., 6., 50)
    samples7 = point_sample(init_x, [r, k, alpha, sigma, 1.1], times)
    plt.figure()
    for ss in samples7:
        plt.plot(times, ss[:,0], 'r.')
        plt.plot(times, ss[:,1], 'y.')

    # TEST 2
    init_x = []
    for ii in range(100):
        init_x.append(np.abs(np.random.normal(5., 0.5, 2)))
    times = np.array([0., 6.])
    samples2 = point_sample(init_x, [r, k, alpha, sigma, gamma], times,
                            num_samples =1)

    plt.figure()
    for xx in init_x:
        plt.plot(0., xx[0], 'b.')
        plt.plot(0., xx[1], 'g.')
    for ss in samples2:
        plt.plot(times, ss[:,0], 'b.')
        plt.plot(times, ss[:,1], 'g.')

    # TEST 3
    init_x = [np.array([5., 5.])]
    times = np.sort(np.random.rand(50) * 6.)
    samples3 = point_sample(init_x, [r, k, alpha, sigma, gamma], times)
    
    plt.figure()
    for ss in samples3:
        plt.plot(times, ss[:,0], 'b.')
        plt.plot(times, ss[:,1], 'g.')

   # TEST 4
    r = np.array([1.])
    k = np.array([100.])
    alpha = ([[1.]])
    sigma = np.array([0.1])
    init_x = [np.array([5.])]
    times = np.linspace(0., 6., 50)
    samples4 = point_sample(init_x, [r, k, alpha, sigma, gamma], times)
    
    plt.figure()
    for ss in samples4:
        plt.plot(times, ss, 'k.')

    # TEST 5
    init_x = []
    for ii in range(100):
        init_x.append(np.array([np.abs(np.random.normal(5., 0.5))]))
    times = np.array([0., 6.])
    samples5 = point_sample(init_x, [r, k, alpha, sigma, gamma], times,
                            num_samples=1)

    plt.figure()
    for xx in init_x:
        plt.plot(0., xx, 'k.')
    for ss in samples5:
        plt.plot(times, ss, 'k.')

    # TEST 6
    init_x = [np.array([5.])]
    times = np.sort(np.random.rand(50) * 6.)
    samples6 = point_sample(init_x, [r, k, alpha, sigma, gamma], times)
    
    plt.figure()
    for ss in samples6:
        plt.plot(times, ss, 'k.')


    plt.show()

def lvsym(x, r, k, p=None):
    if p is None:
        p = 1. / k ** 2
    return k * np.exp(p * k ** 2) * x / np.sqrt(k ** 2 - x ** 2 + np.exp(2 * p * k ** 2) * x ** 2

test_random_time_intervention():
    pass
