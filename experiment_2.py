# file: experiment_2.py

""" No model, or no symmetries known. Compare MI using under an arbitrary 
    intervention.
"""

import numpy as np
from stochastic_systems import *
from stat_methods import *
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import sys
import getopt

def sym_func(x, r, k):
    return x ** (-1) 

def main(num_trials):
    r = np.array([1.5])
    k = np.array([500.])
    alpha = ([[1.]])
    sigma = np.array([0.])
    gamma_A = 1.
    params_A = [r, k, alpha, sigma, gamma_A]
    gamma_B =  1.1
    params_B = [r, k, alpha, sigma, gamma_B]
    params_C = [0.5*r, k, alpha, sigma, gamma_A]
    t_max = 2.
    num_times = 40
    init_x = np.array([5.])

    mi_A = []
    mi_B = []
    mi_C = []
    
    for ii in range(num_trials):
        data_A = np.round(random_time_intervention(init_x, params_A, t_max,
                                                   sym_func, num_times=num_times), 2)
        data_B = np.round(random_time_intervention(init_x, params_B, t_max, sym_func,
                                          num_times=num_times), 2)
        data_C = np.round(random_time_intervention(init_x, params_C, t_max, sym_func,
                                          num_times=num_times), 2)
        mi_A.append(mutual_info_regression(data_A[:,0].reshape(-1,1),
                                           data_A[:,1]))
        mi_B.append(mutual_info_regression(data_B[:,0].reshape(-1,1),
                                           data_B[:,1]))
        mi_C.append(mutual_info_regression(data_C[:,0].reshape(-1,1),
                                           data_C[:,1]))

    mi_A = np.array(mi_A)
    mi_B = np.array(mi_B)
    mi_C = np.array(mi_C)

    mu_A = np.mean(mi_A)
    mu_B = np.mean(mi_B)
    mu_C = np.mean(mi_C)

    std_A = np.std(mi_A)
    std_B = np.std(mi_B)
    std_C = np.std(mi_C)

    out = np.array([mu_A, std_A, mu_B, std_B, mu_C, std_C])

    np.savetxt('./exp2_output/experiment2.txt', out)
    print("System A: mean = {}; std = {}".format(mu_A, std_A))
    print("System B: mean = {}; std = {}".format(mu_B, std_B))
    print("System C: mean = {}; std = {}".format(mu_C, std_C))

    data_A = np.round(random_time_intervention(init_x, params_A, t_max,
                                              sym_func, num_times=num_times), 2)
    data_B = np.round(random_time_intervention(init_x, params_B, t_max,
                                              sym_func, num_times=num_times), 2)
    data_C = np.round(random_time_intervention(init_x, params_C, t_max,
                                              sym_func, num_times=num_times), 2)
    sigAB = mi_test_for_significance(data_A, data_B)
    sigBC = mi_test_for_significance(data_B, data_C)
    sigAC = mi_test_for_significance(data_A, data_C)
    
    print(sigAB)
    print(sigBC)
    print(sigAC)


if __name__ == "__main__":
    argv = sys.argv[1:]

    num_trials=100
    try:
        opts, args = getopt.getopt(argv, "n:")
    except:
        print("Option error.")
 
    for opt, arg in opts:
        if opt in ['-n']:
            num_trials = int(arg)
    main(num_trials)
