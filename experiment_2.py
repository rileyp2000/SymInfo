# file: experiment_1.py

""" Symmetries known exactly from analysis of model.
"""

import numpy as np
from symmetries import *
from stochastic_systems import *
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import ks_2samp
from stat_methods import *
from estimate_symmetries import *
import matplotlib.pyplot as plt
import sys
import getopt
import pdb


def main():
    r = np.array([1.5])
    k = np.array([500.])
    alpha = ([[1.]])
    sigma = np.array([0.1])
    gamma_A = 1.
    params_A = [r, k, alpha, sigma, gamma_A]
    gamma_B =  1.1
    params_B = [r, k, alpha, sigma, gamma_B]
    t_max = 2.
    num_times = 40
    init_x = np.array([5.])

    LV_A_untrans = LotkaVolterraSND(r, k, alpha, sigma, init_x, gamma=gamma_A)
    LV_A_trans = LotkaVolterraSND(r, k, alpha, sigma, lvsym(init_x, r, k), gamma=gamma_A)
    LV_B_untrans = LotkaVolterraSND(r, k, alpha, sigma, init_x, gamma=gamma_B)
    LV_B_trans = LotkaVolterraSND(r, k, alpha, sigma, lvsym(init_x, r, k), gamma=gamma_B)

    # etimate symmetries for each system
    SE_A = SymmetryEstimator(LV_A_untrans, LV_A_trans, t_max)

    x = np.linspace(5., 90., 100)
    y = SE_A._sym_model(x)
    axes = plt.plot(x, y, 'ro')
    ys = lvsym(x, r, k)
    plt.plot(x, ys, 'b-')
    plt.plot(SE_A._sym_data[:,0], SE_A._sym_data[:,1], 'k+')
    plt.show()

if __name__ == "__main__":
#    argv = sys.argv[1:]
#    
#    num_trials=100
#    try:
#        opts, args = getopt.getopt(argv, "n:")
#    except:
#        print("Option error.")
#    
#    for opt, arg in opts:
#        if opt in ['-n']:
#            num_trials = int(arg)
#    main(num_trials)
    main()
