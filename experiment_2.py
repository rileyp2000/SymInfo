# file: experiment_2.py

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


def main(draw_map=False):
    r = 1.5
    k = 500.
    init_x = 5.
    sigma = 0.

    if draw_map:
        ricker = RickerS(r, k, 0., init_x)
        x = init_x
        plt.figure()
        for r in np.linspace(1.2, 8., 1000):
            ricker._r = r
            ricker.update_x(50)
            x = copy.copy(ricker._x)
            for ii in range(100):
                ricker.update_x(1)
                plt.plot(r, ricker._x, 'b.', markersize=1)
            
    plt.show()
#    LV_A_untrans = LotkaVolterraSND(r, k, alpha, sigma, init_x, gamma=gamma_A)
#    LV_A_trans = LotkaVolterraSND(r, k, alpha, sigma, lvsym(init_x, r, k), gamma=gamma_A)
#    LV_B_untrans = LotkaVolterraSND(r, k, alpha, sigma, init_x, gamma=gamma_B)
#    LV_B_trans = LotkaVolterraSND(r, k, alpha, sigma, lvsym(init_x, r, k), gamma=gamma_B)
#
#    # etimate symmetries for each system
#    SE_A = SymmetryEstimator(LV_A_untrans, LV_A_trans, t_max)
#
#    x = np.linspace(5., 90., 100)
#    y = SE_A._sym_model(x)
#    axes = plt.plot(x, y, 'ro')
#    ys = lvsym(x, r, k)
#    plt.plot(x, ys, 'b-')
#    plt.plot(SE_A._sym_data[:,0], SE_A._sym_data[:,1], 'k+')
#    plt.show()

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
    main(True)
