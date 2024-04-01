# file: experiment_1.py

""" Symmetries known exactly from analysis of model, or estimated from model
    runs. Lotka-Volterra systems.
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

    # generate data illustrating dynamics for system of type A and B
    LV_A = LotkaVolterraSND(r, k, alpha, sigma, init_x, gamma=gamma_A)
    LV_B = LotkaVolterraSND(r, k, alpha, sigma, init_x, gamma=gamma_B)
    curve_A = []
    curve_B = []
    times = np.linspace(0., t_max, 100)
    for ii in range(num_times):
        LV_A._x = copy.copy(init_x)
        LV_B._x = copy.copy(init_x)
        curve_A.append(LV_A.check_xs(times).reshape(-1,1))
        curve_B.append(LV_B.check_xs(times).reshape(-1,1))
    tmp = np.concatenate(curve_A, axis=1)    
    curve_A = np.concatenate([times.reshape(-1,1), tmp], axis=1)
    np.save('./exp1_output/curve_A.npy', curve_A)
    tmp = np.concatenate(curve_B, axis=1)    
    curve_B = np.concatenate([times.reshape(-1,1), tmp], axis=1)
    np.save('./exp1_output/curve_B.npy', curve_B)

    # Demonstrate that MI estimate converges as expected over sufficiently many
    # trials
    mi_A = []
    mi_B = []
   
    num_samples = [10, 50, 100, 500, 1000, 5000, 10**4]
    for nn in num_samples:
        data_A = np.round(LV_A.random_time_intervention(t_max, lvsym, num_times=nn), 2)
        data_B = np.round(LV_B.random_time_intervention(t_max, lvsym, num_times=nn), 2)
        mi_A.append(mutual_info_regression(data_A[:,0].reshape(-1,1),
                                           data_A[:,1]))
        mi_B.append(mutual_info_regression(data_B[:,0].reshape(-1,1),
                                           data_B[:,1]))

    mi_A = np.concatenate(mi_A)
    mi_B = np.concatenate(mi_B)

    np.save('./exp1_output/mi_A_convergence.npy', mi_A)
    np.save('./exp1_output/mi_B_convergence.npy', mi_B)
    print(mi_A)
    print(mi_B)


    # Demonstrate discrimination from a single trial
    data_A = np.round(LV_A.random_time_intervention(t_max, lvsym, num_times=num_times), 2)
    data_B = np.round(LV_B.random_time_intervention(t_max, lvsym, num_times=num_times), 2)
    mi_A = mutual_info_regression(data_A[:,0].reshape(-1,1), data_A[:,1])
    mi_B = mutual_info_regression(data_B[:,0].reshape(-1,1), data_B[:,1])

    # Test A
    pval_A = test_for_independence(data_A[:,0], data_A[:,1])
    pval_B = test_for_independence(data_B[:,0], data_B[:,1])

    np.savetxt('./exp1_output/mi_A', mi_A)
    np.savetxt('./exp1_output/mi_B', mi_B)
    np.savetxt('./exp1_output/pval_A', pval_A)
    np.savetxt('./exp1_output/pval_B', pval_B)

    print(mi_A)
    print(mi_B)
    print(pval_A)
    print(pval_B)

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
