# file: experiment_1.py

import numpy as np
from symmetries import *
from stochastic_systems import *
from sklearn.feature_selection import mutual_info_regression
import pdb

def main():
    r = np.array([1.])
    k = np.array([100.])
    alpha = ([[1.]])
    sigma = np.array([0.])
    gamma_A = 1.
    params_A = [r, k, alpha, sigma, gamma_A]
    gamma_B =  1.2
    params_B = [r, k, alpha, sigma, gamma_B]
    t_max = 2.

#    init_x = []
#    for ii in range(10):
#        init_x.append(np.array(np.abs(np.random.normal(5., 0.5)), ndmin=1))
    init_x = np.array([5.])
    data_A = random_time_intervention(init_x, params_A, t_max, lvsym)
    data_B = random_time_intervention(init_x, params_B, t_max, lvsym)
    mi_A = mutual_info_regression(data_A[:,0].reshape(-1,1), data_A[:,1])
    mi_B = mutual_info_regression(data_B[:,0].reshape(-1,1), data_B[:,1])
    print("MI logistic: {} MI generalized logistic: {}".format(mi_A, mi_B))

if __name__ == "__main__":
    main()
