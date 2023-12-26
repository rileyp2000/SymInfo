# file: experiment_1.py

import numpy as np
from symmetries import *
from stochastic_systems import *
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import pdb

def main():
    r = np.array([1.])
    k = np.array([1000.])
    alpha = ([[1.]])
    sigma = np.array([0.1])
    gamma_A = 1.
    params_A = [r, k, alpha, sigma, gamma_A]
    gamma_B =  1.2
    params_B = [r, k, alpha, sigma, gamma_B]
    t_max = 2.
    num_times = 10

#    init_x = []
#    for ii in range(10):
#        init_x.append(np.array(np.abs(np.random.normal(5., 0.5)), ndmin=1))
    init_x = np.array([5.])
    data_A = np.round(random_time_intervention(init_x, params_A, t_max, lvsym,
                                      num_times=num_times), 2)
    data_B = np.round(random_time_intervention(init_x, params_B, t_max, lvsym,
                                      num_times=num_times), 2)
    print("Data A")
    print(data_A)
    print("Data B")
    print(data_B)
    mi_A = mutual_info_regression(data_A[:,0].reshape(-1,1), data_A[:,1])
    mi_B = mutual_info_regression(data_B[:,0].reshape(-1,1), data_B[:,1])
    print("MI logistic: {} MI generalized logistic: {}".format(mi_A, mi_B))
    plt.plot(data_A[:,0], data_A[:,1], 'ob')
    plt.plot(data_B[:,0], data_B[:,1], 'or')
    plt.show()

if __name__ == "__main__":
    main()
