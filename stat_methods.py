# file: stat_methods.py

""" Description
"""

import numpy as np
from scipy.stats import ks_2samp
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from sklearn.feature_selection import mutual_info_regression
import pdb

def test_for_independence(x, y):
    """ Uses an implementation [1] of the density-based empirical likelihood
        ratio test of Vexler et al [2] test for independence.
        [1] 
        [2]
    """
    utils = importr('utils')
    base = importr('base')
    testforDEP = importr('testforDEP')

    """ https://rpy2.github.io/doc/v3.5.x/html/numpy.html#numpy
    """
    # Create a converter that starts with rpy2's default converter
    # to which the numpy conversion rules are added.
    np_cv_rules = default_converter + numpy2ri.converter   
    with np_cv_rules.context():
        result = testforDEP.testforDEP(x, y, test="VEXLER")

    return result.slots['p_value']

def mi_test_for_significance(A, B, alpha=0.05, sample_size=None, num_samples=100):
    """ Uses resampling to assess significance of difference between estimates
        of mutual information.
    """
    if sample_size is None:
        sample_size = int(0.5 * max(A.shape))

    Ax = A[:,0].reshape(-1,1)
    Ay = A[:,1].flatten()
    Bx = B[:,0].reshape(-1,1)
    By = B[:,1].flatten()

    mi_A = []
    mi_B = []
#    for ii in range(len(Ax)):
#        tmp_Ax = Ax.tolist()
#        del tmp_Ax[ii]
#        tmp_Ax = np.array(tmp_Ax).reshape(-1,1)
#        tmp_Ay = Ay.tolist()
#        del tmp_Ay[ii]
#        tmp_Ay = np.array(tmp_Ay)
#        mi_A.append(mutual_info_regression(tmp_Ax, tmp_Ay))
#    for ii in range(len(Bx)):
#        tmp_Bx = Bx.tolist()
#        del tmp_Bx[ii]
#        tmp_Bx = np.array(tmp_Bx).reshape(-1,1)
#        tmp_By = By.tolist()
#        del tmp_By[ii]
#        tmp_By = np.array(tmp_By)
#        mi_B.append(mutual_info_regression(tmp_Bx, tmp_By))

    for ii in range(num_samples):
        indices = np.random.randint(len(Ax), size=sample_size)
        mi_A.append(mutual_info_regression(Ax[indices], Ay[indices]))
        mi_B.append(mutual_info_regression(Bx[indices], By[indices]))

    mi_A = np.array(mi_A)
    mi_B = np.array(mi_B)
    ave_A = np.mean(mi_A)
    ave_B = np.mean(mi_B)
    std_A = np.std(mi_A)
    std_B = np.std(mi_B)
    ks = ks_2samp(mi_A.flatten(), mi_B.flatten())
    return(ks, ave_A, std_A, ave_B, std_B)

