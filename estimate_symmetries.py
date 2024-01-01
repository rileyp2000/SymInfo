# file: estimate_symmetries.py

""" Estimates symmetries at a point given a model.
"""

from stochastic_systems import *
from scipy.interpolate import CubicSpline
import numpy as np
import copy
import pdb


class SymmetryEstimator( object ):

    def __init__(self, model_untrans, model_trans, t_max, resolution=100,
                 replicates=100):
        self._model_untrans = model_untrans
        self._model_trans = model_trans
        self._resolution = resolution
        self._replicates = replicates

        times = np.linspace(0., 1.05*t_max, resolution)
        init_x_untrans = model_untrans._x
        init_x_trans = model_trans._x

        # estimate means at regular time intervals
        untrans = []
        trans = []
        print("Sampling...")
        for ii in range(self._replicates):
            self._model_untrans._x = copy.copy(init_x_untrans)
            untrans.append(model_untrans.check_xs(times).reshape(-1,1))
            self._model_trans._x = copy.copy(init_x_trans)
            trans.append(model_trans.check_xs(times).reshape(-1,1))
        untrans = np.concatenate(untrans, axis=1)
        untrans = np.mean(untrans, axis=1).reshape(-1,1)
        trans = np.concatenate(trans, axis=1)
        trans = np.mean(trans, axis=1).reshape(-1,1)

        # form the rough symmetry estimate
        sym_data = np.concatenate([untrans, trans], axis=1)
        self._sym_data = sym_data[sym_data[:,0].argsort()]

        # build a smoothed (cubic spline) model of the symmetry
        print("Compute spline...")
        self._sym_model = CubicSpline(sym_data[:,0], sym_data[:,1])


    def draw_from_transformed_dist(self, x):
        pass

    def mean_transformed_dist(self, x):
        pass


