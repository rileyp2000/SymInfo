# file: stochastic_systems.py

"""Description.
"""

import eugene as eu
import numpy as np
import copy
import random
import scipy.integrate
import pdb

class LotkaVolterraSND( object ):
    """ Implementation of stochastic N species Competitive Lotka-Volterra 
        equations and generalizations thereof. System is based on 2-species 
        stochastic model presented in Permanance of Stochastic Lotka-Volterra 
        Systems - Liu, M. & Fan M.; J Nonlinear Sci (2017)
    """

    def __init__(self, r, k, alpha, sigma, init_x, gamma=1., init_t=0, steps=10**3):
        """ Initializes a stochastic competitive Lotka-Volterra model with n 
            species
        
            Keyword arguments:
            r -- an array of species growth rates, where r[i] is the growth
                rate of species i.
            k -- an array of species carrying capacities, where k[i] is the 
               carrying capacity of species i. 
            alpha -- the interaction matrix; a matrix of inter-species
                interaction terms, where a[i,j] is the effect of species j on
                the population of species i.
            sigma -- an array of noise intensities where s[i] is the intensity
                of noise affecting species i.
            init_x -- an array of species population size at the start of the
                observation period, where init_x[i] is the initial population
                of species i.
            gamma -- 
            init_t -- the time index at which the observation period starts.
                (default 0)
        """
        
        # set attributes
        self._r = r
        self._alpha = alpha
        self._k = k
        self._gamma = gamma

        self._init_x = copy.copy(init_x)
        self._init_t = float(init_t)

        self._x = copy.copy(init_x)
        self._time = float(init_t)
        self._delta_t = 1
        self._steps = steps
        
        self._sigma = sigma


    def update_x(self, elapsed_time):

        if elapsed_time == 0.:
            return None

        delta = float(elapsed_time) / float(self._steps)
        if delta < 0:
            print(delta)
        X = self._x
        dX = np.zeros(len(X))
        for s in range(self._steps):
            for i in range(len(X)): 
                noise = np.random.normal()
                
                dX[i] = self._r[i] * (X[i] ** self._gamma) * (1. - (np.sum(self._alpha[i] *
                X)/self._k[i]) ) + (self._sigma[i] * X[i] * noise / (
                np.sqrt(delta) ) ) + (self._sigma[i]**2 / 2.) * (X[i]
                * (noise**2 - 1.))

                X[i] = X[i] + dX[i] * delta
                X[i] = np.max([X[i], 0.])
 
        self._x = X


    def check_xs(self, times):
        t_n = 0.
        xs = copy.copy(self._x).reshape(1, len(self._x))
        for i in range(1,len(times)):
            if times[i] == 0.:
                continue
            interval = times[i] - t_n
            t_n = times[i]
            self.update_x(interval)
            xs = np.vstack((xs, self._x.reshape(1, len(self._x))))
        
        return xs

def point_sample(
        init_x, 
        params, 
        times=np.linspace(0., 10., 100), 
        duration=10, 
        num_samples=10
        ):
    
    r, k, alpha, sigma, gamma = params

    LVSND = LotkaVolterraSND(r, k, alpha, sigma, init_x[0], gamma=gamma)

    data = []
    for ii in range(len(init_x)):
        for jj in range(num_samples):
            LVSND._x = copy.copy(init_x[ii])
            data.append(LVSND.check_xs(times))

    return data

def random_time_intervention(
        init_x,
        params,
        t_max,
        func,
        num_times=100
        ):

    r, k, alpha, sigma, gamma = params

    LVSND = LotkaVolterraSND(r, k, alpha, sigma, init_x, gamma=gamma)

#    times = np.sort(np.random.rand(num_times) * t_max)
    
    ts = 0.
    data = []
    for ii in range(num_times):
        # reset the state
        LVSND._x = copy.copy(init_x)

        # choose a random time
        tt = np.random.rand() * t_max

        # evolve to tt
        LVSND.update_x(tt - ts)

        # apply func to change state
        LVSND._x = func(copy.copy(LVSND._x), r, k)

        # evolve to tmax
        LVSND.update_x(t_max - tt)

        # read state and store with tt
        data.append(np.concatenate([np.array(tt, ndmin=2),
                                    LVSND._x.reshape(1,-1)], axis=1))


    data = np.concatenate(data, axis=0)

    return data
