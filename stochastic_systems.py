# file: stochastic_systems.py

"""Description.
"""

import eugene as eu
import numpy as np
import copy
import random
import scipy.integrate
from joblib import Parallel, delayed
import multiprocessing
import pdb

class LotkaVolterraSND( object ):
    """ Implementation of stochastic N species Competitive Lotka-Volterra 
        equations and generalizations thereof. System is based on 2-species 
        stochastic model presented in Permanance of Stochastic Lotka-Volterra 
        Systems - Liu, M. & Fan M.; J Nonlinear Sci (2017)
    """

    def __init__(self, r, k, alpha, sigma, init_x, gamma=1., init_t=0,
                 steps=10**3, fixed_step=False):
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
        self._steps = steps
        self._fixed_step = fixed_step
        self._sigma = sigma


    def update_x(self, elapsed_time):

        if elapsed_time == 0.:
            return None
        
        remainder = None
        if self._fixed_step:
            delta = float(10 ** (-3))
            self._steps = int(round(elapsed_time / delta))
            if not elapsed_time - self._steps * delta <= 10.**(-9):
                pdb.set_trace()
            
        else:
            delta = float(elapsed_time) / float(self._steps)
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
            if remainder is not None:
                noise = np.random.normal()
                
                dX[i] = self._r[i] * (X[i] ** self._gamma) * (1. - (np.sum(self._alpha[i] *
                X)/self._k[i]) ) + (self._sigma[i] * X[i] * noise / (
                np.sqrt(remainder) ) ) + (self._sigma[i]**2 / 2.) * (X[i]
                * (noise**2 - 1.))

                X[i] = X[i] + dX[i] * remainder
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


    def point_sample(self,
#            init_x, 
#            params, 
            times=np.linspace(0., 10., 100), 
#            duration=10, 
            num_samples=10,
            fixed_step=False
            ):
        
#        r, k, alpha, sigma, gamma = params
    
#        LVSND = LotkaVolterraSND(r, k, alpha, sigma, init_x[0], gamma=gamma,
#                                 fixed_step=fixed_step)
        restore_step = copy.copy(self._fixed_step)
        self._fixed_step = fixed_step
        data = []
        for ii in range(len(init_x)):
            for jj in range(num_samples):
                self._x = copy.copy(init_x[ii])
                data.append(self.check_xs(times))
        self._fixed_step = restore_step 
        return data
    

    def random_time_int_loop_func(self,
                                  ts, 
                                  t_max, 
                                  func):
        # reset the state
        self._x = copy.copy(self._init_x)
    
        # choose a random time
        tt = np.round(np.random.rand() * t_max, 3)
    
        # evolve to tt
        self.update_x(np.round(tt - ts, 3))
    
        # apply func to change state
        self._x = func(copy.copy(self._x), self._r, self._k)
    
        # evolve to tmax
        self.update_x(np.round(t_max - tt, 3))
    
        # read state and store with tt
        return np.concatenate([np.array(tt, ndmin=2),
                                        self._x.reshape(1,-1)], axis=1)
    
    
    def random_time_intervention(self,
            t_max,
            func,
            num_times=100
            ):
    
        free_cores = 6
        cpus = max(multiprocessing.cpu_count() - free_cores, 1)
    
        restore_step = copy.copy(self._fixed_step)
        self._fixed_step = True
        
        ts = 0.
    
        data = Parallel(n_jobs=cpus, 
            verbose=0)(delayed(self.random_time_int_loop_func)(ts, t_max, 
                    func) for ii in range(num_times))
    
        data = np.concatenate(data, axis=0)
   
        self._fixed_step = restore_step
        return data
    
    
    def random_x_int_loop_func(self,
                               init_x, 
                               ts, 
                               t_max, 
                               r, 
                               k, 
                               func):

        # reset the state
        self._x = copy.copy(self._init_x)
    
        # choose a random time
        tt = np.round(np.random.rand() * t_max, 3)
    
        # evolve to tt and save state
        self.update_x(np.round(tt - ts, 3))
        x_int = self._x    
    
        # apply func to change state
        self._x = func(copy.copy(self._x), r, k)
    
        # evolve to tmax
        self.update_x(np.round(t_max - tt, 3))
    
        # read state and store with x_int
        return np.concatenate([np.array(x_int, ndmin=2),
                                        self._x.reshape(1,-1)], axis=1)
    
    
    
    def random_x_intervention(self,
            init_x,
            params,
            t_max,
            func,
            num_times=100
            ):
    
        r, k, alpha, sigma, gamma = params
    
        free_cores = 6
        cpus = max(multiprocessing.cpu_count() - free_cores, 1)
        restore_step = copy.copy(self._fixed_step)
        self._fixed_step = True
        
        ts = 0.
    
        data = Parallel(n_jobs=cpus,
                        verbose=0)(delayed(random_time_int_loop_func)(init_x,
                        ts, t_max, self, r, k, func) for ii in range(num_times))
    
        data = np.concatenate(data, axis=0)
    
        self._fixed_step = restore_step

        return data


class RickerS( object ):
    """ Implementation of a stochastic Ricker growth model.

        citations:
    """

    def __init__(self, r, k, sigma, init_x, gamma=1., init_t=0,
                 fixed_step=False):
        
        # set attributes
        self._r = r
        self._k = k
        self._gamma = gamma
        self._init_x = copy.copy(init_x)
        self._init_t = float(init_t)
        self._sigma = sigma
        self._x = init_x


    def update_x(self, time_steps):

        if time_steps == 0:
            return None
            
        X = self._x
        for s in range(time_steps):
            for i in range(time_steps): 
                noise = np.random.normal(0., self._sigma)
                X = self._r * X * np.exp(1. - X / self._k) + noise 
        self._x = X


    def check_xs(self, times):
        # times assumed to be integer time steps
        t_n = 0
        xs = [copy.copy(self._x)]
        for i in range(1,len(times)):
            if times[i] == 0:
                continue
            interval = int(times[i] - t_n)
            t_n = times[i]
            self.update_x(interval)
            xs.append(self._x)
        
        return np.array(xs).reshape(-1,1)


    def point_sample(self,
#            init_x, 
#            params, 
            times=np.linspace(0., 10., 100), 
#            duration=10, 
            num_samples=10,
            fixed_step=False
            ):
        
#        r, k, alpha, sigma, gamma = params
    
#        LVSND = LotkaVolterraSND(r, k, alpha, sigma, init_x[0], gamma=gamma,
#                                 fixed_step=fixed_step)
        restore_step = copy.copy(self._fixed_step)
        self._fixed_step = fixed_step
        data = []
        for ii in range(len(init_x)):
            for jj in range(num_samples):
                self._x = copy.copy(init_x[ii])
                data.append(self.check_xs(times))
        self._fixed_step = restore_step 
        return data
    

    def random_time_int_loop_func(self,
                                  ts, 
                                  t_max, 
                                  func):
        # reset the state
        self._x = copy.copy(self._init_x)
    
        # choose a random time
        tt = np.round(np.random.rand() * t_max, 3)
    
        # evolve to tt
        self.update_x(np.round(tt - ts, 3))
    
        # apply func to change state
        self._x = func(copy.copy(self._x), self._r, self._k)
    
        # evolve to tmax
        self.update_x(np.round(t_max - tt, 3))
    
        # read state and store with tt
        return np.concatenate([np.array(tt, ndmin=2),
                                        self._x.reshape(1,-1)], axis=1)
    
    
    def random_time_intervention(self,
            t_max,
            func,
            num_times=100
            ):
    
        free_cores = 6
        cpus = max(multiprocessing.cpu_count() - free_cores, 1)
    
        restore_step = copy.copy(self._fixed_step)
        self._fixed_step = True
        
        ts = 0.
    
        data = Parallel(n_jobs=cpus, 
            verbose=0)(delayed(self.random_time_int_loop_func)(ts, t_max, 
                    func) for ii in range(num_times))
    
        data = np.concatenate(data, axis=0)
   
        self._fixed_step = restore_step
        return data
    
    
    def random_x_int_loop_func(self,
                               init_x, 
                               ts, 
                               t_max, 
                               r, 
                               k, 
                               func):

        # reset the state
        self._x = copy.copy(self._init_x)
    
        # choose a random time
        tt = np.round(np.random.rand() * t_max, 3)
    
        # evolve to tt and save state
        self.update_x(np.round(tt - ts, 3))
        x_int = self._x    
    
        # apply func to change state
        self._x = func(copy.copy(self._x), r, k)
    
        # evolve to tmax
        self.update_x(np.round(t_max - tt, 3))
    
        # read state and store with x_int
        return np.concatenate([np.array(x_int, ndmin=2),
                                        self._x.reshape(1,-1)], axis=1)
    
    
    
    def random_x_intervention(self,
            init_x,
            params,
            t_max,
            func,
            num_times=100
            ):
    
        r, k, alpha, sigma, gamma = params
    
        free_cores = 6
        cpus = max(multiprocessing.cpu_count() - free_cores, 1)
        restore_step = copy.copy(self._fixed_step)
        self._fixed_step = True
        
        ts = 0.
    
        data = Parallel(n_jobs=cpus,
                        verbose=0)(delayed(random_time_int_loop_func)(init_x,
                        ts, t_max, self, r, k, func) for ii in range(num_times))
    
        data = np.concatenate(data, axis=0)
    
        self._fixed_step = restore_step

        return data
