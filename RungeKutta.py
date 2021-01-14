import numpy as np


class RungeKutta():
    """
    The Runge-Kutta method for ordinary differential equations:
        dy=f(t,y)dt
    where y is k-dimensional.

    Author: Jichun Si, si.jichun@outlook.com

    The parameters needed in init():
        T:       the end time, t ∈ [0,T]
        y0:      a numpy array, the initial point, k-dimensional
        delta_t: step size of t

    Before use, iterms below should be provided:
        f:       a function with input (t,y,paras), where:
                        t:     a scalar, current t
                        y:     numpy array, current y
                        paras: some parameters needed in the computation of f
                 output: dy/dt, which is also k-dimensional numpy array
        params:  the parameters needed in computing f

    After interate(), several functions could provide results:
        get_paths: get the t array and the paths array, output:
            t:     the t array
            paths: k-by-len(t) numpy array of ODE result paths

        get_diff_paths: get the t array and the differenced paths array, output:
            t:     the t array (with len(t)-1 dimension)
            paths: k-by-(len(t)-1) numpy array of differenced ODE result paths

    """
    def __init__(self, T, y0, delta_t):
        self.y0 = y0
        self.k = self.y0.size
        self.T = T
        self.delta_t = delta_t
        self.t = [0]
        self.paths = np.array(y0)

    params = None

    def f(self, t, y, params):
        pass

    def iterate(self):
        tn = 0
        yn = self.y0.copy()
        while tn < self.T:
            tn += self.delta_t
            k1 = self.f(tn, yn, self.params)
            k2 = self.f(tn + self.delta_t / 2, yn + self.delta_t * k1 / 2,
                   self.params)
            k3 = self.f(tn + self.delta_t / 2, yn + self.delta_t * k2 / 2,
                   self.params)
            k4 = self.f(tn + self.delta_t, yn + self.delta_t * k3, self.params)
            yn += 1.0 / 6 * self.delta_t * (k1 + 2 * k2 + 2 * k3 + k4)
            self.t.append(tn)
            self.paths = np.vstack([self.paths, yn])
        self.t = np.array(self.t)
        self.paths = self.paths.T

    def get_paths(self):
        return (self.t, self.paths)

    def get_diff_paths(self):
        return (self.t[:-1], np.diff(self.paths, axis=1))


