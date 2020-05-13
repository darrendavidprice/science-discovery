#   Implementation of SamplingSimulator base class
#   Author:  Stephen Menary  (stmenary@cern.ch)

import numpy as np

from .Simulator import Simulator
from .stats     import randomly_sample_function, randomly_sample_grid, randomly_sample_meshgrid



#  Class:  Generate random datapoints by sampling a NoiseModel or ContrastiveModel
#
class SamplingSimulator (Simulator) :
    def __init__ (self, name="", model=None, axes=[]) :
        self.clear()
        self.name  = name
        self.axes  = axes
        self.model = model
    def add_axis (self, axis) :
        self.axes.append(axis)
    def add_axis_from_limits_and_num_points (self, lower, upper, n_points) :
        self.axes.append(np.linspace(ower, upper, n_points))
    def clear (self) :
        super(SamplingSimulator, self).clear()
        self.name  = ""
        self.axes  = []
        self.model = None
    def generate (self, n_points, *argv, **kwargs) :
        if type(self.model) == type(None) :
            raise RuntimeError("Model not set")
        grid = np.meshgrid(*self.axes)
        Z    = []
        for el in zip(*[a.flatten() for a in grid]) :
            Z.append(self.model.evaluate(el))
        Z = np.array(Z).reshape(grid[0].shape)
        return randomly_sample_meshgrid (n_points, Z, *grid)


