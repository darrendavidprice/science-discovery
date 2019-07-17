##  =======================================================================================================================
##  Brief: Grid class; stores tensor of objects with labelled axes and bins
##  Author: Stephen Menary
##  Email: sbmenary@gmail.com
##  =======================================================================================================================

import numpy as np


##  Store a grid of objects
#
class Grid :
	def __init__ (self, params = []) :
		self.axes, self.keys = [], []
		if type(params) is Grid :
			other = params
			self.keys = other.keys
			self.axes = other.axes
			if hasattr(other, "values") :
				self.values = other.values
			return
		for p in params :
			self.add_param(p)
	def __len__ (self) :
		return len(self.keys)
	def add_param (self, p) :
		if "utils.inputs.ScanParam" not in str(type(p)) :
			raise TypeError(f"Grid.add_param: input of type {type(p)} where inputs.ScanParam expected")
		self.axes.append(p.scan_points())
		self.keys.append(p.name)
	def generate (self, dtype=float) :
		self.values = np.zeros(shape=tuple([len(axis) for axis in self.axes]), dtype=dtype)


##  Create a new grid from the target parameters
#
def create_param_grid (target_params) :
	ret = Grid(target_params)
	return ret