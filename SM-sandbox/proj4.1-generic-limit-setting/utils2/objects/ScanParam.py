##  =======================================================================================================================
##  Brief :  ScanParam class; stores modifiable data associated with a parameter with defined scan steps
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  =======================================================================================================================

import numpy as np


##  ScanParam object
#
class ScanParam :
	##  Constructor
	#
	def __init__ (self, name="", limits=[], stepsize=None, n_points=None, label="", units="") :
		self.name = name
		self.limits = limits
		self.n_points = 2
		if stepsize != None and n_points != None : raise RuntimeError("ScanParam.__init__(): cannot specify both a stepsize and n_points")
		if stepsize != None : self.set_stepsize(stepsize)
		if n_points != None : self.n_points = n_points
		self.label = label
		self.units = units
	##  Length
	#
	def __len__ (self) :
		return len(self.scan_points())
	##  Informal representation
	#
	def __str__ (self) :
		return f"{self.name}"
	##  Formal representation
	#
	def __repr__ (self) :
		return f"ScanParam(name={self.name}, limits={self.limits}, n_points={self.n_points}, label={self.label}, units={self.units})"
	##  Set stepsize
	#
	def set_stepsize (self, stepsize) :
		n_limits = len(self.limits)
		if n_limits is not 2 :
			raise RuntimeWarning(f"ScanParam.set_stepsize(): parameter \'{self.name}\' has {n_limits} where 2 expected. Cannot set stepsize.")
			return
		scan_range = self.limits[1] - self.limits[0]
		if scan_range % stepsize != 0 :
			raise RuntimeWarning(f"ScanParam.set_stepsize(): parameter \'{self.name}\' with limits {self.limits} cannot be exactly split into bins of width {stepsize}. Rounding will occur.")
		self.n_points = 1 + int(float(scan_range)/float(stepsize))
	##  Get array of scan points
	#
	def scan_points (self) :
		n_limits = len(self.limits)
		if n_limits is not 2 :
			raise ValueError(f"ScanParam.scan_points(): parameter \'{self.name}\' has {n_limits} limits where 2 expected")
		if self.n_points < 2 :
			raise ValueError(f"ScanParam.scan_points(): parameter \'{self.name}\' requested with {self.n_points} scan points (must be at least 2)")
		scan_range = self.limits[1] - self.limits[0]
		stepsize = scan_range / (self.n_points-1)
		return np.array([self.limits[0] + i*stepsize for i in range(self.n_points)])

