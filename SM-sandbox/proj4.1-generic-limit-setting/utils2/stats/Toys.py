##  =======================================================================================================================
##  Brief :  Toys class; stores a collection of toys with the same covariance
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  =======================================================================================================================

import numpy as np


##  Store a selection of toys which all share the same covariance matrix
#
class Toys :
	## Constructor
	# 
	def __init__ (self, dim=0, other=None, shallow=False) :
		self.clear(dim)
		if other is None : return
		load_from_other(other=other, shallow=shallow)
	## Length = num_toys
	# 
	def __len__ (self) :
		return len(self.toys)
	## Append a new toy (shallow or deep copy)
	# 
	def append_toy (self, arr, shallow=False) :
		if len(arr) != self.dim :
			raise ValueError(f"Toys.append_toy(): toy with length {len(arr)} where {self.dim} expected")
		if shallow :
			self.toys.append(arr)
		else :
			self.toys.append(np.array(arr))
	## Copy new toys from other Toys object
	# 
	def append_toys (self, other, shallow=False) :
		for toy in other.toys :
			append_toy(toy, shallow=shallow)
	## Clear contents
	# 
	def clear (self, dim=0) :
		self.toys = []
		self.cov  = np.zeros(size=(dim, dim))
		self.dim  = dim
	## dimensionality = number of measurements per toy
	# 
	def dimension (self) :
		return self.dim
	## Loop through toys in another container and make a copy. Also copy covariance. Replaces current contents.
	# 
	def load_from_other (self, other, shallow=False) :
		self.clear(other.dim)
		for toy in other.toys :
			append_toy(toy, shallow=shallow)
		self.cov = np.array(other.cov)
	## num_toys = length
	# 
	def num_toys (self) :
		return len(self)