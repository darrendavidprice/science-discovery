##  =======================================================================================================================
##  Brief :  Grid class; stores tensor of objects with labelled axes and bins
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  =======================================================================================================================

import numpy as np

from   utils2.objects.ScanParam import ScanParam


##  Perform an assignment for every item in an arbitrary dimension np.ndarray instance
#
def assign_for_all_in_tensor_from_friend (A, B, func, index=(), **kwargs) :
	if A.shape != B.shape :
		raise ValueError(f"Grid.assign_for_all_in_tensor_from_friend(): A.shape ({A.shape}) != B.shape ({B.shape})")
	for additional_index in range(len(A[index])) :
		new_index = index + (additional_index,)
		a = A[new_index]
		b = B[new_index]
		if type(a) is np.ndarray :
			assign_for_all_in_tensor_from_friend(A, B, func, new_index, **kwargs)
		else :
			A[new_index] = func(a, b, **kwargs)


##  Perform an action for every item in an arbitrary dimension np.ndarray instance
#
def do_for_all_in_tensor (A, func, **kwargs) :
	for a in A :
		if type(a) is np.ndarray :
			do_for_all_in_tensor(a, func, **kwargs)
		else :
			func(a, **kwargs)

##  Perform an action for every matching pair in two arbitrary (but equal) dimension np.ndarray instances
#
def do_for_all_pairs_in_tensors (A, B, func, **kwargs) :
	for a, b in zip(A, B) :
		if type(a) is np.ndarray :
			do_for_all_pairs_in_tensors(a, b, func, **kwargs)
		else :
			func(a, b, **kwargs)


##  Store a grid of objects
#
class Grid :
	## Clear contents
	# 
	def clear (self) :
		self.axes = []
		self.keys = []
		self.values = None
	## Constructor
	# 
	def __init__ (self, arg=None, generate=False) :
		self.clear()
		if arg is None :
			return
		if type(arg) is Grid :
			self.keys   = arg.keys
			self.axes   = arg.axes
			if generate : self.generate_grid()
			else : self.values = arg.values
			return
		if type(arg) is list :
			for p in arg :
				self.add_axis(p)
			if generate : self.generate_grid()
			return
		raise ValueError(f"Grid.__init__(): argument of type {type(arg)} not valid")
	## class method constructor for param grid
	# 
	@classmethod
	def create_param_grid(cls, params) :
		ret = cls(params)
		return ret
	## Length
	# 
	def __len__ (self) :
		return len(self.keys)
	## Add a new axis
	# 
	def add_axis (self, p) :
		if type(p) is not ScanParam :
			raise ValueError(f"Grid.__init__(): argument of type {type(arg)} where {type(ScanParam)} expected")
		self.axes.append(p.scan_points())
		self.keys.append(p.name)
	## Clear contents
	# 
	def generate_grid (self, element_type=float, **kwargs) :
		self.values = np.full(tuple([len(axis) for axis in self.axes]), element_type(**kwargs))
		values = self.values.flatten() ;
		for idx in range(len(values)) : values[idx] = element_type(**kwargs)
		self.values = values.reshape(self.values.shape)

