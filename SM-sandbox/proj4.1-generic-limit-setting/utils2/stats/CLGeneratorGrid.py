##  ==============================================================================================================================
##  Brief :  CLGeneratorGrid class; grid of CLGenerator objects allows construction and evaluation of whole grid simultaneously
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  ==============================================================================================================================

import numpy as np

import utils2.stats.helpers               as     st
from   utils2.stats.CLGenerator           import CLGenerator
from   utils2.stats.enums                 import LimitsMethod
from   utils2.objects.Grid                import Grid, do_for_all_in_tensor, do_for_all_pairs_in_tensors
import utils2.utils.globals_and_fallbacks as     glob


##  Store a grid of objects
#
class CLGeneratorGrid (Grid) :
	## Constructor
	# 
	def __init__ (self, arg=None, generate=False, limits_method=None, confidence_level=None, **kwargs) :
		super(CLGeneratorGrid, self).__init__(arg)
		self.limits_method    = glob.limits_method
		self.confidence_level = glob.confidence_level
		if type(limits_method) is LimitsMethod : self.limits_method    = limits_method
		if confidence_level    is not None     : self.confidence_level = confidence_level
		if generate : self.generate_grid(**kwargs)
	## Create grid of CLGenerators
	# 
	def generate_grid (self, **kwargs) :
		super(CLGeneratorGrid, self).generate_grid(element_type=CLGenerator, **kwargs)
		self.SM_generator = CLGenerator(**kwargs)
	## Set distributions associated with each grid point, if desired
	# 
	def set_distributions (self, dist_grid) :
		if dist_grid.values.shape != self.values.shape : raise ValueError(f"CLGenerator.set_distributions(): grid provided with shape {dist_grid.values.shape} where {self.values.shape} expected")
		def tmp_func (x, y, **kwargs) : x.dist = y
		do_for_all_pairs_in_tensors(self.values, dist_grid.values, tmp_func)	
	## Set reference distribution associated with the whole grid
	# 
	def set_reference_distribution (self, ref_dist) :
		def tmp_func (x, **kwargs) : x.set_ref_dist(kwargs["ref_dist"])
		do_for_all_in_tensor(self.values, tmp_func, ref_dist=ref_dist)
	## Set SM distribution (used to calculate CLs denominator)
	# 
	def set_SM_distribution (self, SM_dist) :
		self.SM_generator.dist = SM_dist
	## get grid of confidence levels
	# 
	def get_CL_grid (self, dist) :
		if   self.limits_method is LimitsMethod.CLsb :
			CLb = 1.
		elif self.limits_method is LimitsMethod.CLs  :
			if not hasattr(self.SM_generator, "dist") : raise ValueError(f"CLGeneratorGrid.get_CL_grid(): {self.limits_method} method requested but SM_generator.dist not set")
			CLb = self.SM_generator.eval(dist)
		else :
			raise NotImplementedError(f"CLGeneratorGrid.get_CL_grid(): limit setting method {self.limits_method} not recognised")
		ret = Grid(glob.scan_params)
		ret.generate_grid()
		tmp_shape  = ret.values.shape
		tmp_values = ret.values.flatten()
		CL_gens    = self.values.flatten()
		for idx in range(len(CL_gens)) :
			tmp_values[idx] = CL_gens[idx].eval(dist) / CLb
		ret.values = tmp_values.reshape(tmp_shape)
		return ret
	## get confidence limit
	# 
	def get_limit (self, dist, confidence_level=None) :
		if confidence_level is None :
			confidence_level = self.confidence_level
		grid_of_CL = self.get_CL_grid(dist)
		return st.get_limit_from_levels(grid_of_CL, confidence_level)


