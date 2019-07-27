##  ===========================================================================================================================
##  Brief :  CLGenerator class; computes test-statistic from a list of distribution objects and evaluates confidence level
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  ===========================================================================================================================

import numpy as np

from   utils2.stats.CLGeneratorBase import CLGeneratorBase
from   utils2.stats.enums           import TestStatistic, LimitsMethod

##  CLGenerator object
##  (calculates the value of a test statistic and passes it on to CLGeneratorBase to find the CL)
#
class CLGenerator (CLGeneratorBase) :
	## Constructor
	# 
	def __init__ (self, test_stat=None, strategy=None, q_toys=None, func_input_to_q=None, ndof=None) :
		self.clear()
		super(CLGenerator, self).__init__(test_stat=test_stat, strategy=strategy, q_toys=q_toys)
		if ndof            is not None : self.ndof            = ndof
		if func_input_to_q is not None : self.func_input_to_q = func_input_to_q
	## Clear contents
	# 
	def clear (self) :
		super(CLGenerator, self).clear()
		self.func_input_to_q = None
	## Evaluate measurement(s)
	# 
	def eval(self, inputs) :
		q = []
		if type(inputs) not in [list, np.ndarray] : inputs = [inputs]
		if self.func_input_to_q is None :
			self.generate_func_input_to_q()
		for inp in inputs : q.append(self.func_input_to_q(self, inp))
		CL = super(CLGenerator, self).eval(q)
		for i in [i for i in range(len(CL)) if CL[i] < 0.] : q[i] = 0.
		for i in [i for i in range(len(CL)) if CL[i] > 1.] : q[i] = 1.
		return CL
	## Create function which turns input distributions into test statistic measurements
	# 
	def generate_func_input_to_q(self) :
		if self.test_stat == TestStatistic.chi2 :
			def tmp_func (_self, _d) : return _self.dist.chi2(_d)
			self.func_input_to_q = tmp_func
			return self.func_input_to_q
		#if self.test_stat == TestStatistic.dchi2 :
		#	self.func_input_to_q = lambda self, d : return self.dist.chi2(d) - self.ref_dist.chi2(d)
		raise NotImplementedError(f"CLGenerator.generate_func_input_to_q(): not implemented for test stat {self.test_stat}")
	## Set SM distribution: needed for certain test statistics
	# 
	def set_ref_dist (self, ref_dist) : self.ref_dist = ref_dist
