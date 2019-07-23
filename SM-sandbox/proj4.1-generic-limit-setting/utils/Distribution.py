##  =======================================================================================================================
##  Brief: Distribution class; stores vector of measurements or predictions and their covariances.
##  Author: Stephen Menary
##  Email: sbmenary@gmail.com
##  =======================================================================================================================

import sys

import numpy as np

import utils.stats as st



## Distribution class
#
#  Holds values and covariances for a distribution (measurement or prediction). Allows addition and subtraction. Facilitates e.g. chi2 calculation.
class Distribution :
	## Default constructor
	# 
	def __init__ (self, other=None, length=None, values=None, cov=None, name="", includes_SM=True) :
		"Initialise Distribution from other Distribution, numpy array and/or covariance matrix, or as an empty container with a given length. If other then remaining arguments are ignored."
		self.name = name
		self.includes_SM = includes_SM
		if other is not None :
			self.values = other.values
			self.cov    = other.cov
			return
		if length is None and values is None and cov is None :
			length = 0
		if length is not None :
			if values is not None : raise ValueError("Cannot specify both a length and a set of values for initialisation").with_traceback(sys.exc_info()[2])
			if cov    is not None : raise ValueError("Cannot specify both a length and a covariance matrix for initialisation").with_traceback(sys.exc_info()[2])
			self.values = np.zeros(shape=(length))
			self.cov    = np.zeros(shape=(length, length))
		if values is not None :
			length = len(values)
			self.values = np.array(values)
			self.cov    = np.zeros(shape=(length, length))
		if cov is not None :
			cov = np.array(cov)
			length = cov.shape[0]
			if cov.shape != (length, length) : raise ValueError("Covariance matrix must be square").with_traceback(sys.exc_info()[2])
			self.cov = np.array(cov)
			if values is None :
				self.values = np.zeros(shape=(length))
			else :
				if len(self.values) is not length : raise ValueError("Incompatible lengths for cov and values").with_traceback(sys.exc_info()[2])
	## Set covariance from correlation and amplitudes
	# 
	def set_covariance(corr, amp) :
		self.cov = st.get_covariance(corr, amp)
	## Alternative constructor
	# 
	@classmethod
	def from_correlation_and_amplitudes (cls, corr, amp, values=None, name="") :
		"Initialise Distribution from correlations and error amplitudes (also with central values if specified)"
		return cls(values=values, cov=st.get_covariance(corr, amp), name=name)
	## Length
	# 
	def __len__ (self) :
		return len(self.values)
	## Str
	# 
	def __str__ (self) :
		return f"Distribution '{self.name}' with {len(self.values)} entries [" + "  ".join([f"{v}" for v in self.values[:5]]) + "]"
	## Addition of a Distribution object
	# 
	def __add__ (self, other) :
		if len(self) is not len(other) : raise ValueError(f"Cannot add distribution with length {len(self)} to that with length {len(other)}").with_traceback(sys.exc_info()[2])
		values = self.values + other.values
		cov    = self.cov    + other.cov
		return Distribution(values=values, cov=cov)
	def add_to_values (self, v) :
		if len(self) is not len(v) : raise ValueError(f"Cannot add array of length {len(other)} to that with length {len(self)}").with_traceback(sys.exc_info()[2])
		self.values = self.values + v
	def add_to_cov (self, c) :
		if self.cov.shape != c.shape : raise ValueError(f"Cannot add matrix of shape {c.shape} to that with length {self.cov.shape}").with_traceback(sys.exc_info()[2])
		self.cov = self.cov + c
	## Subtraction of a Distribution object
	# 
	def __sub__ (self, other) :
		if len(self) is not len(other) : raise ValueError(f"Cannot subtract distribution with length {len(other)} from that with length {len(self)}").with_traceback(sys.exc_info()[2])
		values = self.values - other.values
		cov    = self.cov    + other.cov
		return Distribution(values=values, cov=cov)
	def subtract_values (self, v) :
		if len(self) is not len(v) : raise ValueError(f"Cannot subtract array of length {len(other)} from that with length {len(self)}").with_traceback(sys.exc_info()[2])
		self.values = self.values - v
	def subtract_cov (self, c) :
		if self.cov.shape != c.shape : raise ValueError(f"Cannot subtract matrix of shape {c.shape} from that with length {self.cov.shape}").with_traceback(sys.exc_info()[2])
		self.cov = self.cov - c
	## Multiply by a constant factor
	# 
	def __mul__ (self, sf) :
		return Distribution(values=sf*self.values, cov=sf*sf*self.cov, name=self.name)
	## Divide by a constant factor
	# 
	def __div__ (self, sf) :
		return Distribution(values=self.values/sf, cov=self.cov/(sf*sf), name=self.name)
	## Calculate chi2 (difference between self and other)
	# 
	def chi2(self, other) :
		# print("=========")
		# print(other.values[:5])
		# print(other.cov[:5,:5])
		# print("  -")
		# print(self.values[:5])
		# print(self.cov[:5,:5])
		# print("  =")
		res = other - self
		# print(res.values[:5])
		# print(res.cov[:5,:5])
		# print("=========")
		return np.matmul(res.values, np.matmul(np.linalg.inv(res.cov), res.values))
	## Generate toys using expected covariance
	# 
	def generate_toys (self, n_toys) :
		v, w = st.get_covariance_eigendirections_and_amplitudes(self.cov)
		toy_shifts = np.random.normal(0, 1, (n_toys, len(w)))
		toys = []
		for i in range(n_toys) :
			toy_values = np.matmul(v, np.multiply(w, toy_shifts[i])) + self.values
			toys.append( Distribution(values=toy_values, cov=self.cov, name=f"{self.name}::toy {i+1}", includes_SM=self.includes_SM) )
		return toys

