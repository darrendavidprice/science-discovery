##  ===========================================================================================================================
##  Brief :  CLGeneratorBase class; configured by test_stat and/or strategy enums and toy q values. Computes confidence level
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  ===========================================================================================================================

import numpy as np
import scipy.stats

from utils2.objects.Grid                  import Grid
from utils2.stats.enums                   import TestStatistic, TestStatisticStrategy
import utils2.utils.globals_and_fallbacks as     glob


##  CLGeneratorBase object
#
class CLGeneratorBase :
	## Clear contents
	# 
	def clear (self) :
		self.test_stat = glob.test_statistic
		self.strategy  = glob.test_stat_strategy
		self.q_toys    = None
	## Constructor
	# 
	def __init__ (self, test_stat=None, strategy=None, q_toys=None) :
		self.clear()
		if type(test_stat) is TestStatistic         : self.test_stat = test_stat
		if type(test_stat) is str                   : self.test_stat = TestStatistic[test_stat]
		if type(strategy ) is TestStatisticStrategy : self.strategy  = strategy
		if type(strategy ) is str                   : self.strategy  = TestStatisticStrategy[strategy]
		self.q_toys = q_toys
	## Evaluate measurement using known distribution
	# 
	def _eval_using_known_distribution (self, q) :
		if type(q) not in [list, np.ndarray] : q = [q]
		if self.test_stat is TestStatistic.chi2 :
			if not hasattr(self, "ndof") :
				if hasattr(self, "dist") : self.ndof = len(self.dist)
				else : raise AttributeError(f"CLGeneratorBase._eval_using_known_distribution(): {TestStatistic.chi2} test statistic requested but self.ndof not set")
			return 1.0 - scipy.stats.chi2.cdf(q, self.ndof)
		raise NotImplementedError(f"CLGeneratorBase._eval_using_known_distribution(): test statistic {self.test_stat} not implemented")
	## Evaluate measurement(s) using toys
	# 
	def _eval_using_toys (self, q) :
		if self.q_toys is None : raise ValueError(f"CLGeneratorBase._eval_using_toys(): toys not set")
		if type(q) not in [list, np.ndarray] : q = [q]
		if type(self.q_toys) in [list, np.ndarray] :
			return np.interp(q, self.q_toys, np.linspace(0., 1., len(self.toys)))
		raise TypeError(f"CLGeneratorBase._eval_using_toys(): type {type(self.q_toys)} not recognised as a format for toy distribution")
	## Evaluate measurement(s)
	# 
	def eval(self, q) :
		if self.strategy is None : raise ValueError(f"CLGeneratorBase.eval(): strategy not set; available values are {TestStatisticStrategy.allowed_values()}")
		if self.strategy is TestStatisticStrategy.assume : return self._eval_using_known_distribution(q)
		if self.strategy is TestStatisticStrategy.toys   : return self._eval_using_toys(q)
		raise ValueError(f"CLGeneratorBase.eval(): strategy {self.strategy} not recognised; available values are {TestStatisticStrategy.allowed_values()}")
