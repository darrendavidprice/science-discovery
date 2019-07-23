##  =======================================================================================================================
##  Brief: module for performing common stat operations
##  Author: Stephen Menary
##  Email: sbmenary@gmail.com
##  =======================================================================================================================

import sys

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import utils.utils  as utils
import utils.config as config
from   utils.Grid import Grid


## Global definitions
#
test_statistics = ["chi2"]
limit_methods   = ["CLs", "CLs+b"]
test_stat_strategies = ["asymptotic", "toys"]


## Global settings
#
target_coverage    = None
test_stat          = None
limits_method      = None
test_stat_strategy = None


## Safe access for global setting target_coverage
#
def set_target_coverage (coverage) :
	global target_coverage
	target_coverage = float(coverage)

def get_target_coverage () :
	global target_coverage
	if target_coverage is None :
		target_coverage = config.get()
		raise ValueError("stats.get_target_coverage(): target_coverage not set")
	return target_coverage


## Safe access for global setting test_stat
#
def set_test_stat (ts) :
	global test_stat, test_statistics
	if test_stat not in test_statistics :
		raise ValueError(f"stats.set_test_stat(): test statistic {ts} not found in list {test_statistics}")
	test_stat = ts

def get_test_stat () :
	global test_stat, test_statistics
	if test_stat is None :
		test_stat = config.get_and_enforce("GENERAL", "TestStatistic", from_selection=test_statistics)
	if test_stat not in test_statistics :
		raise ValueError(f"stats.get_test_stat(): test statistic {test_stat} not found in list {test_statistics}")
	return test_stat


## Safe access for global setting limits_method
#
def set_limits_method (lm) :
	global limits_method, limit_methods
	if limits_method not in limit_methods :
		raise ValueError(f"stats.set_limits_method(): limit setting method {limits_method} not found in list {limit_methods}")
	limits_method = lm

def get_limits_method () :
	global limits_method, limit_methods
	if limits_method is None :
		limits_method = config.get_and_enforce("GENERAL", "LimitsMethod", from_selection=limit_methods)
	if limits_method not in limit_methods :
		raise ValueError(f"stats.get_limits_method(): limit setting method {limits_method} not found in list {limit_methods}")
	return limits_method


## Safe access for global setting test_stat_strategy
#
def set_test_stat_strategy (strat) :
	global test_stat_strategy, test_stat_strategies
	if strat not in test_stat_strategies :
		raise ValueError(f"stats.set_test_stat_strategy(): test stat strategy {strat} not found in list {test_stat_strategies}")
	test_stat_strategy = strat

def get_test_stat_strategy () :
	global test_stat_strategy, test_stat_strategies
	if test_stat_strategy is None :
		test_stat_strategy = config.get_and_enforce("GENERAL", "TestStatistic.Distribution", from_selection=test_stat_strategies)
	if test_stat_strategy not in test_stat_strategies :
		raise ValueError(f"stats.set_test_stat_strategy(): test stat strategy {strat} not found in list {test_stat_strategies}")
	return test_stat_strategy


## Get mean from array
#
def measure_mean(arr) :
	dim = len(arr[0])
	return [np.mean([a[i] for a in arr]) for i in range(dim)]


## Get covariance matrix from data
#
def measure_covariance(arr) :
	means = measure_mean(arr)
	dim = len(arr[0])
	cov = np.zeros(shape=(dim, dim))
	for i in range(dim) :
		for j in range(dim) :
			cov[i][j] = np.mean( [ (x[i]-means[i])*(x[j]-means[j]) for x in arr ] )
	return cov


## Get covariance matrix from correlation and uncertainty amplitudes
#
def get_covariance(corr, amp) :
	e = np.diag(amp)
	return np.matmul(e, np.matmul(corr, e))


## Get eigendirections and associated error amplitudes using the covariance matrix
#
def get_covariance_eigendirections_and_amplitudes (cov) :
	w, v = np.linalg.eig(cov)
	w = np.sqrt(w)
	return v, w


## Get mean and covariance from toys
#
def get_mean_and_covariance (toys) :
	if len(toys) < 2 : raise RuntimeError("stats.get_mean_and_covariance(): number of toys must be at least 2").with_traceback(sys.exc_info()[2])
	length = len(toys[0])
	means = [ np.mean([t[i] for t in toys]) for i in range(2) ]
	cov = [ [ [] for j in range(2) ] for i in range(2) ]
	for i in range(2) :
		for j in range(2) :
			cov[i][j] = []
			for t in toys :
				cov[i][j].append( (t[i]-means[i]) * (t[j]-means[j]) )
			cov[i][j] = np.mean(cov[i][j])
	return means, np.array(cov)


## Get CL for given grid of confidence levels
#
def get_limit_from_levels (grid_of_CL, coverage=None) :
	if coverage is None :
		global target_coverage
		coverage = target_coverage
	if len(grid_of_CL.values.shape) == 1 :
		return np.interp([1-coverage], grid_of_CL.values, grid_of_CL.axes[0])[0]
	if len(grid_of_CL.values.shape) == 2 :
		fig = plt.figure()
		con = plt.contour(grid_of_CL.axes[0], grid_of_CL.axes[1], grid_of_CL.values.transpose(), [0.05], colors="orange", alpha=0.01)
		ret = con.collections[0].get_segments()
		plt.close(fig)
		return ret
	raise NotImplementedError("stats.get_limit_from_levels(): only implemented for 1D or 2D grid")

'''	for idx in range(len(grid_of_CL.values)) :
		if grid_of_CL.values[idx] < 1.0 - coverage : continue
		return grid_of_CL.axes[0][idx]
	return grid_of_CL.axes[0][-1]'''


## Get CL for given
#
def get_CL(this_dist, predicted_dist, SM_dist=None, test_stat=None, test_stat_strategy=None, limits_method=None) :
	global test_statistics, limit_methods, test_stat_strategies
	if test_stat          is None : test_stat          = get_test_stat()
	if test_stat_strategy is None : test_stat_strategy = get_test_stat_strategy()
	if limits_method      is None : limits_method      = get_limits_method()
	if test_stat          not in test_statistics      : raise ValueError(f"stats.get_CL(): test_stat value {test_stat} not in list of known values {test_statistics}")
	if limits_method      not in limit_methods        : raise ValueError(f"stats.get_CL(): limits_method values {limits_method} not in list of known values {limit_methods}")
	if test_stat_strategy not in test_stat_strategies : raise ValueError(f"stats.get_CL(): test_stat_strategy value {test_stat_strategy} not in list of known values {test_stat_strategies}")
	if test_stat != "chi2": raise NotImplementedError(f"stats.get_CL(): not implemented for test_stat = {test_stat}")
	if test_stat_strategy != "asymptotic" : raise NotImplementedError(f"stats.get_CL(): not implemented for test_stat_strategy = {test_stat_strategy}")
	if limits_method not in ["CLs", "CLs+b"] : raise NotImplementedError(f"stats.get_CL(): not implemented for limits_method = {limits_method}")
	ndof = len(this_dist)
	CL_sb = 1.0 - stats.chi2.cdf(this_dist.chi2(predicted_dist), ndof)
	if limits_method == "CLs+b" :
		return CL_sb
	if limits_method == "CLs" :
		if SM_dist is None :
			ValueError("stats.get_CL(): CLs method requires a SM prediction (key SM_dist) to be provided")
		CL_b = 1.0 - stats.chi2.cdf(this_dist.chi2(SM_dist), ndof)
		return CL_sb/CL_b
	NotImplementedError(f"stats.get_CL(): limits_method {limits_method} not implemented")


##  Get CL across a scan grid
#
def get_CL_across_grid (this_dist, BSM_dist_grid, SM_dist=None) :
	ret = Grid(BSM_dist_grid)
	flat_array_of_predictions = ret.values.flatten()
	flat_array_of_CL = np.zeros(shape=flat_array_of_predictions.shape, dtype=np.float64)
	for idx in range(len(flat_array_of_CL)) :
		#print("------->", idx)
		flat_array_of_CL[idx] = get_CL(this_dist, flat_array_of_predictions[idx], SM_dist=SM_dist)
	ret.values = flat_array_of_CL.reshape(ret.values.shape)
	return ret

