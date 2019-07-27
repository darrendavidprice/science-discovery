##  =======================================================================================================================
##  Brief :  module for performing common stat operations
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  =======================================================================================================================


import numpy as np
import matplotlib.pyplot as plt


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
	if len(toys) < 2 : raise RuntimeError("stats.get_mean_and_covariance(): number of toys must be at least 2")
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
def get_limit_from_levels (grid_of_CL, coverage) :
	if len(grid_of_CL.values.shape) == 1 :
		return np.interp([1-coverage], grid_of_CL.values, grid_of_CL.axes[0])[0]
	if len(grid_of_CL.values.shape) == 2 :
		fig = plt.figure()
		con = plt.contour(grid_of_CL.axes[0], grid_of_CL.axes[1], grid_of_CL.values.transpose(), [1.-coverage], colors="orange", alpha=0.01)
		ret = con.collections[0].get_segments()
		plt.close(fig)
		return ret
	raise NotImplementedError("stats.get_limit_from_levels(): only implemented for 1D or 2D grid")

