from iminuit import Minuit
import math
import numpy as np
import matplotlib.pyplot as plt


#   Brief: global vars for minimiser
#
fit_data = None
fit_model = None
fit_cov = None
fit_cov_inv = None
fit_cov_det = None
fit_TNLL_const = 0.


def chi2(val) :
	global fit_data, fit_model, fit_cov_inv
	pred = fit_model*val[0]
	res = fit_data - pred
	chi2 = np.matmul(res, np.matmul(fit_cov_inv, res))
	return chi2


def TNLL(val) :
	global fit_TNLL_const
	return fit_TNLL_const + chi2(val)


def set_model_and_covariance(data, cov) :
	global fit_model, fit_cov, fit_cov_inv, fit_cov_det, fit_TNLL_const
	fit_model = data
	fit_cov = cov
	fit_cov_inv = np.linalg.inv(cov)
	fit_cov_det = np.linalg.det(cov)
	fit_TNLL_const = np.log(2*math.pi) + 2*np.log(fit_cov_det)


def fit_bkg() :
	#fig = plt.figure(figsize=(7,7))
	#ax = fig.add_subplot(111)
	#global fit_data, fit_model, fit_data_errs
	#x = np.array([x for x in range(len(fit_data))])
	#plt.errorbar(x, fit_data, yerr=fit_data_errs, marker='x', ls='')
	#plt.plot(x, fit_model, 'o')
	#plt.show()
	m = Minuit.from_array_func(
		TNLL,
		(1., 0.),
		error = (0.1, 0.1),
		fix = (False, True),
		errordef = 1.,
		name = tuple( ["mu_bkg" , "dummy"] ))
	m.migrad()
	m.minos()
	return m.values["mu_bkg"], m.merrors[("mu_bkg",-1)], m.merrors[("mu_bkg",1)]