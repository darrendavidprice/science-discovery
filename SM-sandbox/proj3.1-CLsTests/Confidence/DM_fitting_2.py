# ====================================================================================================
#  Brief:  Fitting code and likelihood definitions, specific to the "one bkg, one sig" case studied here
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================

import math

import numpy as np
from iminuit import Minuit
from scipy.stats import poisson, chi2
from scipy.special import gamma
import matplotlib.pyplot as plt

import Confidence.messaging as msg
import Confidence.stats as stats
from .Models.Models import Model
import Confidence.Plotting.plotting as plotting


#  Global: minimisation model
model_to_fit = None
data_to_fit = None


#          This is the likelihood calculated as the product of Poissons (unbiased, but slow, often results in overflow errors for large values, and only works for integer k unless we approximate k! as Gamma(k))
#  Likelihood
#
def likelihood (params) :
	if len(params) is not 3 :
		msg.fatal("Confidence.DM_fitting_2.likelihood","Likelihood can only be evaluated for parameter lists of length 2, whereas {0} was provided".format(len(params)))
	global model_to_fit, data_to_fit
	if data_to_fit is None :
		msg.fatal("Confidence.DM_fitting_2.likelihood","No dataset provided!")
	if model_to_fit is None :
		msg.fatal("Confidence.DM_fitting_2.likelihood","No model provided!")
	model_to_fit.coefficients = params
	pred = model_to_fit.generate_prediction(fabs=True)
	L = 1.
	for i in range(len(pred)) :
		mu = pred[i]
		k = data_to_fit[i]
		if k % 1 == 0 :
			L = L * poisson.pmf(k, mu)
		else :
			P = np.exp(-1.*mu) * (mu ** k) / gamma(k)
			L = L * P
	return L


#  -2logL
#
def TNLL (params, **kwargs) :
	l = likelihood(params)
	return -2. * np.log(l)


#  TNLL --> likelihood
#
def TNLL_to_likelihood (tnll, **kwargs) :
	return np.exp(-0.5*tnll)


#  Do fit and return the test statistic q ( q = L(s+b) / L(b) )
#
def do_fit (values, errors, fix, **kwargs) :
	m = Minuit.from_array_func(
		TNLL,
		values,
		error = errors,
		errordef = 1.,
		fix = fix,
		limit = tuple( (None,None,None) for i in range(len(values)) ),
		name = tuple( ["mu_bkg" , "k1", "k2"] ),
		pedantic=True)
	m.migrad()
	if kwargs.get("hesse",True) is True :
		m.hesse()
	if kwargs.get("minos",False) is True :
		m.minos()
	return m


#  Do fit and return the minimised TNLL
#
def get_fitted_TNLL (values, errors, fix) :
	m = do_fit(values, errors, fix)
	return m.get_fmin().fval


#  Do fit and return the minimised TNLL as well as k1, k2 and their errors
#
def get_fitted_TNLL_and_kappas (values, errors, fix, **kwargs) :
	k_true = kwargs.get("k_true",None)
	m = do_fit(values, errors, fix)
	TNLL = m.get_fmin().fval
	if k_true is not None :
		values = (values[0], k_true[0], k_true[1])
		fix = (fix[0], True, True)
		m2 = do_fit(values, errors, fix)
		TNLL_true = m2.get_fmin().fval
	else :
		TNLL_true = -99
	#print(m.get_fmin().fval, m2.get_fmin().fval, m2.get_fmin().fval - m.get_fmin().fval )
	#return q, q_true, m.values["k1"], m.merrors[("k1",-1.0)], m.merrors[("k1",1.0)], m.values["k2"], m.merrors[("k2",-1.0)], m.merrors[("k2",1.0)]
	return TNLL, TNLL_true, m.values["k1"], m.errors["k1"], m.errors["k1"], m.values["k2"], m.errors["k2"], m.errors["k2"]


#  Make copy of model and set it as the fit model
#
def set_fit_model (model) :
	global model_to_fit
	model_to_fit = Model(model.templates)


#  Set global data and covariance information
#
def set_data (values) :
	global data_to_fit
	data_to_fit = values


#  Throw toys around a model, fit them and return the distribution of q = L(s+b) / L(b). Plot the pulls on mu_sig if desired.
#
def throw_and_fit_toys (model, n_toys, **kwargs) :
	set_fit_model(model)
	k1_true, k2_true = model.coefficients[1], model.coefficients[2]
	toys_q, toys_dTNLL, toys_k1, toys_k2 = [], [], [], []
	for i in range(n_toys) :
		if n_toys > 150 and (10*i) % n_toys == 0 :
			msg.info("Confidence.DM_fitting_2.throw_and_fit_toys", "Processing toy {0} out of {1}".format(i, n_toys), verbose_level=1)
		n_tries = 0
		while n_tries > -1 :
			if n_tries == 1000 : msg.fatal("Confidence.DM_fitting_2.throw_and_fit_toys","Toy does not converge after 1000 attempts")
			try :
				y, ey = model.throw_toy(fabs=True)
				set_data(y)
				tnll_s_b, tnll_s_b_true, k1, k1_err_lo, k1_err_hi, k2, k2_err_lo, k2_err_hi = get_fitted_TNLL_and_kappas( model.coefficients, (0.1,0.1,0.1), (False,False,False), k_true=(k1_true, k2_true) )
				tnll_b = get_fitted_TNLL( (1.,1.,0.), (0.1,0.1,0.1), (False,True,True) )
				q = TNLL_to_likelihood(tnll_s_b) / TNLL_to_likelihood(tnll_b)
				dTNLL = tnll_s_b_true - tnll_s_b
				n_tries = -1
			except Exception as e :
				msg.warning("Confidence.DM_fitting_2.throw_and_fit_toys","Error {0} occurred when fitting toy. Retrying.".format(e))
				n_tries = n_tries + 1
		toys_q.append( float(q) )
		toys_dTNLL.append( float(dTNLL) )
		toys_k1.append( (k1, k1_err_lo, k1_err_hi) )
		toys_k2.append( (k2, k2_err_lo, k2_err_hi) )
		#print( "{0}  {1}  with measurements  {2}  +/-  {3}   and  {4}  +/-  {5}".format(k1_true, k2_true, k1, k1_err, k2, k2_err) )
	toys_q.sort()
	toys_dTNLL.sort()
	return toys_q, toys_dTNLL, toys_k1, toys_k2
