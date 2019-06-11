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
cov_to_fit = None
inv_cov_to_fit = None
l_norm = None


'''
#          This is the likelihood following a chi2 distribution
#  Likelihood
#
def likelihood (params) :
	if len(params) is not 2 :
		msg.fatal("Confidence.fitting.likelihood","Likelihood can only be evaluated for parameter lists of length 2, whereas {0} was provided".format(len(params)))
	global model_to_fit, data_to_fit, inv_cov_to_fit, l_norm
	model_to_fit.coefficients = params
	pred = model_to_fit.generate_prediction()
	res = data_to_fit - pred
	v = np.matmul(res, np.matmul(inv_cov_to_fit, res))
	return l_norm * np.exp(-0.5*v)
	return chi2.pdf(v, len(res))   #   doesn't work -- why?
'''


#          This is the likelihood calculated as the product of Poissons (unbiased, but slow, often results in overflow errors for large values, and only works for integer k unless we approximate k! as Gamma(k))
#  Likelihood
#
def likelihood (params) :
	if len(params) is not 2 :
		msg.fatal("Confidence.fitting.likelihood","Likelihood can only be evaluated for parameter lists of length 2, whereas {0} was provided".format(len(params)))
	global model_to_fit, data_to_fit, inv_cov_to_fit, l_norm
	model_to_fit.coefficients = params
	pred = model_to_fit.generate_prediction()
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


'''
                 #   L = L * poisson.pmf(data_to_fit[i], pred[i])
'''


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
		limit = tuple( (None,None) for i in range(len(values)) ),
		name = tuple( ["mu_bkg" , "mu_sig"] ),
		pedantic=True)
	m.migrad()
	if kwargs.get("minos",False) is True :
		m.minos()
	return m


#  Do fit and return the minimised TNLL
#
def get_fitted_TNLL (values, errors, fix) :
	m = do_fit(values, errors, fix)
	return m.get_fmin().fval


#  Do fit and return the minimised TNLL as well as mu_sig and it's error
#
def get_fitted_TNLL_and_signal_strength (values, errors, fix, **kwargs) :
	m = do_fit(values, errors, fix)
	if "profile" in kwargs :
		m.draw_profile(kwargs["profile"], subtract_min=True)
	return m.get_fmin().fval, m.values["mu_sig"], m.errors["mu_sig"]


#  Mkae copy of model and set it as the fit model
#
def set_fit_model (model) :
	global model_to_fit
	model_to_fit = Model(model.templates)


#  Set global data and covariance information
#
def set_data_and_covariance (values, errs, correlation) :
	global data_to_fit, cov_to_fit, inv_cov_to_fit, l_norm
	data_to_fit = values
	cov_to_fit = np.zeros(shape=correlation.shape,dtype=np.double)
	for i in range(len(errs)) :
		for j in range(len(errs)) :
			cov_to_fit[i][j] = errs[i] * errs[j] * correlation[i][j]
	inv_cov_to_fit = np.linalg.inv(cov_to_fit)
	l_norm = np.fabs(np.linalg.det(cov_to_fit)) * ( (2*math.pi) ** len(values) )
	l_norm = 1. / np.sqrt(l_norm)


#  Expected uncertainty on sig
#
def get_expected_mu_sig_error (model, correlation) :
	set_fit_model(model)
	y, ey = model.generate_asimov()
	set_data_and_covariance(y, ey, correlation)
	m = do_fit((1.,1.), (0.5,0.5), (False,False))
	return m.errors["mu_sig"]


#  Throw toys around a model, fit them and return the distribution of q = L(s+b) / L(b). Plot the pulls on mu_sig if desired.
#
def throw_and_fit_toys (model, correlation, n_toys, **kwargs) :
	set_fit_model(model)
	mu_true = model.coefficients[1]
	toys_q = []
	pulls_mu_sig = []
	for i in range(n_toys) :
		if i>0 and (10*i) % n_toys == 0 :
			msg.info("Confidence.DM_fitting.throw_and_fit_toys", "Processing toy {0} out of {1}".format(i,n_toys), verbose_level=1)
		n_tries = 0
		while n_tries > -1 :
			if n_tries == 1000 : msg.fatal("Confidence.DM_fitting.throw_and_fit_toys","Covariance matrix is still singular after 1000 attempts")
			try :
				y, ey = model.throw_toy()
				set_data_and_covariance(y, ey, correlation)
				n_tries = -1
			except Exception as e :
				msg.warning("Confidence.DM_fitting.throw_and_fit_toys","Error {0} occurred when throwing toy. Retrying.".format(e),verbose_level=1)
				n_tries = n_tries + 1
		tnll_s_b, mu_sig, mu_sig_err = get_fitted_TNLL_and_signal_strength( model.coefficients, (0.5,0.5), (False,False) )
		if kwargs.get("plot_first_profile",False) is True :
			x, y = [], []
			for i in range(21) :
				this_x = mu_sig + (i*0.2 - 2.0)*mu_sig_err
				x.append(this_x)
				y.append( get_fitted_TNLL( (1.0,this_x), (0.5,0.5), (False,True) ) )
			fig  = plt.figure(figsize=(7,7))
			ax = fig.add_subplot(111)
			ax.plot(x, y, "-", c="r")
			plt.show()
			if type(plotting.document) is not None :
				fig.savefig(plotting.document, format='pdf')
			plt.close(fig)
		tnll_b = get_fitted_TNLL( (1.,0.), (0.5,0.5), (False,True)  )
		q = TNLL_to_likelihood(tnll_s_b) / TNLL_to_likelihood(tnll_b)
		toys_q.append( float(q) )
		pulls_mu_sig.append( (mu_sig-mu_true) / mu_sig_err )
	toys_q.sort()
	mean, sem, sd, se = stats.get_mean_sem_sd_se(pulls_mu_sig)
	msg.info("Confidence.DM_fitting.throw_and_fit_toys", "    Pulls mean  =  {0:.4f}  +/-  {1:.4f}".format(mean, sem))
	msg.info("Confidence.DM_fitting.throw_and_fit_toys", "    Pulls sd    =  {0:.4f}  +/-  {1:.4f}".format(sd, se))
	if kwargs.get("plot_pulls",False) is False :
		return toys_q
	msg.info("Confidence.DM_fitting.throw_and_fit_toys","Plotting pulls for mu_true = {0:.5f}".format(mu_true))
	plotting.plot_histo_from_data(400, [pulls_mu_sig], xmin=-5, xmax=5, ymin=0., xlabel=kwargs.get("pulls_xlabel","Pull"), ylabel="Num. toys", labels=[kwargs.get("pulls_label","")], overlay_gauss=True)
	return toys_q
