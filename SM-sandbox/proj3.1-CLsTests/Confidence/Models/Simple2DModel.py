#  ========================================================================
#  Brief:   Define simple 2D model
#  Author:  Stephen Menary (stmenary@cern.ch)
#  ========================================================================


from .Models import Model, Template
import numpy as np
from scipy.stats import norm


#  Define some arbitrary SM bkg prediction
#
def SM_bkg_1 (x_lo, x_hi, x) :
	x = (x-x_lo) / (x_hi-x_lo)
	return 10. * (1. + .2*x)


#  Define some arbitrary signal prediction
#
def signal_1 (x_lo, x_hi, x) :
	x = (x-x_lo) / (x_hi-x_lo)
	return 3.2 * (1. - x) * (1. - x)


#  Define some arbitrary signal prediction
#
def signal_2 (x_lo, x_hi, x) :
	x = (x-x_lo) / (x_hi-x_lo)
	return 3.2 * x * x


#  Return some arbitrary prediction according to an external function (as defined above)
#
def get_prediction (x_lo, x_hi, num_x, func, **kwargs) :
	scale = kwargs.get("scale",1.)
	ret = np.zeros(shape=(num_x),dtype=np.double)
	x_interval = (x_hi-x_lo) / (num_x-1)
	for i in range(num_x) :
		x = x_lo + i*x_interval
		ret[i] = 5. * func(x_lo,x_hi,x)
	return scale * ret


#  Return a simple hard-coded DM model according to some global 
#
def create_simple_2D_model_1 (**kwargs) :
	scale_bkg = kwargs.get("scale_bkg", 1.0)
	scale_sig = kwargs.get("scale_sig", 1.0)
	scale_sig1 = kwargs.get("scale_sig1", scale_sig)
	scale_sig2 = kwargs.get("scale_sig2", scale_sig)
	x_low = kwargs.get("x_low", 5.)
	x_high = kwargs.get("x_high", 15.)
	n_bins = int(kwargs.get("n_bins", 15))
	model = Model()
	model.append(get_prediction(x_low, x_high, n_bins, SM_bkg_1, scale=scale_bkg))
	model.append(get_prediction(x_low, x_high, n_bins, signal_1, scale=scale_sig1))
	model.append(get_prediction(x_low, x_high, n_bins, signal_2, scale=scale_sig2))
	return model
