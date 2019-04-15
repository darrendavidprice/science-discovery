import numpy as np


def get_1D_distribution ( dataset_ , key_ ) :
	dist = dataset_._distributions_1D[key_]
	bins = dist._bin_values
	x = [ 0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1) ]
	ex = [ 0.5*(bins[i+1]-bins[i]) for i in range(len(bins)-1) ]
	y = dist._values
	ey_lo = np.zeros(shape=(len(y)))
	ey_hi = np.zeros(shape=(len(y)))
	for key in dist._symm_errors :
		errs = dist._symm_errors[key]
		for i in range(0,len(errs)) :
			ey_lo[i] = ey_lo[i] + errs[i]*errs[i]
			ey_hi[i] = ey_hi[i] + errs[i]*errs[i]
	for key in dist._asymm_errors_up :
		errs1 = dist._asymm_errors_up[key]
		errs2 = dist._asymm_errors_down[key]
		for i in range(0,len(errs1)) :
			err1 = errs1[i]
			err2 = errs2[i]
			if err1 > 0 : ey_hi[i] = ey_hi[i] + err1*err1
			else : ey_lo[i] = ey_lo[i] + err1*err1
			if err2 > 0 : ey_hi[i] = ey_hi[i] + err2*err2
			else : ey_lo[i] = ey_lo[i] + err2*err2
	ey_lo = np.sqrt(ey_lo)
	ey_hi = np.sqrt(ey_hi)
	return x, y, [ey_lo,ey_hi], ex
