# ====================================================================================================
#  Brief:  Some useful stat functions
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================


import numpy as np
import scipy.stats as st


def get_mean(values) :
	return np.mean(values)


def get_sd(values) :
	return np.std(values)


def get_sem(values) :
	return st.sem(values)


def get_mean_sem_sd_se(values) :
	E_x, E_x2 = 0., 0.
	n = float(len(values))
	for v in values :
		E_x = E_x + v
		E_x2 = E_x2 + v*v
	E_x = E_x / (n-1)
	E_x2 = E_x2 / (n-1)
	sd = np.sqrt(E_x2 - E_x*E_x)
	sem = sd / np.sqrt(n)
	se = sd / np.sqrt(2*n - 2)
	return E_x, sem, sd, se