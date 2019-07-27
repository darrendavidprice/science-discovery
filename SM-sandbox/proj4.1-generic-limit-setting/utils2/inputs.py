##  =======================================================================================================================
##  Brief :  manage the manipulation of input objects
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  =======================================================================================================================

import numpy as np

from   utils2.objects.Distribution        import Distribution
from   utils2.objects.Input               import Input
import utils2.utils.globals_and_fallbacks as     glob
import utils2.utils.utils                 as     utils


##  Return a Distribution() object with specified values / covariance
#
def get_dist_from_input_store (values_dist_name, cov_dist_name, in_store=None, name="", n_toys=None) :
	if in_store is None : in_store = glob.input_store
	if in_store is None : raise ValueError("inputs.get_dist_from_input_store(): no in_store argument provided and no fallback set")
	if values_dist_name not in in_store :
		raise KeyError(f"inputs.get_dist_from_input_store(): no distribution called {values_dist_name} in input store")
	values = in_store[values_dist_name].dist.values
	if type(cov_dist_name) is str and cov_dist_name in in_store :
		cov = in_store[cov_dist_name].dist.cov
	else :
		try :
			inp = utils.string_to_object(str(cov_dist_name))
		except ValueError :
			raise KeyError(f"inputs.get_dist_from_input_store(): no distribution called {cov_dist_name} in inputs.input_store")
		if type(inp) in [list, np.ndarray] :
			cov = np.ndarray(inp)
			if cov.shape != (len(values), len(values)) :
				raise ValueError(f"inputs.get_dist_from_input_store(): covariance matrix {cov_dist_name} does not have the expected shape")
		else :
			try :
				cov = inp*np.eye(len(values))
			except TypeError :
				raise ValueError(f"inputs.get_dist_from_input_store(): {cov_dist_name} could not converted into a covariance matrix")
	return Distribution(values=values, cov=cov, name=name, n_toys=n_toys)


##  Return a Distribution() object describing the SM
#
def get_SM_dist_from_input_store (in_store=None, config=None, name="", key="theoretical") :
	if config is None : config = glob.config
	if config is None : raise ValueError("inputs.get_SM_dist_from_input_store(): no config provided and no global fallback set")
	values_dist_name = config.get    ("STEERING", f"SM.{key}.values")
	cov_dist_name    = config.get    ("STEERING", f"SM.{key}.cov")
	n_toys           = int(config.get("STEERING", f"SM.{key}.ntoys", fallback=1000))
	print(n_toys)
	return get_dist_from_input_store(values_dist_name, cov_dist_name, in_store=in_store, name=name, n_toys=n_toys)


##  Return a Distribution() object describing the measurement
#
def get_meas_dist_from_input_store (in_store=None, config=None, name="") :
	if config is None : config = glob.config
	if config is None : raise ValueError("inputs.get_meas_dist_from_input_store(): no config provided and no global fallback set")
	values_dist_name = config.get    ("STEERING", "meas.values")
	cov_dist_name    = config.get    ("STEERING", "meas.cov")
	n_toys           = int(config.get("STEERING", f"meas.ntoys", fallback=1000))
	return get_dist_from_input_store(values_dist_name, cov_dist_name, in_store=in_store, name=name, n_toys=n_toys)


##  Return Distribution() objects describing the BSM input points
#
def get_BSM_dists_from_input_store (in_store=None, config=None, prefix="", scan_params=None) :
	if config      is None : config   = glob.config
	if config      is None : raise ValueError("inputs.get_BSM_dists_from_input_store(): no config provided and no global fallback set")
	if in_store    is None : in_store = glob.input_store
	if in_store    is None : raise ValueError("inputs.get_BSM_dists_from_input_store(): no in_store provided and no global fallback set")
	if scan_params is None : scan_params = glob.scan_params
	if scan_params is None : raise ValueError("inputs.get_BSM_dists_from_input_store(): no scan_params provided and no global fallback set")
	ret = {}
	for BSM_input_name in utils.string_to_object(config.get("STEERING", "BSM.load")) :
		if BSM_input_name not in in_store :
			raise KeyError(f"inputs.get_BSM_dists_from_input_store(): no input {BSM_input_name} in input_store")
		i = in_store[BSM_input_name]
		values = []
		print(i.params)
		for scan_param in [p.name for p in scan_params] :
			if scan_param not in i.params :
				raise KeyError(f"inputs.get_BSM_dists_from_input_store(): input {BSM_input_name} has no param called {scan_param}")
			values.append(i.params[scan_param])
		ret[tuple(values)] = Distribution(i.dist, name=prefix+i.dist.name)
	return ret

