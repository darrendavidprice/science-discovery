##  =======================================================================================================================
##  Brief: Input classes; load input files and store contents
##  Author: Stephen Menary
##  Email: sbmenary@gmail.com
##  =======================================================================================================================

import sys
import pickle
import subprocess
import ast

import numpy as np

import utils.utils as utils
import utils.config as config
from   utils.Distribution import Distribution
from   utils.Grid import Grid

import HEP_data_utils.HEP_data_helpers as HD
from HEP_data_utils.DistributionContainer import DistributionContainer


##  Global instances of InputStore (analagous of Singleton class) and other useful things
#
input_store = None
scan_params = []


##  Stores a single Distribution object read from a file
#
class Input :
	def __init__ (self, name="", type="", origin_file="", keys=[], params=None, dist=Distribution()) :
		self.name = name
		self.type = type
		self.origin_file = origin_file
		self.keys = keys
		self.dist = dist
		self.params = params
	def __str__ (self) :
		return f"Input \'{self.name}\' of type \'{self.type}\' with {len(self.dist)} entries"


##  Big store holding many Input objects
#
class InputStore :
	## Default constructor
	# 
	def __init__ (self, cfg=None, **kwargs) :
		self.entries = {}
		if cfg is None : return
		self.load_from_config(cfg, **kwargs)
	## Indexing
	#
	def __getitem__ (self, key) :
		return self.entries[key]
	## Is key in store?
	#
	def __contains__ (self, key) :
		return key in self.entries
	## Load inputs from a yoda file
	#
	def load_from_yoda (self, fname, **kwargs) :
		subprocess.run(["python", "utils/py2_yoda_interface.py", "-o", ".tmp_processed_yoda.pickle", "-r", ".tmp_py2_yoda_info.record", f"{fname}"])
		yoda_inputs = pickle.load(open(".tmp_processed_yoda.pickle","rb"), encoding='latin1')
		new_entries = []
		look_for_params = kwargs.get("look_for_params", False)
		if "cfg" in kwargs and "key" in kwargs :
			cfg, key = kwargs["cfg"], kwargs["key"]
			fextract = utils.string_to_object(cfg.get("INPUTS", f"{key}.extract", fallback="[]"))
			for entry_name in fextract :
				includes_SM = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.IncludesSM" , fallback="[]"))
				value_keys = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.keys.values" , fallback="[]"))
				covariance_keys = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.keys.covariance" , fallback="[]"))
				if type(covariance_keys) is not str or covariance_keys != "use-errors" :
					raise NotImplementedError("InputStore.load_from_yoda(): covariance not set as use-errors, currently the only supported option").with_traceback(sys.exc_info()[2])
				for value_key in value_keys :
					if value_key in yoda_inputs : continue
					raise KeyError(f"required key {value_key} has not been extracted from {fname}")
				if type(value_keys) is list and len(value_keys) > 0 and type(value_keys[0]) is str :
					values = np.concatenate( [np.array(yoda_inputs[value_key]["y"]) for value_key in value_keys] )
				elif type(value_keys) is int : values = np.zeros(shape=value_keys)
				else : values = value_keys
				cov = np.diag( np.concatenate( [np.array(yoda_inputs[value_key]["ey_hi"]) for value_key in value_keys] ) )
				params = {}
				if look_for_params :
					for param_name in config.get_scan_param_names() :
						params[param_name] = cfg.get("INPUTS", f"{key}.{entry_name}.{param_name}", fallback=None)
				new_entry = Input(name=entry_name, 
								  type=cfg.get("INPUTS", f"{key}.{entry_name}.type", fallback=""),
								  origin_file=fname,
								  keys=value_keys,
								  params=params,
								  dist=Distribution(name=entry_name, values=values, cov=cov, includes_SM=includes_SM))
				if entry_name in self.entries :
					utils.warning("InputStore.load_from_yoda()", f"Entry named {entry_name} already exists")
				self.entries[entry_name] = new_entry
				new_entries.append(entry_name)
		if not self.do_quick_save : return
		if "save" not in kwargs :
			utils.warning("InputStore.load_from_yoda", f"self.do_quick_save is True but no quicksave file specified for {fname}. Not saving.")
			return
		self.quick_save(kwargs["save"], new_entries)
	## Load inputs from a hepdata file
	#
	def load_from_hepdata (self, fname, **kwargs) :
		my_tables = DistributionContainer(f"InputStore.hepdata.tables.{fname}")
		HD.load_all_yaml_files(my_tables, fname)
		new_entries = []
		look_for_params = kwargs.get("look_for_params", False)
		if "cfg" in kwargs and "key" in kwargs :
			cfg, key = kwargs["cfg"], kwargs["key"]
			keyfile = cfg.get("INPUTS", f"{key}.file.hepdata.keyfile", fallback="") 
			if keyfile != "" : my_tables.load_keys(keyfile)
			fextract = utils.string_to_object(cfg.get("INPUTS", f"{key}.extract", fallback="[]"))
			for entry_name in fextract :
				includes_SM = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.IncludesSM" , fallback="[]"))
				value_keys = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.keys.values"    , fallback="[]"))
				cov_keys   = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.keys.covariance", fallback="[]"))
				if type(value_keys) is list and len(value_keys) > 0 and type(value_keys[0]) is str :
					values = np.concatenate( [my_tables[value_key]._dep_var._values for value_key in value_keys] )
				elif type(value_keys) is int : values = np.zeros(shape=value_keys)
				else : values = value_keys
				num_values = len(values)
				cov = np.zeros(shape=(num_values, num_values))
				params = {}
				if look_for_params :
					for param_name in config.get_scan_param_names() :
						params[param_name] = cfg.get("INPUTS", f"{key}.{entry_name}.{param_name}", fallback=None)
				for cov_key in cov_keys : cov = cov + my_tables[cov_key]._dep_var._values
				new_entry = Input(name=entry_name, 
								  type=cfg.get("INPUTS", f"{key}.{entry_name}.type", fallback=""),
								  origin_file=fname,
								  keys=value_keys,
								  params=params,
								  dist=Distribution(name=entry_name, values=values, cov=cov, includes_SM=includes_SM))
				if entry_name in self.entries :
					utils.warning("InputStore.load_from_hepdata()", f"Entry named {entry_name} already exists")
				self.entries[entry_name] = new_entry
				new_entries.append(entry_name)
		if not self.do_quick_save : return
		if "save" not in kwargs :
			utils.warning("InputStore.load_from_hepdata", f"self.do_quick_save is True but no quicksave file specified for {fname}. Not saving.")
			return
		self.quick_save(kwargs["save"], new_entries)
	## Load inputs from a hepdata file
	#
	def load_from_pickle (self, fname, **kwargs) :
		pickle_dict = pickle.load(open(fname,"rb"))
		new_entries = []
		look_for_params = kwargs.get("look_for_params", False)
		if "cfg" in kwargs and "key" in kwargs :
			cfg, key = kwargs["cfg"], kwargs["key"]
			fextract = utils.string_to_object(cfg.get("INPUTS", f"{key}.extract", fallback="[]"))
			for entry_name in fextract :
				includes_SM = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.IncludesSM" , fallback="[]"))
				value_keys  = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.keys.values"    , fallback="[]"))
				cov_keys    = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.keys.covariance", fallback="[]"))
				if type(value_keys) is list and len(value_keys) > 0 and type(value_keys[0]) is str :
					values = np.concatenate( [pickle_dict[value_key] for value_key in value_keys] )
				elif type(value_keys) is int : values = np.zeros(shape=value_keys)
				else : values = value_keys
				num_values = len(values)
				params = {}
				if look_for_params :
					for param_name in config.get_scan_param_names() :
						params[param_name] = cfg.get("INPUTS", f"{key}.{entry_name}.{param_name}", fallback=None)
				cov = np.zeros(shape=(num_values, num_values))
				for cov_key in cov_keys : cov = cov + pickle_dict[cov_key]
				new_entry = Input(name=entry_name, 
								  type=cfg.get("INPUTS", f"{key}.{entry_name}.type", fallback=""),
								  origin_file=fname,
								  keys=value_keys,
								  params=params,
								  dist=Distribution(name=entry_name, values=values, cov=cov, includes_SM=includes_SM))
				if entry_name in self.entries :
					utils.warning("InputStore.load_from_pickle()", f"Entry named {entry_name} already exists")
				self.entries[entry_name] = new_entry
				new_entries.append(entry_name)
		if not self.do_quick_save : return
		if "save" not in kwargs :
			utils.warning("InputStore.load_from_pickle", f"self.do_quick_save is True but no quicksave file specified for {fname}. Not saving.")
			return
		self.quick_save(kwargs["save"], new_entries)
	## Load inputs specified in a config file
	#
	def load_from_config (self, cfg, **kwargs) :
		if "INPUTS" not in cfg :
			utils.error("InputStore.load_from_config()", "No INPUTS in config file. Ignoring.")
			return
		inputs             = utils.string_to_object(cfg.get("INPUTS" , "Inputs", fallback="[]"))
		self.do_quick_load = utils.string_to_object(cfg.get("GENERAL", "LoadQuickAccessValues"    , fallback=False))
		self.do_quick_save = utils.string_to_object(cfg.get("GENERAL", "StoreValuesForQuickAccess", fallback=False))
		look_for_params = kwargs.get("look_for_params", False)
		for key in inputs :
			fname = cfg.get("INPUTS", f"{key}.file.path", fallback="")
			ftype = cfg.get("INPUTS", f"{key}.file.type", fallback="unknown")
			save_fname = "." + fname.replace("/", "_") + ".pickle"
			if self.do_quick_load and self.quick_load(save_fname, extract=utils.string_to_object(cfg.get("INPUTS", f"{key}.extract", fallback="[]")))  :
				utils.info("InputStore.load_from_config()", f"File {save_fname} quick-loaded in the place of {ftype} file {fname}")
			else :
				if ftype == "hepdata" :
					self.load_from_hepdata(fname, cfg=cfg, key=key, save=save_fname, look_for_params=look_for_params)
				elif ftype == "yoda" :
					self.load_from_yoda(fname, cfg=cfg, key=key, save=save_fname, look_for_params=look_for_params)
				elif ftype == "pickle" :
					self.load_from_pickle(fname, cfg=cfg, key=key, save=save_fname, look_for_params=look_for_params)
				else :
					raise ValueError(f"InputStore.load_from_config(): Input {key} file {fname} has an unrecognised type {ftype}")
	## Load inputs from a hepdata file
	#
	def quick_load (self, fname, extract=[]) :
		if not utils.is_file(fname) :
			return False
		try :
			new_entries = pickle.load(open(fname, "rb"))
			for required_entry in extract :
				if required_entry in new_entries : continue
				return False
			for key, item in new_entries.items() :
				self.entries[key] = item
			return True
		except Exception as e :
			pass
		return False
	## Save inputs to pickle file
	#
	def quick_save (self, fname, new_entries) :
		to_save = {}
		for entry in new_entries :
			to_save[entry] = self.entries[entry]
		pickle.dump(to_save, open(fname, "wb"))


##  ScanParam object
#
class ScanParam :
	##  Constructor
	#
	def __init__ (self, name="", limits=[], stepsize=None, n_points=None, label="", units="") :
		self.name = name
		self.limits = limits
		self.n_points = 0
		if stepsize != None and n_points != None : raise RuntimeError("ScanParam.__init__(): cannot specify both a stepsize and n_points")
		if stepsize != None : self.set_stepsize(stepsize)
		if n_points != None : self.n_points = n_points
		self.label = label
		self.units = units
	##  Length
	#
	def __len__ (self) :
		return len(self.scan_points())
	##  Set stepsize
	#
	def set_stepsize (self, stepsize) :
		n_limits = len(self.limits)
		if n_limits is not 2 :
			raise RuntimeWarning(f"ScanParam.set_stepsize(): parameter \'{self.name}\' has {n_limits} where 2 expected. Cannot set stepsize.")
			return
		scan_range = self.limits[1] - self.limits[0]
		if scan_range % stepsize != 0 :
			raise RuntimeWarning(f"ScanParam.set_stepsize(): parameter \'{self.name}\' with limits {self.limits} cannot be exactly split into bins of width {stepsize}. Rounding will occur.")
		self.n_points = 1 + int(scan_range/stepsize)
	##  Get array of scan points
	#
	def scan_points (self) :
		n_limits = len(self.limits)
		if n_limits is not 2 :
			raise ValueError(f"ScanParam.scan_points(): parameter \'{self.name}\' has {n_limits} limits where 2 expected")
		if self.n_points < 2 :
			raise ValueError(f"ScanParam.scan_points(): parameter \'{self.name}\' requested with {self.n_points} scan points (must be at least 2)")
		scan_range = self.limits[1] - self.limits[0]
		stepsize = scan_range / (self.n_points-1)
		return np.array([self.limits[0] + i*stepsize for i in range(self.n_points)])


##  Return a Distribution() object with specified values / covariance
#
def get_dist (values_dist_name, cov_dist_name, in_store=None, name="") :
	if in_store is None :
		global input_store
		in_store = input_store
	if values_dist_name not in in_store :
		raise KeyError(f"inputs.get_dist(): no distribution called {values_dist_name} in inputs.input_store")
	values = in_store[values_dist_name].dist.values
	if cov_dist_name == "0" :
		cov = np.zeros(shape=(len(values), len(values)))
	elif cov_dist_name == "100" :
		cov = 100*np.eye(len(values))
	elif cov_dist_name not in in_store :
		raise KeyError(f"inputs.get_dist(): no distribution called {cov_dist_name} in inputs.input_store")
	else :
		cov = in_store[cov_dist_name].dist.cov
	return Distribution(values=values, cov=cov, name=name)


##  Return a Distribution() object describing the SM
#
def get_SM_dist (in_store=None, name="", key="theoretical") :
	values_dist_name = config.config.get("STEERING", f"SM.{key}.values")
	cov_dist_name    = config.config.get("STEERING", f"SM.{key}.cov")
	return get_dist(values_dist_name, cov_dist_name, in_store=in_store, name=name)


##  Return a Distribution() object describing the measurement
#
def get_meas_dist (in_store=None, name="") :
	values_dist_name = config.config.get("STEERING", "meas.values")
	cov_dist_name    = config.config.get("STEERING", "meas.cov")
	return get_dist(values_dist_name, cov_dist_name, in_store=in_store, name=name)


##  Get BSM distributions
#
def get_BSM_distributions (in_store=None, prefix="", target_params=None) :
	if in_store == None :
		global input_store
		in_store = input_store
	if target_params == None :
		target_params = config.get_scan_param_names()
	ret = {}
	for BSM_input_name in utils.string_to_object(config.config.get("STEERING", "BSM.load")) :
		if BSM_input_name not in in_store :
			raise KeyError(f"inputs.get_BSM_distributions(): no input {BSM_input_name} in input_store")
		i = in_store[BSM_input_name]
		values = []
		for target_param in target_params :
			if target_param not in i.params :
				raise KeyError(f"inputs.get_BSM_distributions(): input {BSM_input_name} has no param called {target_param}")
			values.append(i.params[target_param])
		ret[tuple(values)] = Distribution(i.dist, name=prefix+i.dist.name)
	return ret


##  Get BSM predictions using ScaleByL6 method
#
def populate_scan_grid_using_ScaleByL6 (BSM_input_dists, new_grid, SM=None, target_params=None) :
	if target_params == None :
		target_params = config.get_scan_param_names()
	try :
		lambda_grid_index = new_grid.keys.index("Lambda")
		lambda_list_index = target_params.index("Lambda")
		utils.info("populate_input_grid_using_ScaleByL6()", "successfully found \'Lambda\' in param list")
	except ValueError as e :
		raise KeyError("populate_input_grid_using_ScaleByL6(): no parameter \'Lambda\' found in param list")
	n_dim = len(new_grid.keys)
	if n_dim == 1 :
		if len(BSM_input_dists) > 1 :
			raise RuntimeError("populate_input_grid_using_ScaleByL6(): don't know which input to scale Lambda from...")
		L_ref, dist_ref = 0., None
		for L, i in BSM_input_dists.items() :
			L_ref = float(L[0])
			dist_ref = i
		if dist_ref.includes_SM :
			if SM is None : raise ValueError("populate_scan_grid_using_ScaleByL6(): need to subtract SM from BSM input but none provided")
			dist_ref.subtract_values(SM.values)
			dist_ref.subtract_cov(SM.cov)
		L_values = new_grid.axes[0]
		for idx in range(len(L_values)) :
			sf = (L_ref/L_values[idx]) ** 6
			new_grid.values[idx] = dist_ref * sf
	elif n_dim == 2 :
		other_param_grid_index = 1 - lambda_grid_index
		other_param_list_index = 1 - lambda_list_index
		other_param_key = new_grid.keys[other_param_grid_index]
		for grid_idx_other_param in range(len(new_grid.axes[other_param_grid_index])) :
			other_param_value = new_grid.axes[other_param_grid_index][grid_idx_other_param]
			lambda_ref, dist_ref = None, None
			for key, item in BSM_input_dists.items() :
				if type(other_param_value)(key[other_param_list_index]) != other_param_value : continue
				if dist_ref is not None :
					raise RuntimeError("populate_input_grid_using_ScaleByL6(): don't know which input to scale Lambda from...")
				lambda_ref = np.float64(key[lambda_list_index])
				dist_ref = item
			if dist_ref.includes_SM :
				if SM is None : raise ValueError("populate_scan_grid_using_ScaleByL6(): need to subtract SM from BSM input but none provided")
				dist_ref.subtract_values(SM.values)
				dist_ref.subtract_cov(SM.cov)
			for grid_idx_lambda in range(len(new_grid.axes[lambda_grid_index])) :
				lambda_value = np.float64(new_grid.axes[lambda_grid_index][grid_idx_lambda])
				sf = (lambda_ref/lambda_value) ** 6
				if lambda_grid_index == 0 : idx_x, idx_y = grid_idx_lambda, grid_idx_other_param
				else : idx_x, idx_y = grid_idx_other_param, grid_idx_lambda
				new_grid.values[idx_x][idx_y] = dist_ref * sf
	elif n_dim > 2 :
		raise NotImplementedError(f"populate_input_grid_using_ScaleByL6(): only 1D and 2D scans implemented, {n_dim}D asked")
	return new_grid


##  Get BSM distributions at scan points
#
def generate_BSM_predictions (BSM_input_dists, param_grid=None, SM=None) :
	if param_grid == None :
		param_grid = generate_param_grid()
	new_grid = Grid(param_grid)
	new_grid.generate(dtype=Distribution)
	method = config.config.get("STEERING", "PredictionMethod", fallback=None)
	if method and method == "ScaleByL6" :
		populate_scan_grid_using_ScaleByL6(BSM_input_dists, new_grid, SM=SM)
	else :
		raise NotImplementedError("generate_param_grid(): only ScaleByL6 method currently implemented")
	return new_grid


##  Load inputs from config file
#
def load_cfg_to_input_store (cfg=None, **kwargs) :
	if cfg is None :
		cfg = config.config
	global input_store
	if input_store is None :
		input_store = InputStore(cfg, **kwargs)
	else :
		input_store.load_from_config(**kwargs)
	return input_store


##  Load scan params from config file
#
def load_cfg_to_scan_params (cfg=None) :
	if cfg is None :
		cfg = config.config
	global scan_params
	param_name_list = config.get_scan_param_names(cfg=cfg)
	for param_name in param_name_list :
		limits   = ast.literal_eval(cfg.get("PARAMS", f"{param_name}.scan.limits", fallback="[]"))
		stepsize = ast.literal_eval(cfg.get("PARAMS", f"{param_name}.scan.stepsize", fallback="None"))
		n_points = ast.literal_eval(cfg.get("PARAMS", f"{param_name}.scan.n_points", fallback="None"))
		label    = str(cfg.get("PARAMS", f"{param_name}.label", fallback=""))
		units    = str(cfg.get("PARAMS", f"{param_name}.units", fallback=""))
		scan_params.append(ScanParam(name=param_name, limits=limits, stepsize=stepsize, n_points=n_points, label=label, units=units))
	return scan_params



