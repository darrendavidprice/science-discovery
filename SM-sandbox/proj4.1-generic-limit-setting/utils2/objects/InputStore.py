##  ===========================================================================================================================
##  Brief :  InputStore class; loads distributions from input files of many types
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  ===========================================================================================================================

import sys
import pickle
import subprocess

import numpy as np

import utils2.utils.utils                 as     utils
import utils2.utils.globals_and_fallbacks as     glob
from   utils2.objects.Input               import Input
from   utils2.objects.Distribution        import Distribution

import HEP_data_utils.HEP_data_helpers      as     HD
from   HEP_data_utils.DistributionContainer import DistributionContainer


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
		subprocess.run(["python", "utils2/utils/py2_yoda_interface.py", "-o", ".tmp_processed_yoda.pickle", "-r", ".tmp_py2_yoda_info.record", f"{fname}"])
		yoda_inputs = pickle.load(open(".tmp_processed_yoda.pickle","rb"), encoding='latin1')
		new_entries = []
		look_for_params = kwargs.get("look_for_params", False)
		if "cfg" in kwargs and "key" in kwargs :
			cfg, key = kwargs["cfg"], kwargs["key"]
			fextract = utils.string_to_object(cfg.get("INPUTS", f"{key}.extract", fallback="[]"))
			for entry_name in fextract :
				includes_SM     = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.IncludesSM"     , fallback="[]"))
				value_keys      = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.keys.values"    , fallback="[]"))
				covariance_keys = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.keys.covariance", fallback="[]"))
				if covariance_keys not in ["use-errors", 0] :
					raise NotImplementedError("InputStore.load_from_yoda(): covariance not set as use-errors or 0, currently the only supported options")
				for value_key in value_keys :
					if value_key in yoda_inputs : continue
					raise KeyError(f"required key {value_key} has not been extracted from {fname}")
				if type(value_keys) is list and len(value_keys) > 0 and type(value_keys[0]) is str :
					values = np.concatenate( [np.array(yoda_inputs[value_key]["y"]) for value_key in value_keys] )
				elif type(value_keys) is int : values = np.zeros(shape=value_keys)
				else : values = value_keys
				if covariance_keys == 0 : cov = np.zeros(shape=(len(values), len(values)))
				else : cov = np.diag( np.concatenate( [np.array(yoda_inputs[value_key]["ey_hi"]) for value_key in value_keys] ) )
				params = {}
				if look_for_params :
					for param in glob.scan_params :
						params[param.name] = cfg.get("INPUTS", f"{key}.{entry_name}.{param.name}", fallback=None)
				new_entry = Input(name       =entry_name, 
								  type       =cfg.get("INPUTS", f"{key}.{entry_name}.type", fallback=""),
								  origin_file=fname,
								  keys       =value_keys,
								  params     =params,
								  dist       =Distribution(name=entry_name, values=values, cov=cov, includes_SM=includes_SM))
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
				includes_SM = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.IncludesSM"     , fallback="[]"))
				value_keys  = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.keys.values"    , fallback="[]"))
				cov_keys    = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.keys.covariance", fallback="[]"))
				if type(value_keys) is list and len(value_keys) > 0 and type(value_keys[0]) is str :
					values = np.concatenate( [my_tables[value_key]._dep_var._values for value_key in value_keys] )
				elif type(value_keys) is int : values = np.zeros(shape=value_keys)
				else : values = value_keys
				num_values = len(values)
				cov = np.zeros(shape=(num_values, num_values))
				params = {}
				if look_for_params :
					for param in glob.scan_params :
						params[param.name] = cfg.get("INPUTS", f"{key}.{entry_name}.{param.name}", fallback=None)
				for cov_key in cov_keys : cov = cov + my_tables[cov_key]._dep_var._values
				new_entry = Input(name       =entry_name, 
								  type       =cfg.get("INPUTS", f"{key}.{entry_name}.type", fallback=""),
								  origin_file=fname,
								  keys       =value_keys,
								  params     =params,
								  dist       =Distribution(name=entry_name, values=values, cov=cov, includes_SM=includes_SM))
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
				includes_SM = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.IncludesSM"     , fallback="[]"))
				value_keys  = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.keys.values"    , fallback="[]"))
				cov_keys    = utils.string_to_object(cfg.get("INPUTS", f"{key}.{entry_name}.keys.covariance", fallback="[]"))
				if type(value_keys) is list and len(value_keys) > 0 and type(value_keys[0]) is str :
					values = np.concatenate( [pickle_dict[value_key] for value_key in value_keys] )
				elif type(value_keys) is int : values = np.zeros(shape=value_keys)
				else : values = value_keys
				num_values = len(values)
				params = {}
				if look_for_params :
					for param in glob.scan_params :
						params[param.name] = cfg.get("INPUTS", f"{key}.{entry_name}.{param.name}", fallback=None)
				cov = np.zeros(shape=(num_values, num_values))
				for cov_key in cov_keys : cov = cov + pickle_dict[cov_key]
				new_entry = Input(name       =entry_name, 
								  type       =cfg.get("INPUTS", f"{key}.{entry_name}.type", fallback=""),
								  origin_file=fname,
								  keys       =value_keys,
								  params     =params,
								  dist       =Distribution(name=entry_name, values=values, cov=cov, includes_SM=includes_SM))
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
		inputs             = utils.string_to_object(cfg.get("INPUTS" , "Inputs"                   , fallback="[]"))
		self.do_quick_load = glob.custom_store.get("QuickStoreDistributions", False)
		self.do_quick_save = glob.custom_store.get("QuickLoadDistributions" , False)
		look_for_params    = kwargs.get("look_for_params", False)
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
					self.load_from_yoda   (fname, cfg=cfg, key=key, save=save_fname, look_for_params=look_for_params)
				elif ftype == "pickle" :
					self.load_from_pickle (fname, cfg=cfg, key=key, save=save_fname, look_for_params=look_for_params)
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