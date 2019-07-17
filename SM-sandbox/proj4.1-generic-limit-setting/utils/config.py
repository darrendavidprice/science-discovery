##  =======================================================================================================================
##  Brief: interface to configparser
##  Author: Stephen Menary
##  Email: sbmenary@gmail.com
##  =======================================================================================================================

import configparser
import ast


config = None


def read_config (fname) :
	global config
	if config is None :
		config = configparser.ConfigParser()
	config.read(fname)
	return config


def get_and_enforce(section, name, fallback=None, cfg=None, to_type=None, from_selection=None) :
	global config
	if cfg is None : cfg = config
	str_entry = cfg.get(section, name, fallback=fallback)
	try : entry = ast.literal_eval(str_entry)
	except Exception as e : entry = str_entry
	if entry is None : raise KeyError(f"config.get_and_enforce(): entry {name} not found in section {section}")
	if to_type is not None :
		if to_type is bool  : return cfg.getbool(section, name, fallback=fallback)
		if to_type is float : return cfg.getfloat(section, name, fallback=fallback)
		return to_type(entry)
	if from_selection is not None :
		if type(from_selection) is not list : raise ValueError("config.get_and_enforce(): argument from_selection must be a list")
		if entry not in from_selection :
			raise ValueError(f"config.get_and_enforce(): section {section} entry {name} with value {entry} not found in allowed list {from_selection}")
	return entry


def get_scan_param_names (cfg=None) :
	global config
	if cfg is None : cfg = config
	return ast.literal_eval(cfg.get("PARAMS", "ScanParams", fallback="[]"))

