##  =======================================================================================================================
##  Brief: interface to configparser
##  Author: Stephen Menary
##  Email: sbmenary@gmail.com
##  =======================================================================================================================

import configparser

import utils2.utils.globals_and_fallbacks as     glob
import utils2.utils.utils                 as     utils
from   utils2.objects.InputStore          import InputStore as InputStore
from   utils2.objects.ScanParam           import ScanParam  as ScanParam
from   utils2.stats.enums                 import LimitsMethod, TestStatistic, TestStatisticStrategy
from   utils2.misc.enums                  import BSMPredictionMethod


##  Load a config file
#
def read_config (fname, update_fallback=False) :
	config = configparser.ConfigParser()
	config.read(fname)
	if glob.config is None : glob.config = config
	if update_fallback : set_global_constants_from_config(config)
	return config


##  Get an entry from a config file whilst enforcing some of its properties
#
def get_and_enforce(section, name, fallback=None, config=None, to_type=None, from_selection=None) :
	if config is None : config = glob.config
	if config is None : raise ValueError("config.get_and_enforce(): no config provided and no fallback set")
	entry = config.get(section, name, fallback=fallback)
	try :
		entry = utils.string_to_objects(entry)
	except Exception as e :
		pass
	if to_type is not None :
		try : entry = to_type(entry)
		except ValueError : raise ValueError(f"config.get_and_enforce(): could not cast {entry} to type {to_type}")
	if from_selection is not None :
		if type(from_selection) is not list :
			raise ValueError("config.get_and_enforce(): argument from_selection must be a list")
		if entry not in from_selection :
			raise ValueError(f"config.get_and_enforce(): section {section} entry {name} with value {entry} not found in allowed list {from_selection}")
	return entry


##  Create (or update) an input store using a config file
#
def load_cfg_to_input_store (config=None, input_store=None, update_fallback=False, **kwargs) :
	if config      is None : config = glob.config
	if config      is None : raise ValueError("config.load_cfg_to_input_store(): no config provided and no fallback set")
	if input_store is None and update_fallback : input_store = glob.input_store
	if input_store is None :
		input_store = InputStore(config, **kwargs)
	else :
		input_store.load_from_config(config, **kwargs)
	if update_fallback : glob.input_store = input_store
	return input_store


##  Get a list of scan param names from config file
#
def get_scan_param_names (config=None) :
	if config is None : config = glob.config
	if config is None : raise ValueError("config.get_scan_param_names(): no config provided and no fallback set")
	return utils.string_to_object(config.get("PARAMS", "ScanParams", fallback="[]"))


##  Get the scan params from config file
#
def load_cfg_to_scan_params (config=None, update_fallback=False) :
	if config      is None : config = glob.config
	if config      is None : raise ValueError("config.load_cfg_to_scan_params(): no config provided and no fallback set")
	scan_params = []
	param_name_list = get_scan_param_names(config=config)
	for param_name in param_name_list :
		limits   = utils.string_to_object(config.get("PARAMS", f"{param_name}.scan.limits"  , fallback="[]"))
		stepsize = utils.string_to_object(config.get("PARAMS", f"{param_name}.scan.stepsize", fallback="None"))
		n_points = utils.string_to_object(config.get("PARAMS", f"{param_name}.scan.n_points", fallback="None"))
		label    = str(config.get("PARAMS", f"{param_name}.label", fallback=""))
		units    = str(config.get("PARAMS", f"{param_name}.units", fallback=""))
		scan_params.append(ScanParam(name=param_name, limits=limits, stepsize=stepsize, n_points=n_points, label=label, units=units))
	if update_fallback :
		glob.scan_params = scan_params
	return scan_params


##  Set fallbacks using config file values
#
def set_global_constants_from_config (config=None) :
	if config is None : config = glob.config
	if config is None : raise ValueError("config.set_global_constants_from_config(): no config provided and no fallback set")
	entry_BSM_prediction_method  = config.get("STEERING", "PredictionMethod"          , fallback=None)
	entry_test_stat              = config.get("GENERAL" , "TestStatistic"             , fallback=None)
	entry_test_stat_distribution = config.get("GENERAL" , "TestStatistic.Distribution", fallback=None)
	entry_limits_method          = config.get("GENERAL" , "LimitsMethod"              , fallback=None)
	entry_confidence_level       = config.get("GENERAL" , "ConfidenceLevel"           , fallback=None)
	if entry_BSM_prediction_method  != None : glob.BSM_prediction_method = BSMPredictionMethod  [entry_BSM_prediction_method ]
	if entry_test_stat              != None : glob.test_statistic        = TestStatistic        [entry_test_stat             ]
	if entry_test_stat_distribution != None : glob.test_stat_strategy    = TestStatisticStrategy[entry_test_stat_distribution]
	if entry_limits_method          != None : glob.limits_method         = LimitsMethod         [entry_limits_method         ]
	if entry_confidence_level       != None : glob.confidence_level      = float(entry_confidence_level)
	glob.custom_store["QuickStoreDistributions"] = config.getboolean("GENERAL", "QuickStoreDistributions", fallback=False)
	glob.custom_store["QuickLoadDistributions" ] = config.getboolean("GENERAL", "QuickLoadDistributions" , fallback=False)
	glob.custom_store["QuickStoreSMToys"       ] = config.getboolean("GENERAL", "QuickStoreSMToys"       , fallback=False)
	glob.custom_store["QuickLoadSMToys"        ] = config.getboolean("GENERAL", "QuickLoadSMToys"        , fallback=False)
	load_cfg_to_scan_params(config, update_fallback=True)
	load_cfg_to_input_store(config, update_fallback=True, look_for_params=True)


