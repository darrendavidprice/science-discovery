##  =======================================================================================================================
##  Brief: 1-2D limit setting plot
##  Author: Stephen Menary
##  Email: sbmenary@gmail.com
##  =======================================================================================================================

import sys
import getopt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker  import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.lines   import Line2D
from matplotlib.patches import Patch
import scipy.stats
import pickle

import utils.config as config
import utils.stats  as stats
import utils.inputs as inputs
import utils.utils  as utils
from utils.Distribution import Distribution
from utils.Grid import Grid, create_param_grid



##  Some globally accessible objects (allows loading by other modules)
#
num_scan_params = None
meas_dist       = None
SM_model_dist   = None
SM_exp_dist     = None
SM_pred_dist    = None
BSM_input_dists = {}
BSM_scan_dists  = None
param_grid      = None


##  Parse command line arguments
#
def parse_command_line_arguments () :
	argv = sys.argv[1:]
	try :
		opts, rest = getopt.getopt(argv, "", ["save=", "tag=", "show="])
	except getopt.GetoptError as err :
		utils.error("parse_command_line_arguments()", "The following error was thrown whilst parsing command-line arguments")
		utils.fatal("parse_command_line_arguments()", err)
	if len(rest) is not 1 :
		raise ValueError(f"parse_command_line_arguments(): expected 1 unlabelled argument where {len(argv)} provided").with_traceback(sys.exc_info()[2])
	cfg_name = rest[0]
	save_fname, do_show, tag = None, True, None
	if not utils.is_file(cfg_name) :
		raise RuntimeError(f"parse_command_line_arguments(): config file {cfg_name} not found").with_traceback(sys.exc_info()[2])
	for option, value in opts :
		if option in ["--tag"]  :
			tag = str(value)
			utils.info("parse_command_line_arguments()", f"Labelling temporary files using the tag: {tag}")
		if option in ["--save"] :
			save_fname = str(value)
			utils.info("parse_command_line_arguments()", f"Opening plots file {save_fname}")
			utils.open_plots_pdf(save_fname)
		if option in ["--show"] :
			do_show = utils.string_to_object(value)
			if type(do_show) != bool : raise ValueError(f"parse_command_line_arguments(): --show value \"{value}\" could not be cast to a bool")
	return cfg_name, do_show, save_fname, tag


##  Do very general setup (things which might want to be invoked by separate programs)
#
def do_general_setup (prog_name="do_general_setup()") :
	global num_scan_params, meas_dist, SM_model_dist, SM_exp_dist, SM_pred_dist, BSM_input_dists, BSM_scan_dists, param_grid
	#
	#  parse arguments
	#
	utils.info(prog_name, "Parsing arguments")
	cfg_name, do_show_figs, plots_fname, tag = parse_command_line_arguments()
	#
	#  read config file
	#
	utils.info(prog_name, f"Loading config file {cfg_name}")
	config.read_config(cfg_name)
	#
	#  load scan param details
	#
	utils.info(prog_name, "Interpreting scan parameters")
	params = inputs.load_cfg_to_scan_params()
	num_scan_params = len(params)
	utils.info(prog_name, "Scan params are: " + " / ".join([p for p in config.get_scan_param_names()]))
	stats.set_target_coverage(config.config.getfloat("GENERAL", "ConfidenceLevel", fallback=0.95))
	utils.info(prog_name, f"Target coverage is {stats.target_coverage}")
	#
	#  load input files
	#
	utils.info(prog_name, "Loading input files")
	inputs.load_cfg_to_input_store(look_for_params=True)
	#
	#  get input measured / expected / SM / BSM distributions
	#
	meas_dist       = inputs.get_meas_dist(name="get_limits.py::meas")
	SM_model_dist   = inputs.get_SM_dist  (name="get_limits.py::SM::model", key="theoretical")
	SM_exp_dist     = inputs.get_SM_dist  (name="get_limits.py::SM::pred" , key="experimental")
	SM_pred_dist    = Distribution(SM_exp_dist)
	SM_pred_dist.cov= SM_pred_dist.cov + SM_model_dist.cov
	BSM_input_dists = inputs.get_BSM_distributions(prefix="get_limits.py::BSM::")
	utils.info(prog_name, f"Measured distribution is {meas_dist}")
	utils.info(prog_name, f"SM experimental distribution is {SM_exp_dist}")
	utils.info(prog_name, f"SM theoretical distribution is {SM_model_dist}")
	utils.info(prog_name, f"SM combined distribution is {SM_pred_dist}")
	for key, item in BSM_input_dists.items() :
		utils.info(prog_name, f"BSM input distribution at point {key} is {item}")
	#
	#  generate model distributions across BSM grid
	#
	utils.info(prog_name, "Generating param grid")
	param_grid = create_param_grid(params)
	utils.info(prog_name, "Populating predictions across param grid")
	BSM_scan_dists = inputs.generate_BSM_predictions(BSM_input_dists, param_grid, SM=SM_model_dist)
	utils.info(prog_name, "Adding SM to BSM")
	shape = BSM_scan_dists.values.shape
	BSM_scan_dists.values = BSM_scan_dists.values.flatten()
	for idx in range(len(BSM_scan_dists.values)) :
		BSM_scan_dists.values[idx] = BSM_scan_dists.values[idx] + SM_model_dist     # not using add_to_values as I want SM error to be included
		utils.info(prog_name, f"BSM scan distribution at index {idx} is {BSM_scan_dists.values[idx]}")
	BSM_scan_dists.values = BSM_scan_dists.values.reshape(shape)
	#
	#  return config options
	#
	return cfg_name, do_show_figs, plots_fname, tag


##  Get some object representing the 68% spread of limits (load from file if possible)
#
def get_toy_spread_of_limits(SM_exp_dist, SM_model_dist, BSM_scan_dists, n_toys, target_coverage=None, tag=None) :
	global param_grid
	do_store_toys = config.config.getboolean("GENERAL", "SaveAndLoadToys", fallback=False)
	if tag is None : fname = ".get_limits__get_toy_spread_of_1D_limits.pickle"
	else : fname = f".get_limits__get_toy_spread_of_1D_limits.{tag}.pickle"
	if target_coverage is None : target_coverage = stats.get_target_coverage()
	required_kwargs = {}
	required_kwargs["test_stat"] = config.get_and_enforce("GENERAL", "TestStatistic", from_selection=stats.test_statistics)
	required_kwargs["limits_method"] = config.get_and_enforce("GENERAL", "LimitsMethod" , from_selection=stats.limit_methods)
	required_kwargs["test_stat_strategy"] = config.get_and_enforce("GENERAL", "TestStatistic.Distribution", from_selection=stats.test_stat_strategies)
	required_kwargs["target_coverage"] = target_coverage
	required_kwargs["n_toys"] = n_toys
	required_kwargs["n_axes"] = len(BSM_scan_dists.axes)
	for idx in range(len(BSM_scan_dists.keys)) :
		required_kwargs[f"axis.index_{idx}"] = BSM_scan_dists.keys[idx]
	for axis, key in zip(BSM_scan_dists.axes, BSM_scan_dists.keys) :
		required_kwargs[f"{key}.length"] = len(axis)
		for idx in range(len(axis)) :
			required_kwargs[f"{key}.index_{idx}"] = axis[idx]
	if do_store_toys :
		load_success, pickle_file = utils.open_from_pickle(fname, **required_kwargs)
		if load_success :
			utils.info("get_toy_spread_of_1D_limits()", f"Sucessfully loaded toy limits from {fname}")
			return pickle_file["limits"], pickle_file["coverage"]
	SM_exp_toys = SM_pred_dist.generate_toys(n_toys)
	utils.info("get_toy_spread_of_1D_limits()", "Getting expected confidence levels for toys")

	SM_toy_limits = []
	grid_of_coverage = Grid(param_grid)
	grid_of_coverage.generate(dtype=np.float64)
	tmp_array_of_coverage = grid_of_coverage.values.flatten()
	for toy_idx in range(n_toys) :
		grid_of_this_CL = stats.get_CL_across_grid(SM_exp_toys[toy_idx], BSM_scan_dists, SM_dist=SM_model_dist)
		limit = stats.get_limit_from_levels(grid_of_this_CL, target_coverage)
		SM_toy_limits.append(limit)
		flattened_grid_of_this_CL = grid_of_this_CL.values.flatten()
		for i in range(len(tmp_array_of_coverage)) :
			this_CL = flattened_grid_of_this_CL[i]
			if this_CL > (1.0 - target_coverage) : continue
			tmp_array_of_coverage[i] = tmp_array_of_coverage[i] + 1
		pct_complete = 100. * (1+toy_idx) / n_toys
		if pct_complete % 10 == 0 :
			utils.info("get_toy_spread_of_1D_limits()", f"{int(pct_complete)}% toys processed")
	tmp_array_of_coverage = tmp_array_of_coverage / float(n_toys)
	grid_of_coverage.values = tmp_array_of_coverage.reshape(grid_of_coverage.values.shape)

	if len(BSM_scan_dists.axes) == 1 : SM_toy_limits.sort()
	if do_store_toys :
		utils.save_to_pickle(fname, {"limits": SM_toy_limits, "coverage": grid_of_coverage}, **required_kwargs)
	return SM_toy_limits, grid_of_coverage
 

def format_axis_from_config (section, cfg=None, axis=None, pedantic=True) :
	if cfg is None : cfg = config.config
	if cfg is None :
		if not pedantic : return
		raise ValueError("format_axis_from_config(): no cfg provided and config.config not set")
	if axis is None : axis = plt.gca()
	if axis is None :
		if not pedantic : return
		raise ValueError("format_axis_from_config(): no axis provided and plt.gca() is None")
	xlim        = utils.string_to_object(cfg.get(section, "axis.xlim"           , fallback="None"))
	x_minor_loc = utils.string_to_object(cfg.get(section, "xaxis.minor_locator" , fallback="None"))
	x_major_loc = utils.string_to_object(cfg.get(section, "xaxis.major_locator" , fallback="None"))
	x_ticks_pos = utils.string_to_object(cfg.get(section, "xaxis.ticks_position", fallback="None"))
	ylim        = utils.string_to_object(cfg.get(section, "axis.ylim"           , fallback="None"))
	y_minor_loc = utils.string_to_object(cfg.get(section, "yaxis.minor_locator" , fallback="None"))
	y_major_loc = utils.string_to_object(cfg.get(section, "yaxis.major_locator" , fallback="None"))
	y_ticks_pos = utils.string_to_object(cfg.get(section, "yaxis.ticks_position", fallback="None"))
	tick_params = utils.string_to_object(cfg.get(section, "axis.tick_params"    , fallback="None"))
	if xlim        is not None : axis.set_xlim(xlim)
	if x_minor_loc is not None : axis.xaxis.set_minor_locator(MultipleLocator(x_minor_loc))
	if x_major_loc is not None : axis.xaxis.set_major_locator(MultipleLocator(x_major_loc))
	if x_ticks_pos is not None : axis.xaxis.set_ticks_position(x_ticks_pos)
	if ylim        is not None : axis.set_ylim(ylim)
	if y_minor_loc is not None : axis.yaxis.set_minor_locator(MultipleLocator(y_minor_loc))
	if y_major_loc is not None : axis.yaxis.set_major_locator(MultipleLocator(y_major_loc))
	if y_ticks_pos is not None : axis.yaxis.set_ticks_position(y_ticks_pos)
	if tick_params is not None : axis.tick_params(tick_params)

	for label in utils.string_to_object( cfg.get(section, "Labels", fallback=[]) ) :
		plt.text( label[0], label[1], label[2], size="small" )

##  Main
#
def main () :
	#
	#  config and setup
	#
	cfg_name, do_show_figs, plots_fname, tag = do_general_setup(prog_name="get_limits.py")
	global num_scan_params, meas_dist, SM_model_dist, SM_exp_dist, BSM_input_dists, BSM_scan_dists, param_grid
	#
	#  get SM expected limit
	#
	utils.info("get_limits.py", "Getting expected confidence levels for scan points")
	grid_of_pred_CL = stats.get_CL_across_grid (SM_exp_dist, BSM_scan_dists, SM_dist=SM_model_dist)
	#
	#  get observed limit
	#
	utils.info("get_limits.py", "Getting observed confidence levels for scan points")
	grid_of_obs_CL  = stats.get_CL_across_grid (meas_dist, BSM_scan_dists, SM_dist=SM_model_dist)
	if num_scan_params == 1 :
		utils.info("get_limits.py", f"Observed {100.*stats.target_coverage:.2f}% confidence limit is {stats.get_limit_from_levels(grid_of_obs_CL )}")
		utils.info("get_limits.py", f"Expected {100.*stats.target_coverage:.2f}% confidence limit is {stats.get_limit_from_levels(grid_of_pred_CL)}")
	#
	#  generate SM toys and get limits
	#
	utils.info("get_limits()", "Getting expected spread of confidence levels for scan points")
	n_toys = config.get_and_enforce("STEERING", "SM.experimental.ntoys", to_type=int)
	if n_toys < 2 : raise ValueError(f"get_limits(): SM.experimental.ntoys must be at least 2 ({n_toys} provided)")
	utils.info("get_limits()", f"Throwing {n_toys} toys around the experimental SM expectation")
	SM_toy_limits, SM_coverage_grid = get_toy_spread_of_limits(SM_exp_dist, SM_model_dist, BSM_scan_dists, n_toys, tag=tag)
	if num_scan_params == 1 :
		utils.info("get_limits.py", f"Median {100.*stats.target_coverage:.2f}% limit of SM toys is {SM_toy_limits[int(0.5*n_toys)]:.0f}")
		utils.info("get_limits.py", f"16th percentile {100.*stats.target_coverage:.2f}% limit of SM toys is {SM_toy_limits[int(0.16*n_toys)]:.0f}")
		utils.info("get_limits.py", f"84th percentile {100.*stats.target_coverage:.2f}% limit of SM toys is {SM_toy_limits[int(0.84*n_toys)]:.0f}")
	#
	# plot
	#
	utils.set_mpl_style()
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	if num_scan_params == 1 :
		limit_obs = stats.get_limit_from_levels(grid_of_obs_CL )
		limit_SM  = stats.get_limit_from_levels(grid_of_pred_CL)
		limit_toys_5pc, limit_toys_16pc, limit_toys_median, limit_toys_84pc, limit_toys_95pc = SM_toy_limits[int(0.05*n_toys)], SM_toy_limits[int(0.16*n_toys)], SM_toy_limits[int(0.5*n_toys)], SM_toy_limits[int(0.84*n_toys)], SM_toy_limits[int(0.95*n_toys)]
		plt.axvspan(limit_toys_5pc , limit_toys_95pc           , color="darkorange", linestyle=None)
		plt.axvspan(limit_toys_16pc, limit_toys_84pc           , color="gold"      , linestyle=None)
		plt.plot([limit_toys_median, limit_toys_median], [0, 1], color="darkblue"  , linestyle="dashed", linewidth=1)
		plt.plot([limit_SM         , limit_SM ]        , [0, 1], color="green"     )
		plt.plot([limit_obs        , limit_obs]        , [0, 1], color="purple"    )
		ax.yaxis.set_visible(False)
		ax.set_ylim([0,1])
	else :
		plt.contourf(SM_coverage_grid.axes[0], SM_coverage_grid.axes[1], SM_coverage_grid.values.transpose(), [0.05, 0.16, 0.84, 0.95], linestyles=None, colors=["gold", "darkorange", "gold"])
		plt.contour(SM_coverage_grid.axes[0], SM_coverage_grid.axes[1], SM_coverage_grid.values.transpose(), [0.5], linestyles="dashed", colors=["darkblue"], linewidths=1)
		plt.contour(grid_of_pred_CL.axes[0], grid_of_pred_CL.axes[1], grid_of_pred_CL.values.transpose(), [1.0-stats.target_coverage], colors="green")
		plt.contour(grid_of_obs_CL.axes[0], grid_of_obs_CL.axes[1], grid_of_obs_CL.values.transpose(), [1.0-stats.target_coverage], colors="purple")
		'''for l in SM_toy_limits :
			l = l[0]
			plt.plot(l[:,0], l[:,1], "-", color="gold", linewidth=0.1, alpha=1)'''

		ax.set_ylabel(f"{inputs.scan_params[1].label}  [{inputs.scan_params[1].units}]")
		plt.ylabel(f"{inputs.scan_params[1].label}  [{inputs.scan_params[1].units}]", horizontalalignment='right', y=1.0, fontsize="large")

	format_axis_from_config("GET_LIMITS") ;

	ax.set_xlabel(f"{inputs.scan_params[0].label}  [{inputs.scan_params[0].units}]")

	plt.xlabel(f"{inputs.scan_params[0].label}  [{inputs.scan_params[0].units}]", horizontalalignment='right', x=1.0, fontsize="large")

	plt.legend( [Line2D([0], [0], color="purple"  , lw=2), 
				 Line2D([0], [0], color="green"   , lw=2), 
				 Line2D([0], [0], color="darkblue", linestyle="dashed", lw=1), 
				 Patch (          color="gold"    , linestyle=None), 
				 Patch (          color="darkorange", linestyle=None)],
				[f"Obs. ({100*stats.target_coverage:.0f}% $CL_s$)",
				 f"Exp. ({100*stats.target_coverage:.0f}% $CL_s$)",
				 "SM toys: median",
				 "SM toys: 68% coverage",
				 "SM toys: 95% coverage"],
				loc=utils.string_to_object(config.config.get("GET_LIMITS","legend.position",fallback="\'best\'")))

	if do_show_figs : plt.show()
	if plots_fname is not None : utils.save_figure(fig)
	utils.close_plots_pdf()



##  If called as script, call main()
#
if __name__ == "__main__" :
	main()

















	'''to_pickle = {}
	to_pickle["meas_values"] = meas_dist.values
	to_pickle["meas_cov"   ] = meas_dist.cov
	to_pickle["SM_model_values"] = SM_model_dist.values
	to_pickle["SM_model_cov"   ] = SM_model_dist.cov
	to_pickle["SM_exp_values"] = SM_exp_dist.values
	to_pickle["SM_exp_cov"   ] = SM_exp_dist.cov
	to_pickle["BSM400_values"] = BSM_scan_dists.values[0].values
	to_pickle["BSM400_cov"   ] = BSM_scan_dists.values[0].cov
	pickle.dump(to_pickle, open("tmp_store.pickle","wb"))
	exit()'''



	'''
	{
		obs_CL_str  = [f"{x:.2f}" for x in grid_of_obs_CL.values]
		pred_CL_str = [f"{x:.2f}" for x in grid_of_pred_CL.values]
		for i in range(len(grid_of_obs_CL.axes[0])) :
			print(f"L  /  OBS  /  EXP:\t\t{grid_of_obs_CL.axes[0][i]}\t\t{obs_CL_str[i]}\t\t{pred_CL_str[i]}")
	}
	'''

	'''
	{
		means1 = stats.measure_mean([x.values for x in SM_exp_toys])
		means2 = SM_exp_dist.values
		print(np.divide((means2-means1), means2))

		cov1 = stats.measure_covariance([x.values for x in SM_exp_toys])
		cov2 = SM_exp_dist.cov
		print(np.divide((cov2-cov1), cov2))
	}
	'''

	'''
	{
		plt.hist([t.chi2(SM_model_dist) for t in SM_exp_toys], bins=range(5,51,1))
		x = np.linspace(5, 50, 900)
		y = n_toys*scipy.stats.chi2.pdf(x, len(meas_dist))
		plt.plot(x, y)
		plt.show()
	}
	'''

'''
	grid_of_SM_coverage = Grid(param_grid)
	grid_of_SM_coverage.generate(dtype=np.float64)
	tmp_array_of_coverage = grid_of_SM_coverage.values.flatten()
	utils.info("get_limits.py", "Getting expected confidence limits for toys")
	for toy_idx in range(n_toys) :
		grid_of_this_CL = get_CL_across_grid(SM_exp_toys[toy_idx], BSM_scan_dists, SM_dist=SM_model_dist)
		flattened_grid_of_this_CL = grid_of_this_CL.values.flatten()
		for i in range(len(tmp_array_of_coverage)) :
			this_CL = flattened_grid_of_this_CL[i]
			if this_CL > (1.0 - target_coverage) : continue
			tmp_array_of_coverage[i] = tmp_array_of_coverage[i] + 1
		pct_complete = 100. * (1+toy_idx) / n_toys
		if pct_complete % 10 == 0 :
			utils.info("get_limits.py", f"{int(pct_complete)}% toys processed")
	for i in range(len(tmp_array_of_coverage)) :
		tmp_array_of_coverage[i] = tmp_array_of_coverage[i] / float(n_toys)
	grid_of_SM_coverage.values = tmp_array_of_coverage.reshape(grid_of_SM_coverage.values.shape)


	for i in range(len(tmp_array_of_coverage)) :
		tmp_array_of_coverage[i] = tmp_array_of_coverage[i] / float(n_toys)
	grid_of_SM_coverage.values = tmp_array_of_coverage.reshape(grid_of_SM_coverage.values.shape)

	for i in range(len(grid_of_SM_coverage.axes[0])) :
		print(f"{grid_of_SM_coverage.axes[0][i]}\t\t{grid_of_SM_coverage.values[i]}")
'''