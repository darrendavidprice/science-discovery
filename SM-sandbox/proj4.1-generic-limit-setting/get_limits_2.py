##  =======================================================================================================================
##  Brief: 1-2D limit setting plot
##  Author: Stephen Menary
##  Email: sbmenary@gmail.com
##  =======================================================================================================================

import sys
import getopt
import pickle

import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.ticker  import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from   matplotlib.lines   import Line2D
from   matplotlib.patches import Patch

import utils2.config                      as     config
import utils2.inputs                      as     inputs
import utils2.prediction                  as     prediction
import utils2.utils.utils                 as     utils
import utils2.utils.plotting              as     plotting
import utils2.utils.globals_and_fallbacks as     glob
from   utils2.stats.CLGenerator           import CLGenerator
from   utils2.stats.CLGeneratorGrid       import CLGeneratorGrid
from   utils2.objects.Distribution        import Distribution
from   utils2.objects.Grid                import Grid, do_for_all_in_tensor, assign_for_all_in_tensor_from_friend
import utils2.stats.helpers               as     st


##  Parse command line arguments
#
def parse_command_line_arguments () :
	utils.info("parse_command_line_arguments()", "Parsing arguments")
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
			plotting.open_plots_pdf(save_fname)
		if option in ["--show"] :
			do_show = utils.string_to_object(value)
			if type(do_show) != bool : raise ValueError(f"parse_command_line_arguments(): --show value \"{value}\" could not be cast to a bool")
	glob.custom_store["config name"   ] = cfg_name
	glob.custom_store["do show plots" ] = do_show
	glob.custom_store["plots filename"] = save_fname
	glob.custom_store["tag"           ] = tag


##  Print the setup details
#
def print_setup () :
	utils.info("print_setup()", "Scan params are: " + " vs. ".join([f"{p}" for p in glob.scan_params]))
	utils.info("print_setup()", f"Using {glob.limits_method} method")
	utils.info("print_setup()", f"Using test statistic {glob.test_statistic} following the distribution strategy [{glob.test_stat_strategy}]")
	utils.info("print_setup()", f"Limits at confidence level {glob.confidence_level}")
	utils.info("print_setup()", "Measured distribution is {}"       .format(glob.custom_store["meas_dist"    ]))
	utils.info("print_setup()", "SM experimental distribution is {}".format(glob.custom_store["SM_exp_dist"  ]))
	utils.info("print_setup()", "SM theoretical distribution is {}" .format(glob.custom_store["SM_model_dist"]))
	utils.info("print_setup()", "SM combined distribution is {}"    .format(glob.custom_store["SM_pred_dist" ]))
	for key, item in glob.custom_store["BSM_input_dists"].items() :
		utils.info("print_setup()", f"BSM input distribution at point {key} is {item}")


##  Do very general setup (things which might want to be invoked by separate programs)
#
def do_general_setup (cfg_name=None) :
	if cfg_name is None :
		cfg_name  = glob.custom_store["config name"]
	do_show_plots = glob.custom_store.get("do show plots" , True)
	plots_fname   = glob.custom_store.get("plots filename", None)
	tag           = glob.custom_store.get("tag", None)
	#
	#  read config file
	#
	utils.info("do_general_setup()", f"Loading config file {cfg_name}")
	config.read_config(cfg_name, update_fallback=True)
	#
	#  get input measured / expected / SM / BSM distributions
	#
	utils.info("do_general_setup()", "Creating meas, SM and BSM distributions")
	meas_dist                            = inputs.get_meas_dist_from_input_store(name="get_limits.py::meas")
	SM_model_dist                        = inputs.get_SM_dist_from_input_store  (name="get_limits.py::SM::model", key="theoretical")
	SM_exp_dist                          = inputs.get_SM_dist_from_input_store  (name="get_limits.py::SM::pred" , key="experimental")
	SM_pred_dist                         = Distribution(SM_exp_dist)
	SM_pred_dist.cov                     = SM_pred_dist.cov + SM_model_dist.cov
	BSM_input_dists                      = inputs.get_BSM_dists_from_input_store(prefix="get_limits.py::BSM::")
	glob.custom_store["meas_dist"      ] = meas_dist
	glob.custom_store["SM_model_dist"  ] = SM_model_dist
	glob.custom_store["SM_exp_dist"    ] = SM_exp_dist
	glob.custom_store["SM_pred_dist"   ] = SM_pred_dist
	glob.custom_store["BSM_input_dists"] = BSM_input_dists
	#
	#  generate model distributions across BSM grid
	#
	utils.info("do_general_setup()", "Populating predictions across param grid")
	BSM_scan_dists = prediction.generate_BSM_predictions(BSM_input_dists, SM=SM_model_dist)
	utils.info("do_general_setup()", "Adding SM to BSM")
	def add_to_dist (x, **kwargs) : x.add(kwargs["SM_model_dist"])
	do_for_all_in_tensor(BSM_scan_dists.values, add_to_dist, SM_model_dist=SM_model_dist)
	glob.custom_store["BSM_scan_dists" ] = BSM_scan_dists
	#
	#  configure confidence level generators
	#
	CL_generator = CLGeneratorGrid(glob.scan_params, generate=True)
	CL_generator.set_distributions(BSM_scan_dists)
	CL_generator.set_SM_distribution(SM_model_dist)
	glob.CL_generator = CL_generator


##  Get some object representing the 68% spread of limits (load from file if possible)
#
def get_toy_spread_of_limits(n_toys=None, confidence_level=None, tag=None) :
	#
	#  resolve settings from defaults and those provided
	#
	SM_pred_dist   = glob.custom_store["SM_pred_dist"  ]
	BSM_scan_dists = glob.custom_store["BSM_scan_dists"]
	if tag              is None : tag              = glob.custom_store["tag"]
	if n_toys           is None : n_toys           = SM_pred_dist.n_toys
	if confidence_level is None : confidence_level = glob.confidence_level
	if tag is None : fname = ".get_limits__get_toy_spread_of_1D_limits.pickle"
	else           : fname = f".get_limits__get_toy_spread_of_1D_limits.{tag}.pickle"
	#
	#  specify the settings which must match when saving/loading results from file
	#
	required_kwargs = {}
	required_kwargs["test_stat"         ] = glob.test_statistic
	required_kwargs["limits_method"     ] = glob.limits_method
	required_kwargs["test_stat_strategy"] = glob.test_stat_strategy
	required_kwargs["confidence_level"  ] = confidence_level
	required_kwargs["n_toys"            ] = n_toys
	n_axes = len(BSM_scan_dists.axes)
	required_kwargs["n_axes"            ] = n_axes
	for idx in range(n_axes) : required_kwargs[f"axis.index_{idx}"] = BSM_scan_dists.keys[idx]
	for axis, key in zip(BSM_scan_dists.axes, BSM_scan_dists.keys) :
		required_kwargs[f"{key}.length"] = len(axis)
		for idx in range(len(axis)) :
			required_kwargs[f"{key}.index_{idx}"] = axis[idx]
	#
	#  load toys limits if specified *and* the saved settings match those required
	#
	if glob.custom_store.get("QuickLoadSMToys", False) :
		load_success, pickle_file = utils.open_from_pickle(fname, **required_kwargs)
		if load_success :
			utils.info("get_toy_spread_of_limits()", f"Sucessfully loaded toy limits from {fname}")
			return pickle_file["limits"], pickle_file["coverage"]
	#
	#  otherwise throw toys
	#
	utils.info("get_toy_spread_of_limits()", f"Throwing {n_toys} toys")
	SM_exp_toys = SM_pred_dist.generate_toys(n_toys)
	#
	#  and get the limits
	#
	utils.info("get_toy_spread_of_limits()", "Getting expected confidence limits for toys")
	SM_toy_limits = []
	grid_of_coverage = Grid(glob.scan_params, generate=True)
	tmp_array_of_coverage = grid_of_coverage.values.flatten()
	for toy_idx in range(n_toys) :
		grid_of_CL = glob.CL_generator.get_CL_grid(SM_exp_toys[toy_idx])
		limit      = st.get_limit_from_levels(grid_of_CL, confidence_level)
		SM_toy_limits.append(limit)
		flattened_grid_of_CL = grid_of_CL.values.flatten()
		for i in range(len(tmp_array_of_coverage)) :
			this_CL = flattened_grid_of_CL[i]
			if this_CL > (1.0 - confidence_level) : continue
			tmp_array_of_coverage[i] = tmp_array_of_coverage[i] + 1
		pct_complete = 100. * (1+toy_idx) / n_toys
		if pct_complete % 10 == 0 :
			utils.info("get_toy_spread_of_limits()", f"{int(pct_complete)}% toys processed")
	tmp_array_of_coverage   = tmp_array_of_coverage / float(n_toys)
	grid_of_coverage.values = tmp_array_of_coverage.reshape(grid_of_coverage.values.shape)
	#
	#  sort the limits
	#
	if len(BSM_scan_dists.axes) == 1 : SM_toy_limits.sort()
	if glob.custom_store.get("QuickStoreSMToys", False) :
		utils.save_to_pickle(fname, {"limits": SM_toy_limits, "coverage": grid_of_coverage}, **required_kwargs)
	return SM_toy_limits, grid_of_coverage
 

##  Format axis based on what was found in the config file
#
def format_axis_from_config (section, config=None, axis=None, pedantic=True) :
	if config is None : config = glob.config
	if config is None :
		if not pedantic : return
		raise ValueError("format_axis_from_config(): no config provided and config.config not set")
	if axis is None : axis = plt.gca()
	if axis is None :
		if not pedantic : return
		raise ValueError("format_axis_from_config(): no axis provided and plt.gca() is None")
	xlim        = utils.string_to_object(config.get(section, "axis.xlim"           , fallback="None"))
	x_minor_loc = utils.string_to_object(config.get(section, "xaxis.minor_locator" , fallback="None"))
	x_major_loc = utils.string_to_object(config.get(section, "xaxis.major_locator" , fallback="None"))
	x_ticks_pos = utils.string_to_object(config.get(section, "xaxis.ticks_position", fallback="None"))
	ylim        = utils.string_to_object(config.get(section, "axis.ylim"           , fallback="None"))
	y_minor_loc = utils.string_to_object(config.get(section, "yaxis.minor_locator" , fallback="None"))
	y_major_loc = utils.string_to_object(config.get(section, "yaxis.major_locator" , fallback="None"))
	y_ticks_pos = utils.string_to_object(config.get(section, "yaxis.ticks_position", fallback="None"))
	tick_params = utils.string_to_object(config.get(section, "axis.tick_params"    , fallback="None"))
	if xlim        is not None : axis.set_xlim(xlim)
	if x_minor_loc is not None : axis.xaxis.set_minor_locator(MultipleLocator(x_minor_loc))
	if x_major_loc is not None : axis.xaxis.set_major_locator(MultipleLocator(x_major_loc))
	if x_ticks_pos is not None : axis.xaxis.set_ticks_position(x_ticks_pos)
	if ylim        is not None : axis.set_ylim(ylim)
	if y_minor_loc is not None : axis.yaxis.set_minor_locator(MultipleLocator(y_minor_loc))
	if y_major_loc is not None : axis.yaxis.set_major_locator(MultipleLocator(y_major_loc))
	if y_ticks_pos is not None : axis.yaxis.set_ticks_position(y_ticks_pos)
	if tick_params is not None : axis.tick_params(tick_params)
	for label in utils.string_to_object( config.get(section, "Labels", fallback=[]) ) :
		plt.text( label[0], label[1], label[2], size="small" )


##  Main
#
def main () :
	#
	#  config and setup
	#
	parse_command_line_arguments()
	do_general_setup()
	print_setup()
	num_scan_params = len(glob.scan_params)
	#
	#  get SM expected limit
	#
	utils.info("get_limits.py", "Getting expected and observed confidence limits")
	exp_limit = glob.CL_generator.get_limit(glob.custom_store["SM_exp_dist"])
	obs_limit = glob.CL_generator.get_limit(glob.custom_store["meas_dist"  ])
	if num_scan_params == 1 :
		utils.info("get_limits.py", f"Observed {100.*glob.confidence_level:.2f}% confidence limit is {exp_limit}")
		utils.info("get_limits.py", f"Expected {100.*glob.confidence_level:.2f}% confidence limit is {obs_limit}")
	#
	#  generate SM toys and get limits
	#
	utils.info("get_limits()", f"Throwing toys around the experimental SM expectation and getting limits")
	SM_toy_limits, SM_coverage_grid = get_toy_spread_of_limits()
	n_toys = len(SM_toy_limits)
	if num_scan_params == 1 :
		utils.info("get_limits.py", f"Median {100.*glob.confidence_level:.2f}% limit of SM toys is {SM_toy_limits[int(0.5*n_toys)]:.0f}")
		utils.info("get_limits.py", f"16th percentile {100.*glob.confidence_level:.2f}% limit of SM toys is {SM_toy_limits[int(0.16*n_toys)]:.0f}")
		utils.info("get_limits.py", f"84th percentile {100.*glob.confidence_level:.2f}% limit of SM toys is {SM_toy_limits[int(0.84*n_toys)]:.0f}")
	#
	# plot
	#
	plotting.set_mpl_style()
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	if num_scan_params == 1 :
		limit_toys_5pc, limit_toys_16pc, limit_toys_median, limit_toys_84pc, limit_toys_95pc = SM_toy_limits[int(0.05*n_toys)], SM_toy_limits[int(0.16*n_toys)], SM_toy_limits[int(0.5*n_toys)], SM_toy_limits[int(0.84*n_toys)], SM_toy_limits[int(0.95*n_toys)]
		plt.axvspan(limit_toys_5pc , limit_toys_95pc           , color="darkorange", linestyle=None)
		plt.axvspan(limit_toys_16pc, limit_toys_84pc           , color="gold"      , linestyle=None)
		plt.plot([limit_toys_median, limit_toys_median], [0, 1], color="darkblue"  , linestyle="dashed", linewidth=1)
		plt.plot([exp_limit        , exp_limit ]       , [0, 1], color="green"     )
		plt.plot([obs_limit        , obs_limit]        , [0, 1], color="purple"    )
		ax.yaxis.set_visible(False)
		ax.set_ylim([0,1])
	else :
		plt.contourf(SM_coverage_grid.axes[0], SM_coverage_grid.axes[1], SM_coverage_grid.values.transpose(), [0.05, 0.16, 0.84, 0.95], linestyles=None, colors=["gold", "darkorange", "gold"])
		plt.contour (SM_coverage_grid.axes[0], SM_coverage_grid.axes[1], SM_coverage_grid.values.transpose(), [0.5], linestyles="dashed", colors=["darkblue"], linewidths=1)
		for limit in exp_limit : plt.plot([x[0] for x in limit], [y[1] for y in limit], color="green")
		for limit in obs_limit : plt.plot([x[0] for x in limit], [y[1] for y in limit], color="purple")
		plt.ylabel(f"{glob.scan_params[1].label}  [{glob.scan_params[1].units}]", horizontalalignment='right', y=1.0, fontsize="large")

	format_axis_from_config("GET_LIMITS") ;

	plt.xlabel(f"{glob.scan_params[0].label}  [{glob.scan_params[0].units}]", horizontalalignment='right', x=1.0, fontsize="large")

	plt.legend( [Line2D([0], [0], color="purple"    , lw=2), 
				 Line2D([0], [0], color="green"     , lw=2), 
				 Line2D([0], [0], color="darkblue"  , linestyle="dashed", lw=1), 
				 Patch (          color="gold"      , linestyle=None), 
				 Patch (          color="darkorange", linestyle=None)],
				[f"Obs. ({100*glob.confidence_level:.0f}% $CL_s$)",
				 f"Exp. ({100*glob.confidence_level:.0f}% $CL_s$)",
				 "SM toys: median",
				 "SM toys: 68% coverage",
				 "SM toys: 95% coverage"],
				loc=utils.string_to_object(glob.config.get("GET_LIMITS","legend.position",fallback="\'best\'")))

	if glob.custom_store["do show plots" ]             : plt.show()
	if glob.custom_store["plots filename"] is not None : plotting.save_figure(fig)
	plotting.close_plots_pdf()


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