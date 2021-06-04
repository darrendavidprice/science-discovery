#======================================#
#   Brief:  Train a likelihood model   #
#   Author: stmenary@cern.ch           #
#======================================#


#=======================#
#  1. Required imports  #
#=======================#

print("Importing standard library")
import configparser, os, sys, time

print("Importing python data libraries")
import numpy as np
from   matplotlib import pyplot as plt, colors

print("Importing third party libraries")
import dill as pickle

path = os.getcwd().split("/")
path = "/".join(path[:path.index("proj5.4-EFT-Density-Estimation")+1])
print(f"Adding {path} to system paths")
sys.path.append(path)

print("Importing custom backends")
from backends.density_model    import DensityModel, get_sum_gauss_density
from backends.plot             import histo_to_line, plot_data, plot_ratio, plot_pull, get_ratio_1D
from backends.stats            import whiten_axes, unwhiten_axes
from backends.utils            import INFO, make_sure_dir_exists_for_filename, joint_shuffle

from backends import plot as plot, density_model as density_model, VBFZ_analysis as VBFZ


#==============================#
#  2. Configuration fallbacks  #
#==============================#

#  Input data

input_fname = "../../Data/SM_EWK_1M_rivet_output.pickle"

#  Model config

num_gaussians_per_continuous_observable = 25
max_epochs                              = 500
batch_size                              = 1000
early_stopping_patience                 = 20
early_stopping_min_delta                = 1e-9
validation_split                        = -1
gauss_width_factor                      = 1./4.
observables                             = []

learning_rate      = 1e-3
learning_rate_evo_factor   = 0.5
learning_rate_evo_patience = 4
optimiser          = "adam"
gauss_mean_scale   = 1./100.
gauss_frac_scale   = 1./100.
gauss_sigma_scale  = 1./100.
A1                 = 400
A2                 = 0
B1                 = 300
B2                 = 50
C_float            = 3
C_int              = [500, 200]
D2                 = 10

#  Projection config

white_linear_fraction_gauss = 0.
whitening_num_points        = 200
whitening_func_form         = "step"
whitening_alpha, whitening_beta, whitening_gamma = 4, 3, 1

load_whitening_funcs = None
save_whitening_funcs = None

load_model_dir = None
save_model_dir = ".EWK_density_model_paper_0D_<TAG>"

skip_initial_density_estimation = False

obs_white_linear_fraction_data_space = {}
obs_white_linear_fraction_data_space ["Dphi_j_j"] = 0.8
obs_white_linear_fraction_data_space ["Dy_j_j"  ] = 0.8
obs_white_linear_fraction_data_space ["m_jj"    ] = 0.2
obs_white_linear_fraction_data_space ["m_ll"    ] = 0.8
obs_white_linear_fraction_data_space ["pT_j1"   ] = 0.2
obs_white_linear_fraction_data_space ["pT_j2"   ] = 0.2
obs_white_linear_fraction_data_space ["pT_jj"   ] = 0.2
obs_white_linear_fraction_data_space ["pT_ll"   ] = 0.2
obs_white_linear_fraction_data_space ["rap_jj"  ] = 0.2
obs_white_linear_fraction_data_space ["rap_ll"  ] = 0.8

white_linear_fraction_data = None

#  Choose observables
#
remove_observables = ["pT_jj", "N_jets", "N_gap_jets", "m_ll", "Dy_j_j"]
observables_order  = []
reset_observables  = []



def load_settings (config_fname="") :
	if len(config_fname) == 0 :
		sys_args = sys.argv
		assert len(sys_args) == 2, f"Expected command line 'python3 program args_file' but '{' '.join(sys_args)}' provided"
		config_fname = sys.argv[1]
	cfg = configparser.ConfigParser()
	cfg.read(config_fname)
	assert "SETTINGS" in cfg.sections(), f"[SETTINGS] header not found in config file {config_fname}"
	settings = cfg["SETTINGS"]
	for key, val in settings.items() :
		INFO("load_settings", f"Found configuration {key} = {val}")
	global input_fname, num_gaussians_per_continuous_observable, max_epochs, batch_size, early_stopping_patience, early_stopping_min_delta, validation_split, learning_rate
	global optimiser, gauss_mean_scale, gauss_frac_scale, gauss_sigma_scale, A1, A2, B1, B2, C_float, C_int, D2, white_linear_fraction_gauss, whitening_num_points, reset_observables
	global whitening_func_form, whitening_alpha, whitening_beta, whitening_gamma, load_whitening_funcs, save_whitening_funcs, load_model_dir, save_model_dir, observables, observables_order
	global obs_white_linear_fraction_data_space, remove_observables, gauss_width_factor, learning_rate_evo_factor, learning_rate_evo_factor, learning_rate_evo_patience, skip_initial_density_estimation
	run_tag                         = str(settings.get("run_tag", "untagged"))
	input_fname                     = str(settings.get("input_fname", input_fname)).replace("<TAG>", run_tag)
	num_gaussians_per_continuous_observable = int(settings.get("num_gaussians_per_continuous_observable", num_gaussians_per_continuous_observable))
	max_epochs                      = int(settings.get("max_epochs", max_epochs))
	batch_size                      = int(settings.get("batch_size", batch_size))
	early_stopping_patience         = int(settings.get("early_stopping_patience", early_stopping_patience))
	early_stopping_min_delta        = float(settings.get("early_stopping_min_delta", early_stopping_min_delta))
	validation_split                = float(settings.get("validation_split", validation_split))
	learning_rate                   = float(settings.get("learning_rate", learning_rate))
	learning_rate_evo_factor        = float(settings.get("learning_rate_evo_factor", learning_rate_evo_factor))
	learning_rate_evo_patience      = float(settings.get("learning_rate_evo_patience", learning_rate_evo_patience))
	optimiser                       = str(settings.get("optimiser", optimiser))
	if "observables"       in settings : observables       = [int(s) for s in settings["observables"      ].split(" ") if len(s) > 0]
	if "observables_order" in settings : observables_order = [str(s) for s in settings["observables_order"].split(" ") if len(s) > 0]
	if "reset_observables" in settings : reset_observables = [int(s) for s in settings["reset_observables"].split(" ") if len(s) > 0]
	gauss_mean_scale                = float(settings.get("gauss_mean_scale", gauss_mean_scale))
	gauss_frac_scale                = float(settings.get("gauss_frac_scale", gauss_frac_scale))
	gauss_sigma_scale               = float(settings.get("gauss_sigma_scale", gauss_sigma_scale))
	gauss_width_factor              = float(settings.get("gauss_width_factor", gauss_width_factor))
	A1                              = int(settings.get("A1", A1))
	A2                              = int(settings.get("A2", A2))
	B1                              = int(settings.get("B1", B1))
	B2                              = int(settings.get("B2", B2))
	C_float                         = int(settings.get("C_float", C_float))
	if "C_int" in settings : C_int  = [int(s) for s in settings["C_int"].split(" ") if len(s) > 0]
	D2                              = int(settings.get("D2", D2))
	white_linear_fraction_gauss     = float(settings.get("white_linear_fraction_gauss", white_linear_fraction_gauss))
	whitening_num_points            = int(settings.get("whitening_num_points", whitening_num_points))
	whitening_func_form             = str(settings.get("whitening_func_form", whitening_func_form))
	whitening_alpha                 = float(settings.get("whitening_alpha", whitening_alpha))
	whitening_beta                  = float(settings.get("whitening_beta", whitening_beta))
	whitening_gamma                 = float(settings.get("whitening_gamma", whitening_gamma))
	load_whitening_funcs            = settings.get("load_whitening_funcs", load_whitening_funcs)
	save_whitening_funcs            = settings.get("save_whitening_funcs", save_whitening_funcs)
	load_model_dir                  = settings.get("load_model_dir", load_model_dir)
	save_model_dir                  = settings.get("save_model_dir", save_model_dir)
	skip_initial_density_estimation = bool(settings.get("skip_initial_density_estimation", skip_initial_density_estimation))
	if type(load_whitening_funcs) is str : load_whitening_funcs = load_whitening_funcs.replace("<TAG>", run_tag)
	if type(save_whitening_funcs) is str : save_whitening_funcs = save_whitening_funcs.replace("<TAG>", run_tag)
	if type(load_model_dir) is str : load_model_dir = load_model_dir.replace("<TAG>", run_tag)
	if type(save_model_dir) is str : save_model_dir = save_model_dir.replace("<TAG>", run_tag)
	for obs in VBFZ.all_observables :
		key = f"obs_white_linear_fraction_data_space_{obs}"
		if key in settings : obs_white_linear_fraction_data_space[obs] = settings[key]
	if "remove_observables" in settings :
		remove_observables = [str(s) for s in settings["remove_observables"].split(" ") if len(s) > 0]


def print_settings () :
	INFO("print_settings", f"Using input_fname = {input_fname}")
	INFO("print_settings", f"Using num_gaussians_per_continuous_observable = {num_gaussians_per_continuous_observable}")
	INFO("print_settings", f"Using max_epochs = {max_epochs}")
	INFO("print_settings", f"Using batch_size = {batch_size}")
	INFO("print_settings", f"Using early_stopping_patience = {early_stopping_patience}")
	INFO("print_settings", f"Using early_stopping_min_delta = {early_stopping_min_delta}")
	INFO("print_settings", f"Using validation_split = {validation_split}")
	INFO("print_settings", f"Using learning_rate = {learning_rate}")
	INFO("print_settings", f"Using learning_rate_evo_factor = {learning_rate_evo_factor}")
	INFO("print_settings", f"Using learning_rate_evo_patience = {learning_rate_evo_patience}")
	INFO("print_settings", f"Using optimiser = {optimiser}")
	INFO("print_settings", f"Training observables = {', '.join([str(s) for s in observables])}")
	INFO("print_settings", f"Using gauss_mean_scale = {gauss_mean_scale:.3f}")
	INFO("print_settings", f"Using gauss_frac_scale = {gauss_frac_scale:.3f}")
	INFO("print_settings", f"Using gauss_sigma_scale = {gauss_sigma_scale:.3f}")
	INFO("print_settings", f"Using gauss_width_factor = {gauss_width_factor:.3f}")
	INFO("print_settings", f"Using A1 = {A1}")
	INFO("print_settings", f"Using A2 = {A2}")
	INFO("print_settings", f"Using B1 = {B1}")
	INFO("print_settings", f"Using B2 = {B2}")
	INFO("print_settings", f"Using C_float = {C_float}")
	INFO("print_settings", f"Using C_int = {C_int}")
	INFO("print_settings", f"Using D2 = {D2}")
	INFO("print_settings", f"Using white_linear_fraction_gauss = {white_linear_fraction_gauss}")
	INFO("print_settings", f"Using whitening_num_points = {whitening_num_points}")
	INFO("print_settings", f"Using whitening_func_form = {whitening_func_form}")
	INFO("print_settings", f"Using whitening_alpha = {whitening_alpha}")
	INFO("print_settings", f"Using whitening_beta = {whitening_beta}")
	INFO("print_settings", f"Using whitening_gamma = {whitening_gamma}")
	INFO("print_settings", f"Using load_whitening_funcs = {load_whitening_funcs}")
	INFO("print_settings", f"Using save_whitening_funcs = {save_whitening_funcs}")
	INFO("print_settings", f"Using load_model_dir = {load_model_dir}")
	INFO("print_settings", f"Using save_model_dir = {save_model_dir}")
	INFO("print_settings", f"Using remove_observables = {remove_observables}")
	INFO("print_settings", f"Using observables order = {observables_order}")
	INFO("print_settings", f"Using reset_observables = {reset_observables}")
	INFO("print_settings", f"Using skip_initial_density_estimation = {skip_initial_density_estimation}")
	for obs, frac in obs_white_linear_fraction_data_space.items() :
		INFO("print_settings", f"Using obs_white_linear_fraction_data_space[{obs}] = {frac:.3f}")


def VBFZ_setup () :
	global white_linear_fraction_data, observables, observables_order
	VBFZ.configure(remove_observables, order=observables_order)
	INFO("VBFZ_setup", f"Configured with {VBFZ.num_observables} observables: " + ", ".join(VBFZ.observables))
	white_linear_fraction_data = [obs_white_linear_fraction_data_space[obs] if obs in obs_white_linear_fraction_data_space else 0. for obs in VBFZ.observables]
	plot.int_observables   = VBFZ.int_observables
	plot.observable_limits = VBFZ.transformed_observable_limits
	if len(observables) == 0 : observables = [i for i in range(VBFZ.num_observables)]


def get_original_and_projected_data_as_dict (data_table) :
	#
	#  Load whitening funcs if a file was provided -  this is faster when re-running with the same data and whitening settings later on
	#
	whitening_funcs = None
	if type(load_whitening_funcs) != type(None) :
	    INFO("get_original_and_projected_data_as_dict", f"Loading whitening functions from file {load_whitening_funcs}")
	    whitening_funcs = pickle.load(open(load_whitening_funcs, "rb"))
	#
	#  Collect information on axis limits as well as whitening functions, in case new whitening functions need to be generated
	#
	axis_configs = [[VBFZ.transformed_observable_limits[key][0], VBFZ.transformed_observable_limits[key][1], whitening_num_points, white_linear_fraction_data[idx], white_linear_fraction_gauss] for idx, key in enumerate(data_table.keys)]
	#
	#  Seperate data from weights
	#
	true_data, true_data_weights = {}, {}
	true_data [1.], true_data_weights [1.] = data_table.get_observables_and_weights()
	#
	#  Transform data
	#
	INFO("get_original_and_projected_data_as_dict", "Transforming data")
	transformed_data = {}
	transformed_data [1.] = VBFZ.transform_observables_fwd(true_data[1.], data_table.keys)
	#
	#  Whiten data
	#
	INFO("get_original_and_projected_data_as_dict", "Projecting data onto latent space")
	white_data = {}
	white_data[1.], whitening_funcs = whiten_axes(transformed_data[1.], data_table.types, axis_configs=axis_configs, whitening_funcs=whitening_funcs, weights=true_data_weights[1.], 
	                                              func_form=whitening_func_form, alpha=whitening_alpha, beta=whitening_beta, gamma=whitening_gamma)
	#
	#  Save whitening funcs to file, if requested
	#
	if type(save_whitening_funcs) != type(None) :
	    INFO("get_original_and_projected_data_as_dict", f"Saving whitening functions to file {save_whitening_funcs}")
	    pickle.dump(whitening_funcs, open(save_whitening_funcs, "wb"))
	#
	#  Return whitening_funcs and dict objects
	#
	return whitening_funcs, true_data, true_data_weights, transformed_data, white_data



def load_build_fit_model (white_data, true_data_weights, observables, save_model_dir=None) :
	#
	#  Load model if requested, otherwise build and fit
	#
	if type(load_model_dir) != type(None) :
	    density_model = DensityModel.from_dir(load_model_dir)
	else :
		#
		#   Figure out the limits of the observables
		#
		white_observables_limits = []
		for obs_idx, (obs_name, obs_type) in enumerate(zip(VBFZ.observables, VBFZ.observable_types)) :
		    if obs_type is int :
		        white_observables_limits.append([float(x) for x in VBFZ.transformed_observable_limits[obs_name]])
		        continue        
		    all_data = np.concatenate([item[:,obs_idx] for c,item in white_data.items()])
		    min_dp, max_dp = np.min(all_data), np.max(all_data)
		    range_dp_per_gauss = (max_dp - min_dp) / num_gaussians_per_continuous_observable
		    white_observables_limits.append([min_dp + 0.5*range_dp_per_gauss, max_dp - 0.5*range_dp_per_gauss])
		#
		#   Build model
		#
		density_model = DensityModel(name               = "EWK_Zjj_density_model_0D"              , 
		                             num_gaussians      = num_gaussians_per_continuous_observable , 
		                             num_conditions     = 1                                       , 
		                             num_observables    = VBFZ.num_observables                    , 
		                             types              = VBFZ.observable_types                   ,
		                             observables_limits = white_observables_limits                ,
		                             verbose            = True                                    , 
		                             gauss_mean_scale   = gauss_mean_scale                        ,
		                             gauss_frac_scale   = gauss_frac_scale                        ,
		                             gauss_sigma_scale  = gauss_sigma_scale                       ,
		                             gauss_width_factor = gauss_width_factor                      ,
		                             optimiser          = optimiser                               ,
		                             learning_rate      = learning_rate                           ,
		                             learning_rate_evo_factor = 1                                 ,  #  instead evolve learning rate during training using callback
		                             A1                 = A1                                      ,
		                             A2                 = A2                                      ,
		                             B1                 = B1                                      ,
		                             B2                 = B2                                      ,
		                             C_float            = C_float                                 ,
		                             C_int              = C_int                                   ,
		                             D2                 = D2                                      )
	#
	#
	#
	for index in reset_observables :
		assert index < len(density_model.likelihood_models), f"Observable index {index} out of bounds"
		INFO("load_build_fit_model", f"Resetting observable at index {index}")
		density_model.likelihood_models[index].reset_weights()
	#
	#   Make sure initial state has no NaN/Inf loss
	#
	if skip_initial_density_estimation :
		INFO("load_build_fit_model", "Skipping initial density estimation (caution: we have not ensured that initial likelihoods evaluate to something real)")
	else :
		density_model.ensure_valid_over_dataset (white_data, true_data_weights)
	#
	#   Fit density model
	#
	density_model.fit(white_data                                             , 
	                  true_data_weights                                      ,
	                  observable                 = observables               ,
	                  max_epochs_per_observable  = max_epochs                ,
	                  early_stopping_patience    = early_stopping_patience   ,
	                  early_stopping_min_delta   = early_stopping_min_delta  ,
	                  validation_split           = validation_split          ,
	                  batch_size_per_observable  = batch_size                ,
	                  learning_rate_evo_factor   = learning_rate_evo_factor  ,
	                  learning_rate_evo_patience = learning_rate_evo_patience,
	                  save_to_dir                = save_model_dir            )
	#
	#   Return density model
	#
	return density_model



#====================================#
#  N. Fallback to running as script  #
#====================================#

if __name__ == "__main__" :
	load_settings()
	print_settings()
	VBFZ_setup()
	data_table = VBFZ.load_table(input_fname)
	whitening_funcs, true_data, true_data_weights, transformed_data, white_data = get_original_and_projected_data_as_dict (data_table)
	true_data[1.], transformed_data[1.], white_data[1.], true_data_weights[1.] = joint_shuffle(true_data[1.], transformed_data[1.], white_data[1.], true_data_weights[1.])
	density_model = load_build_fit_model (white_data, true_data_weights, observables, save_model_dir)
	if type(save_model_dir) != type(None) :
	    density_model.save_to_dir(save_model_dir)
