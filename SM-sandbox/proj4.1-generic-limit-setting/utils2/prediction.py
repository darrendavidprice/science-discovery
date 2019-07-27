##  =======================================================================================================================
##  Brief: interface to provided BSM predictions from a set of inputs
##  Author: Stephen Menary
##  Email: sbmenary@gmail.com
##  =======================================================================================================================

import numpy as np

import utils2.utils.utils                 as     utils
import utils2.utils.globals_and_fallbacks as     glob
from   utils2.misc.enums                  import BSMPredictionMethod
from   utils2.objects.Distribution        import Distribution
from   utils2.objects.Grid                import Grid


##  Get BSM predictions using ScaleByL6 method
#
def populate_scan_grid_using_ScaleByL6 (BSM_input_dists, new_grid, scan_params, SM=None) :
	try :
		lambda_grid_index = new_grid.keys.index("Lambda")
		lambda_list_index = [p.name for p in scan_params].index("Lambda")
		utils.info("populate_input_grid_using_ScaleByL6()", "successfully found \'Lambda\' in param list")
	except ValueError as e :
		raise KeyError("populate_input_grid_using_ScaleByL6(): no parameter \'Lambda\' found in param list")
	n_dim = len(new_grid.keys)
	if n_dim == 1 :
		if len(BSM_input_dists) > 1 :
			raise RuntimeError("populate_input_grid_using_ScaleByL6(): don't know which input to scale Lambda from...")
		L_ref, dist_ref = 0., None
		for L, i in BSM_input_dists.items() :
			L_ref    = float(L[0])
			dist_ref = i
		if dist_ref.includes_SM :
			if SM is None : raise ValueError("populate_scan_grid_using_ScaleByL6(): need to subtract SM from BSM input but none provided")
			dist_ref.subtract_values(SM.values)
			dist_ref.subtract_cov   (SM.cov)
		L_values = new_grid.axes[0]
		for idx in range(len(L_values)) :
			sf = (L_ref/L_values[idx]) ** 6
			new_grid.values[idx] = dist_ref * sf
	elif n_dim == 2 :
		other_param_grid_index = 1 - lambda_grid_index
		other_param_list_index = 1 - lambda_list_index
		other_param_key        = new_grid.keys[other_param_grid_index]
		for other_param_axis_idx in range(len(new_grid.axes[other_param_grid_index])) :
			other_param_value = new_grid.axes[other_param_grid_index][other_param_axis_idx]
			lambda_ref, dist_ref = None, None
			for key, item in BSM_input_dists.items() :
				if type(other_param_value)(key[other_param_list_index]) != other_param_value : continue
				if dist_ref is not None :
					raise RuntimeError("populate_input_grid_using_ScaleByL6(): don't know which input to scale Lambda from...")
				lambda_ref = np.float64(key[lambda_list_index])
				dist_ref   = item
			if dist_ref.includes_SM :
				if SM is None : raise ValueError("populate_scan_grid_using_ScaleByL6(): need to subtract SM from BSM input but none provided")
				dist_ref.subtract_values(SM.values)
				dist_ref.subtract_cov   (SM.cov)
			for grid_idx_lambda in range(len(new_grid.axes[lambda_grid_index])) :
				lambda_value = np.float64(new_grid.axes[lambda_grid_index][grid_idx_lambda])
				sf = (lambda_ref/lambda_value) ** 6
				if lambda_grid_index == 0 : idx_x, idx_y = grid_idx_lambda, other_param_axis_idx
				else : idx_x, idx_y = other_param_axis_idx, grid_idx_lambda
				new_grid.values[idx_x][idx_y] = dist_ref * sf
	elif n_dim > 2 :
		raise NotImplementedError(f"populate_input_grid_using_ScaleByL6(): only 1D and 2D scans implemented, {n_dim}D asked")
	return new_grid


##  Get BSM distributions at scan points
#
def generate_BSM_predictions (BSM_input_dists, method=None, param_grid=None, scan_params=None, SM=None) :
	if method == None :
		method = glob.BSM_prediction_method
		if method is None : raise ValueError("prediction.generate_BSM_predictions(): no method provided and no fallback BSM_prediction_method set")
	if param_grid is not None and scan_params is not None :
		raise RuntimeWarning("prediction.generate_BSM_predictions(): both param_grid and scan_params provided - the latter will be ignored")
	if param_grid is None and scan_params is None :
		scan_params = glob.scan_params
		if scan_params is None : raise ValueError("prediction.generate_BSM_predictions(): no param_grid or scan_params provided and no fallback scan_params set")
	if param_grid is None :
		param_grid = Grid.create_param_grid(scan_params)
	new_grid = Grid(param_grid)
	new_grid.generate_grid(element_type=Distribution)
	if method == BSMPredictionMethod.ScaleByL6 :
		populate_scan_grid_using_ScaleByL6(BSM_input_dists, new_grid, scan_params, SM=SM)
	else :
		raise NotImplementedError("generate_BSM_predictions(): only ScaleByL6 method currently implemented")
	return new_grid

	