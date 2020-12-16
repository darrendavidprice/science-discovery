#   Statistical methods to help with generative modelling, in particular
#   Author:  Stephen Menary  (stmenary@cern.ch)

import numpy as np
from   scipy import stats

from .utils import get_flat_copy_of_data


#  Brief:  Take datapoints and bin them accordingly
#          - bins and data must be one-dimensional arrays
#
def bin_data (data, bins=None, weights=None) :
    if type(bins) == type(None) :
        data_min, data_max = np.min(data), np.max(data)
        data_range = data_max - data_min
        bins = np.linspace(data_min-0.05*data_range, data_max+0.05*data_range, 11)
    if type(bins) != np.ndarray :
        bins = np.array(bins)
    if len(bins.shape) != 1 : raise RuntimeError("Provided bins must be one-dimensional array")
    if bins.shape[0] < 1    : raise RuntimeError("Provided bins must be one-dimensional array")
    if type(data) != np.ndarray :
        data = np.array(data)
    if len(data.shape) != 1 : raise RuntimeError("Provided data must be one-dimensional array")
    if type(weights) == type(None) :
        weights = np.full(shape=data.shape, fill_value=1.)
    bin_contents = np.zeros(shape=(bins.shape[0]-1,))
    for idx in range(len(bin_contents)) :
        bin_edge_low  = bins[idx  ]
        bin_edge_high = bins[idx+1]
        content = 0
        for datum, weight in zip(data, weights) :
            if datum < bin_edge_low  : continue
            if datum > bin_edge_high : continue
            content = content + weight
        bin_contents[idx] = content
    return bins, bin_contents


#  Brief:  Return a covariance matrix describing the spread of data
#
def get_cov_from_dataset (x) :
    if len(x.shape) != 2 :
        raise RuntimeError(f"can only derive covariance from a 2D object but shape {x.shape} provided")
    return np.cov(x.transpose())


#  Brief:  Return the mean and std dev along every axis of the dataset
#
def get_data_mean_std_from_dataset (x) :
    if len(x.shape) != 2 :
        raise RuntimeError(f"Dataset must be a 2D object but shape {x.shape} provided")
    x = get_flat_copy_of_data(x)
    means, stds = [], []
    for i in range(x.shape[1]) :
        means.append(np.mean(x [:,i]))
        stds.append (np.std (x [:,i]))
    return means, stds


#  Brief:  Get the means / std devs /eigenvectors required to whiten the dataset
#
def get_data_whitening_constants (x) :
    x_norm = get_flat_copy_of_data(x)
    means, stds = get_data_mean_std_from_dataset(x_norm)
    for i in range(x_norm.shape[1]) :
        x_norm[:,i] = (x_norm[:,i] - means[i]) / stds[i]
    data_cov = get_cov_from_dataset(x_norm)
    eigenvalues, eigenvectors = np.linalg.eig(data_cov)
    for i, v in enumerate(eigenvectors) :
        eigenvectors[i] = v / np.sqrt(eigenvalues[i])
    return (means, stds, eigenvectors)

#  Brief:  Create lambda functions used to map a distribution with hard boundaries onto a smoothish piecewise Gaussian
#
def get_special_encoding_constants_for_axis (dataset, axis, axmin, axmax, ax_npoints, data_frac_constant, gauss_frac_constant=0., weights=None, **kwargs) :
    if type(axis) == type(None) : tmp_dataset = dataset
    else : tmp_dataset = dataset[:,axis]

    if type(weights) == type(None) : weights = np.ones(shape=(len(dataset),))
    weights = weights / np.sum(weights)

    ax_scan_points = np.linspace(axmin, axmax, 1+ax_npoints)
    tmp_dataset = np.array([x for x in tmp_dataset if (x>axmin and x<axmax)])
    
    data_cdf = []
    for A in ax_scan_points :
        data_cdf.append(np.sum([w for x,w in zip(tmp_dataset,weights) if x < A]))
    data_cdf     = np.array(data_cdf)
    constant_cdf = (ax_scan_points - axmin) / (axmax - axmin)
    combined_cdf = data_frac_constant*constant_cdf + (1-data_frac_constant)*data_cdf

    whitened_func_form = kwargs.get("func_form", "gaus")
    if whitened_func_form == "step" :
        alpha, beta, gamma = kwargs.get("alpha", 3), kwargs.get("beta", 2), kwargs.get("gamma", 0)
        white_space_x   = np.linspace(-6, 6, 241)
        Smooth_step_y   = 1. / (1 + np.exp((white_space_x-beta)*alpha-gamma)) / (1 + np.exp(-(white_space_x+beta)*alpha-gamma))
        Smooth_step_cdf = np.array([np.sum(Smooth_step_y[:i+1]) for i in range(len(Smooth_step_y))])
        Smooth_step_cdf = Smooth_step_cdf / Smooth_step_cdf[-1]
        Smooth_step_cdf[0] = 0.
        constant_cdf    = (white_space_x + 5.) / 10.
        white_space_cdf = gauss_frac_constant*constant_cdf + (1-gauss_frac_constant)*Smooth_step_cdf
    else :
        white_space_x   = np.linspace(-6, 6, 241)
        Gauss_cdf       = stats.norm.cdf(white_space_x)
        Gauss_cdf[0], Gauss_cdf[-1] = 0., 1.
        constant_cdf    = (white_space_x + 5.) / 10.
        white_space_cdf = gauss_frac_constant*constant_cdf + (1-gauss_frac_constant)*Gauss_cdf 
        
    A_to_z = lambda A : np.interp(A, ax_scan_points, combined_cdf  )
    z_to_A = lambda z : np.interp(z, combined_cdf  , ax_scan_points)

    z_to_g = lambda z : np.interp(z, white_space_cdf, white_space_x  )
    g_to_z = lambda g : np.interp(g, white_space_x  , white_space_cdf)

    A_to_g = lambda A : z_to_g(A_to_z(A))
    g_to_A = lambda g : z_to_A(g_to_z(g))
    
    return A_to_g, g_to_A


#  Brief:  Whiten a dataset (using the "special" hard-boundary smoothing method)
#
def special_whiten_axis (dataset, axis_config=None, whitening_func=None, weights=None, **kwargs) :
    if type(whitening_func) == type(None) :
        if type(axis_config) is type(None) :
            raise TypeError("Argument axis_config must be provided when whitening_func is None")
        if len(axis_config) == 4 :
            whitening_func = get_special_encoding_constants_for_axis (dataset, None, axis_config[0], axis_config[1], axis_config[2], axis_config[3], 0., weights=weights, **kwargs)
        else :
            whitening_func = get_special_encoding_constants_for_axis (dataset, None, axis_config[0], axis_config[1], axis_config[2], axis_config[3], axis_config[4], weights=weights, **kwargs)
    white_dataset = np.array([whitening_func[0](x) for x in dataset])
    return white_dataset, whitening_func


#  Brief:  Whiten a dataset (using the "special" hard-boundary smoothing method)
#
def special_whiten_dataset (dataset, *axis_configs, whitening_funcs=None, whitening_params=None, rotate=True) :
    num_axes = dataset.shape[1]
    if type(whitening_funcs) == type(None) :
        if num_axes != len(axis_configs) : 
            raise ValueError(f"Dataset with shape {dataset.shape} requires {num_axes} axis configs but {len(axis_configs)} provided")
        whitening_funcs = []
        for axis_idx in range(num_axes) :
            axis_config = axis_configs[axis_idx]
            if len(axis_config) == 4 :
                whitening_funcs.append(get_special_encoding_constants_for_axis (dataset, axis_idx, axis_config[0], axis_config[1], axis_config[2], axis_config[3], 0.))
            else :
                whitening_funcs.append(get_special_encoding_constants_for_axis (dataset, axis_idx, axis_config[0], axis_config[1], axis_config[2], axis_config[3], axis_config[4]))
    white_dataset = np.array([[whitening_funcs[idx][0](x[idx]) for idx in range(num_axes)] for x in dataset])
    white_dataset, whitening_params = whiten_data(white_dataset, params=whitening_params, rotate=rotate)
    return white_dataset, whitening_funcs, whitening_params


#  Brief:  Unwhiten a dataset (using the "special" hard-boundary smoothing method)
#
def special_unwhiten_axis (white_dataset, whitening_func) :
    unwhite_dataset = np.array([whitening_func[1](x) for x in white_dataset])
    return unwhite_dataset


#  Brief:  Unwhiten a dataset (using the "special" hard-boundary smoothing method)
#
def special_unwhiten_dataset (white_dataset, whitening_funcs, whitening_params) :
    num_axes = white_dataset.shape[1]
    unwhite_dataset = unwhiten_data(white_dataset, whitening_params)
    unwhite_dataset = np.array([[whitening_funcs[idx][1](x[idx]) for idx in range(num_axes)] for x in unwhite_dataset])
    return unwhite_dataset


#  Return whitened dataset
#
def whiten_axes (data, types, axis_configs=None, whitening_funcs=None, weights=None, **kwargs) :
    white_data, new_whitening_funcs = [], []
    num_observables = len(types)
    if type(axis_configs) == type(None) :
        axis_configs = [None for i in range(num_observables)]
    for obs_idx in range(num_observables) :
        if types[obs_idx] == int :
            white_axis, new_whitening_func = data[:,obs_idx], None
        else :
            if type(whitening_funcs) == type(None) :
                white_axis, new_whitening_func = special_whiten_axis(data[:,obs_idx], axis_configs[obs_idx], weights=weights, **kwargs)
            else :
                white_axis, new_whitening_func = special_whiten_axis(data[:,obs_idx], axis_configs[obs_idx], whitening_func=whitening_funcs[obs_idx], weights=weights, **kwargs)
        new_whitening_funcs.append(new_whitening_func)
        white_data         .append(white_axis)
    white_data = np.array(white_data).transpose()
    return white_data, new_whitening_funcs


#  Brief:  Whiten a dataset (generate whitening params if none provided)
#
def whiten_data (x, params=None, rotate=True) :
    if type(params) == type(None) :
        params = get_data_whitening_constants(x)
    means, stds, eigenvectors = params
    if type(x) is dict :
        ret = {}
        for c, xc in x.items() :
            ret[c], _ = whiten_data (xc, params=params)
        return ret, params
    if type(x) is list :
        x = np.array(x)
    if type(x) != np.ndarray :
        raise TypeError(f"Can only whiten data of type {type(np.ndarray)} where {type(x)} provided")
    new_data = np.copy(x)
    for i in range(x.shape[1]) :
        new_data[:,i] = (new_data[:,i] - means[i]) / stds[i]
    if rotate : 
        new_data = np.array([np.matmul(eigenvectors, xp) for xp in new_data])
    else      : 
        eigenvectors = None
    return new_data, (means, stds, eigenvectors)


#  Return unwhitened dataset
#
def unwhiten_axes (white_data, whitening_funcs) :
    unwhite_data = []
    for obs_idx, whitening_func in enumerate(whitening_funcs) :
        if type(whitening_func) == type(None) :
            unwhite_axis = white_data[:,obs_idx]
        else :
            unwhite_axis = special_unwhiten_axis(white_data[:,obs_idx], whitening_func)
        unwhite_data.append(unwhite_axis)
    return np.array(unwhite_data).transpose()

 
#  Brief:  Unwhiten a dataset (using the params provided)
#
def unwhiten_data (x, params) :
    means, stds, eigenvectors = params
    if type(x) is dict :
        ret = {}
        for c, xc in x.items() :
            ret[c] = unwhiten_data (xc, params)
        return ret
    if type(x) is list :
        x = np.array(x)
    if type(x) != np.ndarray :
        raise TypeError(f"Can only unwhiten data of type {type(np.ndarray)} where {type(x)} provided")
    new_data = x.copy()
    if type(eigenvectors) != type(None) :
        inv_eigenvectors = np.linalg.inv(eigenvectors)
        new_data = np.array([np.matmul(inv_eigenvectors, xp) for xp in new_data])
    for i in range(new_data.shape[1]) :
        new_data [:,i] = np.array(stds[i]*new_data [:,i] + means[i])
    return new_data
