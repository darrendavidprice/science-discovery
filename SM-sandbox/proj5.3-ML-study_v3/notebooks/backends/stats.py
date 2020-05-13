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
def get_special_encoding_constants_for_axis (dataset, axis, axmin, axmax, ax_npoints, frac_constant) :
    tmp_dataset = dataset[:,axis]
    ax_scan_points = np.linspace(axmin, axmax, 1+ax_npoints)
    tmp_dataset = np.array([x for x in tmp_dataset if (x>axmin and x<axmax)])
    
    data_cdf = []
    for A in ax_scan_points :
        data_cdf.append(len([x for x in tmp_dataset if x < A]) / len(tmp_dataset))
    data_cdf     = np.array(data_cdf)
    constant_cdf = (ax_scan_points - axmin) / (axmax - axmin)
    combined_cdf = frac_constant*constant_cdf + (1-frac_constant)*data_cdf
    
    Gauss_x   = np.linspace(-5, 5, 201)
    Gauss_cdf = stats.norm.cdf(Gauss_x)
    Gauss_cdf[0], Gauss_cdf[-1] = 0., 1.
    
    A_to_z = lambda A : np.interp(A, ax_scan_points, combined_cdf  )
    z_to_A = lambda z : np.interp(z, combined_cdf  , ax_scan_points)

    z_to_g = lambda z : np.interp(z, Gauss_cdf, Gauss_x  )
    g_to_z = lambda g : np.interp(g, Gauss_x  , Gauss_cdf)

    A_to_g = lambda A : z_to_g(A_to_z(A))
    g_to_A = lambda g : z_to_A(g_to_z(g))
    
    return A_to_g, g_to_A
    
    

#  Brief:  Take density function y(x) and return n_points randomly sampled points
#          - x and y are discrete arrays, and points are linearly interpolated between
#          - converges towards continuum function as x-spacing -> 0
#
def randomly_sample_function (x, y, n_points) :
    y_cdf = np.zeros(shape=y.shape)
    for idx in range(1, len(y)) :
        slice_integral = 0.5 *np.fabs( y[idx-1] + y[idx] )
        y_cdf [idx] = y_cdf [idx-1] + slice_integral
    y_cdf = y_cdf / y_cdf[-1]
    u = np.random.uniform(0, 1, n_points)
    return np.interp(u, y_cdf, x)


#  Brief:  Take density function z(*vx) and return n_points randomly sampled points
#          - vx are individual axes which will be used to make a meshgrid
#          - z is the grid of probabilities
#          - individual grid points are not interpolated between - just randomly sampled (discretely)
#          - converges towards continuum function as x-spacing -> 0
#
def randomly_sample_grid (n_points, z, *vx) :
    n_dim = len(z.shape)
    #
    #  Make sure input object shapes are compatible
    #
    bad_shape_error_message = f"Grid of shape {z.shape} but coordinates with shapes {[x.shape for x in vx]}"
    if n_dim != len(vx) :
        raise RuntimeError(bad_shape_error_message)
    for axis in range(n_dim) :
        if len(vx[axis].shape) > 1 :
            raise RuntimeError(bad_shape_error_message)
        if z.shape[axis] != vx[axis].shape[0] :
            raise RuntimeError(bad_shape_error_message)
    #
    #  Flatten coordinates and cumulative integral
    #
    grid = np.meshgrid(*reversed(vx))
    flat_grid = []
    for el in zip(*[a.flatten() for a in grid]) : flat_grid.append(list(el))
    flat_grid = np.array(flat_grid)
    flat_z    = z.flatten()
    flat_z    = flat_z/np.sum(flat_z)
    #
    #  Return random sampling of input points
    #
    return_bins = np.random.choice(len(flat_grid), n_points, p=flat_z)
    return flat_grid[return_bins]


#  Brief:  Take density function z(*vx) and return n_points randomly sampled points
#          - vx are meshgrids describing differetn coordinates
#          - z is the grid of probabilities
#          - individual grid points are not interpolated between - just randomly sampled (discretely)
#          - converges towards continuum function as x-spacing -> 0
#
def randomly_sample_meshgrid (n_points, z, *vx) :
    n_dim = len(z.shape)
    #
    #  Make sure input object shapes are compatible
    #
    bad_shape_error_message = f"Grid of shape {z.shape} but coordinates with shapes {[x.shape for x in vx]}"
    if n_dim != len(vx) :
        raise RuntimeError(bad_shape_error_message)
    for axis in range(n_dim) :
        if vx[axis].shape != z.shape :
            raise RuntimeError(bad_shape_error_message)
    #
    #  Flatten coordinates and cumulative integral
    #
    flat_grid = []
    for el in zip(*[a.flatten() for a in vx]) : flat_grid.append(list(el))
    flat_grid = np.array(flat_grid)
    flat_z    = z.flatten()
    flat_z    = flat_z/np.sum(flat_z)
    #
    #  Return random sampling of input points
    #
    return_bins = np.random.choice(len(flat_grid), n_points, p=flat_z)
    return flat_grid[return_bins]


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
            whitening_funcs.append(get_special_encoding_constants_for_axis (dataset, axis_idx, axis_config[0], axis_config[1], axis_config[2], axis_config[3]))
    white_dataset = np.array([[whitening_funcs[idx][0](x[idx]) for idx in range(num_axes)] for x in dataset])
    white_dataset, whitening_params = whiten_data(white_dataset, params=whitening_params, rotate=rotate)
    return white_dataset, whitening_funcs, whitening_params


#  Brief:  Unwhiten a dataset (using the "special" hard-boundary smoothing method)
#
def special_unwhiten_dataset (white_dataset, whitening_funcs, whitening_params) :
    num_axes = white_dataset.shape[1]
    unwhite_dataset = unwhiten_data(white_dataset, whitening_params)
    unwhite_dataset = np.array([[whitening_funcs[idx][1](x[idx]) for idx in range(num_axes)] for x in unwhite_dataset])
    return unwhite_dataset

 
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
