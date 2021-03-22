import math

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    
plot_cbar_blues = cm.get_cmap('Blues_r', 128)
plot_cbar_reds  = cm.get_cmap('Reds', 128)
newcolors       = np.vstack((plot_cbar_blues(np.linspace(0, 1, 128)), plot_cbar_reds(np.linspace(0, 1, 128))))
newcolors [math.ceil(258*3/8)-1:math.floor(258*5/8)-1] = np.array([68/256, 223/256, 68/256, 1])
custom_colormap = ListedColormap(newcolors, name='BlueToRed')

observable_limits = {}
int_observables   = []


#  Remove events for which key lies outside the interval [minimum, maximum]
#
def filter_data (events, weights, keys, key, minimum, maximum) :
    col_idx = keys.index(key)
    new_events, new_weights = [], []
    for row, weight in zip(events, weights) :
        val = row[col_idx]
        if val < minimum : continue
        if val > maximum : continue
        new_events.append(row)
        new_weights.append(weight)
    return np.array(new_events), np.array(new_weights)


#  Convert histogram values to a plottable outline
#
def histo_to_line (bins, values, errors=None) :
    X, Z, EZ = [], [], []
    for i in range(len(bins)-1) :
        X .append(bins[i])
        X .append(bins[i+1])
    for zp in values :
        Z .append(zp)
        Z .append(zp)
    if type(errors) is type(None) :
    	return np.array(X), np.array(Z)
    for ezp in errors :
        EZ.append(ezp)
        EZ.append(ezp)
    return np.array(X), np.array(Z), np.array(EZ)


#  Plot the datapoints provided
#
def plot_data (observables, weights=None, keys=None, cuts=[], save="", lims=True, bins=20) :
    if type(weights) == type(None) :
        weights = np.ones(shape=(observables.shape[0],))
    if type(keys) == type(None) :
        keys = [f"obs{i}" for i in range(len(observables))]
    filtered_observables, filtered_weights = observables, weights
    for cut in cuts :
        print(f"Filtering {cut[0]} between {cut[1]} and {cut[2]}")
        filtered_observables, filtered_weights = filter_data (filtered_observables, filtered_weights, keys, cut[0], cut[1], cut[2])
    print(f"Filter efficiency is {100.*np.sum(filtered_weights)/np.sum(weights):.3f}%")
    num_observables = filtered_observables.shape[1]
    fig = plt.figure(figsize=(2*num_observables, 2*num_observables))
    for row_idx, row_obs in enumerate(keys) :
        row_lims = observable_limits[row_obs]
        if row_obs in int_observables : row_bins = np.linspace(row_lims[0]-0.5, row_lims[1]+0.5, 2+(row_lims[1]-row_lims[0]))
        else                          : row_bins = np.linspace(row_lims[0], row_lims[1], bins+1)
        if not lims : row_bins = np.linspace(-6, 6, bins+1)
        for col_idx, col_obs in enumerate(keys) :
            col_lims = observable_limits[col_obs]
            if col_obs in int_observables : col_bins = np.linspace(col_lims[0]-0.5, col_lims[1]+0.5, 2+(col_lims[1]-col_lims[0]))
            else                          : col_bins = np.linspace(col_lims[0], col_lims[1], bins+1)
            if not lims : col_bins = np.linspace(-6, 6, bins+1)
            ax = fig.add_subplot(num_observables, num_observables, row_idx*num_observables + col_idx + 1)
            if row_idx == 0 :
                ax.set_title(col_obs, weight="bold", fontsize=12)
            if col_idx == 0 :
                ax.set_ylabel(row_obs, weight="bold", fontsize=12)
            if row_idx == col_idx :
                ax.hist(filtered_observables[:,row_idx], weights=filtered_weights, bins=row_bins, density=True)
                ax.set_yscale("log")
            else : 
                ax.hist2d(filtered_observables[:,row_idx], filtered_observables[:,col_idx], weights=filtered_weights, bins=(row_bins, col_bins))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.4, wspace=0.4)
    if len(save) > 0 :
        plt.savefig(save, bbox_inches="tight")
    plt.show()


#  Make sure bins are monotonic
#
def assert_good_bins_format (bins) :
    assert len(bins) > 1
    assert bins[0] != bins[1]
    do_ascending = True if bins[1] > bins[0] else False
    last_bin = bins[0]
    for b in bins[1:] :
        if do_ascending :
            assert b > last_bin
        else :
            assert b < last_bin
        last_bin = b
    return True


#  Bin data in 1 dimension
#
def bin_data_1D (data, bins, weights=None, as_lines=False, normed=True) :
    if type(bins   ) == int        : bins    = np.linspace(np.min(data), np.max(data), bins+1)
    if type(data   ) != np.ndarray : data    = np.array(data)
    if type(weights) == type(None) : weights = np.ones(shape=data.shape)
    if normed : weights = weights / np.sum(weights)
    assert len(data) == len(weights)
    assert_good_bins_format(bins)
    z , _ = np.histogram(data, bins=bins, weights=weights, density=False)
    ez, _ = np.histogram(data, bins=bins, weights=weights*weights, density=False)
    ez    = np.sqrt(ez)
    if not as_lines :
        return 0.5 * ( bins[:-1] + bins[1:] ), z, ez
    X, Z, EZ = [], [], []
    if as_lines :
        for zp in z :
            Z .append(zp)
            Z .append(zp)
        for ezp in ez :
            EZ.append(ezp)
            EZ.append(ezp)
        for i in range(len(bins)-1) :
            X .append(bins[i])
            X .append(bins[i+1])
    return np.array(X), np.array(Z), np.array(EZ)


#  Bin data in 2 dimensions
#
def bin_data_2D (data_x, data_y, bins_x, bins_y, weights=None, normed=True) :
    if type(bins_x)  == int : bins_x = np.linspace(np.min(data_x), np.max(data_x), bins_x+1)
    if type(bins_y)  == int : bins_y = np.linspace(np.min(data_y), np.max(data_y), bins_y+1)
    if type(data_x)  != np.ndarray : data_x = np.array(data_x)
    if type(data_y)  != np.ndarray : data_y = np.array(data_y)
    if type(weights) == type(None) : weights = np.ones(shape=data_x.shape)
    if normed : weights = weights / np.sum(weights)
    assert len(data_x) == len(data_y)
    assert len(data_x) == len(weights)
    assert_good_bins_format(bins_x)
    assert_good_bins_format(bins_y)
    Z , _, _ = np.histogram2d(data_x, data_y, bins=[bins_x, bins_y], weights=weights, density=False)
    EZ, _, _ = np.histogram2d(data_x, data_y, bins=[bins_x, bins_y], weights=weights*weights, density=False)
    EZ = np.sqrt(EZ)
    return bins_x, bins_y, Z, EZ


#  Get 1D ratio between two histograms
#
def get_ratio_1D (data1, data2, bins, weights1=None, weights2=None, as_lines=False, normed=True) :
    X, Z1, EZ1 = bin_data_1D(data1, bins, weights1, as_lines=as_lines, normed=normed)
    X, Z2, EZ2 = bin_data_1D(data2, bins, weights2, as_lines=as_lines, normed=normed)
    frac_EZ1  = EZ1 / Z1
    frac_EZ2  = EZ2 / Z2
    ratio     = Z2 / Z1
    ratio_err = ratio * np.sqrt(frac_EZ1*frac_EZ1 + frac_EZ2*frac_EZ2)
    return X, ratio, ratio_err


#  Get 1D pull
#
def get_pull_1D (data_x1, data_x2, bins_x, weights1=None, weights2=None, as_lines=False) :
    X, Z1, EZ1 = bin_data_1D(data_x1, bins_x, weights1, as_lines=as_lines)
    X, Z2, EZ2 = bin_data_1D(data_x2, bins_x, weights2, as_lines=as_lines)
    num = Z2 - Z1
    den = np.sqrt(EZ1*EZ1 + EZ2*EZ2)
    return X, num/den


#  Get 2D ratio between two histograms
#
def get_ratio_2D (data_x1, data_y1, data_x2, data_y2, bins_x, bins_y, weights1=None, weights2=None) :
    X, Y, Z1, EZ1 = bin_data_2D(data_x1, data_y1, bins_x, bins_y, weights1)
    X, Y, Z2, EZ2 = bin_data_2D(data_x2, data_y2, bins_x, bins_y, weights2)
    frac_EZ1 = EZ1 / Z1
    frac_EZ2 = EZ2 / Z2
    ratio     = Z2 / Z1
    ratio_err = ratio * np.sqrt(frac_EZ1*frac_EZ1 + frac_EZ2*frac_EZ2)
    return X, Y, ratio, ratio_err


#  Get 2D pull
#
def get_pull_2D (data_x1, data_y1, data_x2, data_y2, bins_x, bins_y, weights1=None, weights2=None) :
    X, Y, Z1, EZ1 = bin_data_2D(data_x1, data_y1, bins_x, bins_y, weights1)
    X, Y, Z2, EZ2 = bin_data_2D(data_x2, data_y2, bins_x, bins_y, weights2)
    num = Z2 - Z1
    den = np.sqrt(EZ1*EZ1 + EZ2*EZ2)
    return X, Y, num/den


#  Plot the datapoints provided
#
def plot_ratio (data_num, data_den, weights_num=None, weights_den=None, keys=None, cuts=[], save="", lims=True, bins=20) :

    if type(keys) == type(None) :
        keys = [f"obs{i}" for i in range(len(observables))]
    
    if type(weights_num) == type(None) :
        weights_num = np.ones(shape=(data_num.shape[0],))
    filtered_data_num, filtered_weights_num = data_num, weights_num
    for cut in cuts :
        print(f"Filtering {cut[0]} between {cut[1]} and {cut[2]} (numerator)")
        filtered_data_num, filtered_weights_num = filter_data (filtered_data_num, filtered_weights_num, keys, cut[0], cut[1], cut[2])
    num_observables = filtered_data_num.shape[1]
    print(f"Numerator filter efficiency is {100.*np.sum(filtered_weights_num)/np.sum(weights_num):.3f}%")
    
    if type(weights_den) == type(None) :
        weights_den = np.ones(shape=(data_den.shape[0],))
    filtered_data_den, filtered_weights_den = data_den, weights_den
    for cut in cuts :
        print(f"Filtering {cut[0]} between {cut[1]} and {cut[2]} (numerator)")
        filtered_data_den, filtered_weights_den = filter_data (filtered_data_den, filtered_weights_den, keys, cut[0], cut[1], cut[2])
    print(f"Denominator filter efficiency is {100.*np.sum(filtered_weights_den)/np.sum(weights_den):.3f}%")
    assert filtered_data_den.shape[1] == filtered_data_den.shape[1]
    
    fig = plt.figure(figsize=(2*num_observables, 2*num_observables))
    for row_idx, row_obs in enumerate(keys) :
        row_lims = observable_limits[row_obs]
        if row_obs in int_observables : row_bins = np.linspace(row_lims[0]-0.5, row_lims[1]+0.5, 2+(row_lims[1]-row_lims[0]))
        else                          : row_bins = np.linspace(row_lims[0], row_lims[1], bins+1)
        if not lims : row_bins = np.linspace(-6, 6, bins+1)
        for col_idx, col_obs in enumerate(keys) :
            col_lims = observable_limits[col_obs]
            if col_obs in int_observables : col_bins = np.linspace(col_lims[0]-0.5, col_lims[1]+0.5, 2+(col_lims[1]-col_lims[0]))
            else                          : col_bins = np.linspace(col_lims[0], col_lims[1], bins+1)
            if not lims : col_bins = np.linspace(-6, 6, bins+1)   
            ax = fig.add_subplot(num_observables, num_observables, row_idx*num_observables + col_idx + 1)
            if row_idx == 0 :
                ax.set_title(col_obs, weight="bold", fontsize=12)
            if col_idx == 0 :
                ax.set_ylabel(row_obs, weight="bold", fontsize=12)
            if row_idx == col_idx :
                data_x, data_z, data_ez = get_ratio_1D(filtered_data_num[:,row_idx], filtered_data_den[:,row_idx], col_bins, filtered_weights_num, filtered_weights_den, as_lines=True)
                data_z = np.nan_to_num(data_z)
                ax.plot(data_x, data_z-1., color="k")
                ax.fill_between(data_x, data_z+data_ez-1, data_z-data_ez-1, alpha=0.2, color="grey")
                ax.set_ylim([-0.2, 0.2])
                ax.axhline(0 , linestyle="--", c="grey", linewidth=1)
            else : 
                data_x, data_y, data_z, data_ez = get_ratio_2D(filtered_data_num[:,row_idx], filtered_data_num[:,col_idx], filtered_data_den[:,row_idx], filtered_data_den[:,col_idx], row_bins, col_bins, filtered_weights_num, filtered_weights_den)
                data_z = np.nan_to_num(data_z)
                im = ax.pcolormesh(data_x, data_y, data_z.transpose()-1, cmap=custom_colormap, vmin=-0.2, vmax=0.2)
                #cf = ax.contourf(data_x, data_y, data_z-1., levels=np.linspace(-0.5, 0.5, 51), cmap="bwr") # , color=["darkred", "red", "salmon", "gold", "green", "darkgreen", "azure", "blue", "darkblue"])
                if col_idx == num_observables - 1 :
                    plt.colorbar(im, ax=ax)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.4, wspace=0.4)
    if len(save) > 0 :
        plt.savefig(save, bbox_inches="tight")
    plt.show()


#  Plot the datapoints provided
#
def plot_pull (data_num, data_den, weights_num=None, weights_den=None, keys=None, cuts=[], save="", lims=True, bins=20) :
    if type(keys) == type(None) :
        keys = [f"obs{i}" for i in range(len(observables))]
    
    if type(weights_num) == type(None) :
        weights_num = np.ones(shape=(data_num.shape[0],))
    filtered_data_num, filtered_weights_num = data_num, weights_num
    for cut in cuts :
        print(f"Filtering {cut[0]} between {cut[1]} and {cut[2]} (numerator)")
        filtered_data_num, filtered_weights_num = filter_data (filtered_data_num, filtered_weights_num, keys, cut[0], cut[1], cut[2])
    num_observables = filtered_data_num.shape[1]
    print(f"Numerator filter efficiency is {100.*np.sum(filtered_weights_num)/np.sum(weights_num):.3f}%")
    
    if type(weights_den) == type(None) :
        weights_den = np.ones(shape=(data_den.shape[0],))
    filtered_data_den, filtered_weights_den = data_den, weights_den
    for cut in cuts :
        print(f"Filtering {cut[0]} between {cut[1]} and {cut[2]} (numerator)")
        filtered_data_den, filtered_weights_den = filter_data (filtered_data_den, filtered_weights_den, keys, cut[0], cut[1], cut[2])
    print(f"Denominator filter efficiency is {100.*np.sum(filtered_weights_den)/np.sum(weights_den):.3f}%")
    assert filtered_data_den.shape[1] == filtered_data_den.shape[1]
    
    fig = plt.figure(figsize=(2*num_observables, 2*num_observables))
    for row_idx, row_obs in enumerate(keys) :
        row_lims = observable_limits[row_obs]
        if row_obs in int_observables : row_bins = np.linspace(row_lims[0]-0.5, row_lims[1]+0.5, 2+(row_lims[1]-row_lims[0]))
        else                          : row_bins = np.linspace(row_lims[0], row_lims[1], bins+1)
        if not lims : row_bins = np.linspace(-6, 6, bins+1)
        for col_idx, col_obs in enumerate(keys) :
            col_lims = observable_limits[col_obs]
            if col_obs in int_observables : col_bins = np.linspace(col_lims[0]-0.5, col_lims[1]+0.5, 2+(col_lims[1]-col_lims[0]))
            else                          : col_bins = np.linspace(col_lims[0], col_lims[1], bins+1)
            if not lims : col_bins = np.linspace(-6, 6, bins+1)   
            ax = fig.add_subplot(num_observables, num_observables, row_idx*num_observables + col_idx + 1)
            if row_idx == 0 :
                ax.set_title(col_obs, weight="bold", fontsize=12)
            if col_idx == 0 :
                ax.set_ylabel(row_obs, weight="bold", fontsize=12)
            if row_idx == col_idx :
                data_x, data_z = get_pull_1D(filtered_data_num[:,row_idx], filtered_data_den[:,row_idx], col_bins, filtered_weights_num, filtered_weights_den, as_lines=True)
                data_z = np.nan_to_num(data_z)
                ax.plot(data_x, data_z, color="k")
                ax.set_ylim([-4, 4])
                ax.axhline(0 , linestyle="--", c="grey", linewidth=1)
                ax.axhline(1 , linestyle="--", c="grey", linewidth=1)
                ax.axhline(-1, linestyle="--", c="grey", linewidth=1)
                ax.axhline(2 , linestyle="--", c="grey", linewidth=1)
                ax.axhline(-2, linestyle="--", c="grey", linewidth=1)
                ax.axhline(3 , linestyle="--", c="grey", linewidth=1)
                ax.axhline(-3, linestyle="--", c="grey", linewidth=1)
            else : 
                data_x, data_y, data_z = get_pull_2D(filtered_data_num[:,row_idx], filtered_data_num[:,col_idx], filtered_data_den[:,row_idx], filtered_data_den[:,col_idx], row_bins, col_bins, filtered_weights_num, filtered_weights_den)
                data_z = np.nan_to_num(data_z)
                im = ax.pcolormesh(data_x, data_y, data_z.transpose(), cmap=custom_colormap, vmin=-4, vmax=4)
                if col_idx == num_observables - 1 :
                    plt.colorbar(im, ax=ax)
                #cf = ax.contourf(data_x, data_y, data_z, levels=np.linspace(-5., 5., 51), cmap="bwr") #, color=["darkred", "red", "salmon", "gold", "green", "darkgreen", "azure", "blue", "darkblue"])
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.4, wspace=0.4)
    if len(save) > 0 :
        plt.savefig(save, bbox_inches="tight")
    plt.show()