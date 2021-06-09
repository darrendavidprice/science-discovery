#======================================#
#   Brief:  Plot a likelihood model    #
#   Author: stmenary@cern.ch           #
#======================================#


#=======================#
#  1. Required imports  #
#=======================#

print("Importing standard library")
import argparse, math, os, sys, time

print("Importing python data libraries")
import numpy as np
from   matplotlib import cm, colors, pyplot as plt
from   matplotlib.colors import ListedColormap, LinearSegmentedColormap

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

import train_0D_density_model


#==============================#
#  2. Configuration fallbacks  #
#==============================#

#  Input data

config_fname = ""
num_evts     = int(-1)
num_proc     = int(8)

log_observables = "pT_j1", "pT_j2", "pT_jj", "pT_ll", "m_jj", "rap_jj"

VBFZ.obs_ticks ["rap_jj"] = [1, 2.5, 4]
VBFZ.obs_ticklabels ["rap_jj"] = ["1", "2.5", "4"]

log_axis_functions = (lambda x : x**(1./3.), lambda x : x*x*x)

cmap_bwr  = cm.get_cmap('bwr', 256)
newcolors = cmap_bwr(np.linspace(0, 1, 256))
newcolors [math.ceil(256*2/6)-1:math.floor(256*4/6)-1] = np.array([68/256, 223/256, 68/256, 1])
custom_colormap = ListedColormap(newcolors, name='BlueToRed')


def get_adaptive_bins(data, weights=None, obs=None, num_increments=100, max_err=0.05, is_latent=False) :
    if not is_latent :
        if (type(obs) != type(None)) and (obs == "N_gap_jets") : return np.array([-0.5, 0.5, 2.5])
        if (type(obs) != type(None)) and (obs == "N_jets")     : return np.array([1.5, 2.5, 5.5])
        if obs : limits = VBFZ.observable_limits[obs]
        else   : limits = [np.min(data), np.max(data)]
    elif (obs == "N_gap_jets") : return np.array([-0.5, 0.5, 2.5])
    elif (obs == "N_jets")     : return np.array([1.5, 2.5, 5.5]) - 2.
    else : limits = [-5, 5]
    if type(weights) == type(None) : weights   = np.ones(shape=(len(data),))
    bins = [x for x in np.linspace(limits[0], limits[1], 1+num_increments)]
    keep_merging = True
    while keep_merging :
        vals, _   = np.histogram(data, bins=bins)
        errs      = np.sqrt(vals)
        frac_errs = [x for x in plot.safe_divide(errs, vals)]
        curr_err  = np.max(frac_errs)
        if len(bins) == 2 :
            keep_merging = False
        elif 0 in frac_errs :
            idx = len(frac_errs) - frac_errs[::-1].index(0) - 1
            del bins[idx]
        elif curr_err < max_err :
            keep_merging = False
        else :
            idx = frac_errs.index(curr_err)
            if   idx == 0             : del bins[1]
            elif idx == len(bins) - 2 : del bins[-2]
            elif frac_errs[idx-1] > frac_errs[idx>1] : del bins[idx]
            else : del bins[idx+1]
    return np.array(bins)


def get_bins_latent (obs, num_bins=20) :
    #global int_observables, transformed_observable_limits  #  VBFZ-tag
    transformed_observable_limits = VBFZ.transformed_observable_limits  #  NB-tag
    observable_limits             = VBFZ.observable_limits  #  NB-tag
    int_observables               = VBFZ.int_observables   #  NB-tag
    if obs in int_observables :
        obs_lims = observable_limits[obs]
        #obs_lims = transformed_observable_limits[obs]
        return np.linspace(obs_lims[0]-0.5, obs_lims[1]+0.5, 2+(obs_lims[1]-obs_lims[0]))
    return np.linspace(-5, 5, num_bins+1)

def get_bins_physical (obs, num_bins=20, base=np.e) :
    #global int_observables, observable_limits  #  VBFZ-tag
    observable_limits = VBFZ.observable_limits  #  NB-tag
    int_observables   = VBFZ.int_observables   #  NB-tag
    obs_lims = observable_limits[obs] 
    if obs in int_observables : 
        return np.linspace(obs_lims[0]-0.5, obs_lims[1]+0.5, 2+(obs_lims[1]-obs_lims[0]))
    if obs in log_observables :
        log_physical_limits = np.log(np.array(obs_lims) + 10) / np.log(base)
        bins = np.exp(np.linspace(log_physical_limits[0], log_physical_limits[1], num_bins+1)*np.log(base)) - 10
        if np.fabs(bins[0]) < 1e-15 : bins[0] = 0
        return bins
    return np.linspace(obs_lims[0], obs_lims[1], num_bins+1)
            
            
def get_bins (obs, is_latent=False, num_bins=20) :
    if is_latent :
        return get_bins_latent (obs, num_bins=num_bins)
    return get_bins_physical (obs, num_bins=num_bins)


def get_obs_label (obs) :
    return VBFZ.get_obs_label(obs)
    

def get_obs_ticks (obs, is_latent=False) :
    #global int_observables  #  VBFZ-tag
    int_observables = VBFZ.int_observables   #  NB-tag
    if is_latent :
        if obs not in int_observables : return np.array([-3, 0, 3])
        return VBFZ.get_obs_ticks(obs)
    return VBFZ.get_obs_ticks(obs)


def get_obs_ticklabels (obs, is_latent=False) :
    #global int_observables  #  VBFZ-tag
    int_observables = VBFZ.int_observables   #  NB-tag
    if is_latent :
        if obs not in int_observables : return np.array(["-3", "0", "3"])
        return VBFZ.get_obs_ticklabels(obs)
    return VBFZ.get_obs_ticklabels(obs)


def get_obs_for_2D_plot (observables) :
    num_observables = len(observables)
    obs_to_plot = []
    for obs_idx_x, obs_x in enumerate(observables) :
        if obs_idx_x == num_observables-1 : continue  #  Don't plot observable -1 on x axis
        for obs_idx_y, obs_y in enumerate(observables) :
            if obs_idx_y == 0         : continue   #  Don't plot observable 0 on y axis
            if obs_idx_y <= obs_idx_x : continue   #  Don't plot above diagonal on y axis
            obs_to_plot.append((obs_idx_x, obs_x, obs_idx_y, obs_y))
    return obs_to_plot


def parse_args () :
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fname', type=str, help='Name of the config file used to run train_0D_density_model.')
    parser.add_argument('--nevt'      , type=int, default=-1, help='Number of events to generate with the density model. Default and values <= 0 mean to match the number of MG5 events.')
    parser.add_argument('--nproc'     , type=int, default=-1, help='Number of processes used to generate events.')
    args = parser.parse_args()
    global config_fname, num_evts, num_proc
    config_fname, num_evts, num_proc = args.config_fname, args.nevt, args.nproc


def load_model (load_model_dir) :
    print(f"Loading density model from file {load_model_dir}")
    model = DensityModel.from_dir(load_model_dir)
    return model


def generate (model, num_evts, data_table, whitening_funcs, num_proc=8) :
    num_evts = num_evts if num_evts > 0 else data_table.get_num_events()
    print(f"Generating {num_evts} fake datapoints")
    start = time.time()
    fake_white_datapoints = model.sample(num_evts, [1.], n_processes=num_proc)
    end = time.time()
    print(f"{num_evts} datapoints generated in {int(end-start):.0f}s")
    print("Unwhitening fake datapoints")
    start = time.time()
    fake_transformed_datapoints = unwhiten_axes(fake_white_datapoints, whitening_funcs)
    fake_datapoints = VBFZ.transform_observables_back(fake_transformed_datapoints, data_table.keys)
    end = time.time()
    print(f"{num_evts} datapoints unwhitened in {int(end-start):.0f}s")
    return fake_white_datapoints, fake_transformed_datapoints, fake_datapoints


def plot_2D_projections (datapoints, weights=None, label="", savefig="", is_latent=False, num_bins=20, vmin=1e-5) :
    """plot the 2D projections of the datapoints provided"""
    
    #global observables, num_observables, observable_limits, transformed_observable_limits, int_observables, log_observables   # VBFZ-tag
    observables      , num_observables = VBFZ.observables      , VBFZ.num_observables    # NB-tag
    observable_limits, int_observables = VBFZ.observable_limits, VBFZ.int_observables    # NB-tag
    transformed_observable_limits = VBFZ.transformed_observable_limits    # NB-tag
    #
    #  If no weights provided then assume uniform
    #
    if type(weights) == type(None) :
        weights = np.ones(shape=(datapoints.shape[0],))
    #
    #  Save the list of indices to plot (to make sure all loops are over consistent sets)
    #
    norm_const = {}
    obs_to_plot = get_obs_for_2D_plot (observables)
    for obs_idx_x, obs_x, obs_idx_y, obs_y in obs_to_plot :
        bins_x, bins_y = get_bins(obs_x, is_latent=is_latent, num_bins=num_bins), get_bins(obs_y, is_latent=is_latent, num_bins=num_bins)
        vals, _, _     = np.histogram2d(datapoints[:,obs_idx_x], datapoints[:,obs_idx_y], weights=weights, bins=(bins_x, bins_y))
        norm_const[(obs_idx_x, obs_idx_y)] = np.nanmax(vals.flatten())
    #
    #  Make plot
    #
    fig = plt.figure(figsize=(20, 14))
    for obs_idx_x, obs_x, obs_idx_y, obs_y in obs_to_plot :
        xlo    = obs_idx_x / (num_observables-1)    #  Get axis x coordinates
        xwidth = 1.        / (num_observables-1)
        ylo     = (num_observables-obs_idx_y-1) / (num_observables-1)   #  Get axis y coordinates
        yheight = 1.                            / (num_observables-1)
        #
        #  Create axis
        #
        ax = fig.add_axes([xlo, ylo, 0.95*xwidth, 0.95*yheight])
        #
        #  Format log axes
        #
        if not is_latent :
            if obs_x in log_observables : ax.set_xscale("function", functions=log_axis_functions )
            if obs_y in log_observables : ax.set_yscale("function", functions=log_axis_functions )
        #
        #  Draw axis ticks and labels
        #
        ax.set_xticks(get_obs_ticks(obs_x, is_latent=is_latent))
        ax.set_yticks(get_obs_ticks(obs_y, is_latent=is_latent))
        if obs_idx_y == num_observables-1 : 
            ax.get_xaxis().set_ticklabels(get_obs_ticklabels(obs_x, is_latent=is_latent))
            ax.set_xlabel(get_obs_label(obs_x).replace("  [","\n["), fontsize=19, labelpad=20, va="top", ha="center")
        else :
            ax.get_xaxis().set_ticklabels([])
        if obs_idx_x == 0 : 
            ax.get_yaxis().set_ticklabels(get_obs_ticklabels(obs_y, is_latent=is_latent))
            ax.set_ylabel(get_obs_label(obs_y).replace("  [","\n["), fontsize=19, labelpad=20, rotation=0, va="center", ha="right")
        else :
            ax.get_yaxis().set_ticklabels([])
        #
        #  Format tick params
        #
        ax.tick_params(which="both", right=True, top=True, direction="in", labelsize=15)
        #
        #  Draw histogram
        #
        bins_x, bins_y = get_bins(obs_x, is_latent=is_latent, num_bins=num_bins), get_bins(obs_y, is_latent=is_latent, num_bins=num_bins)
        _, _, _, patches = ax.hist2d(datapoints[:,obs_idx_x], datapoints[:,obs_idx_y], weights=weights/norm_const[(obs_idx_x, obs_idx_y)], bins=(bins_x, bins_y),
                                  vmin=vmin, vmax=1, norm=colors.LogNorm(), cmap="inferno")
        #
        #  Draw label
        #
        if (obs_idx_x==0) and (obs_idx_y==1) and len(label) > 0 :
            ax.text(0, 1.2, label, weight="bold", ha="left", va="bottom", transform=ax.transAxes, fontsize=21)
    #
    #  Draw colour bar
    #
    cbar_ax = fig.add_axes([0.76, 0.5, 0.03, 0.45])
    cbar    = fig.colorbar(patches, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=14)
    cbar   .set_ticks([1, 0.1, 0.01, 0.001, 0.0001, 1e-5])
    cbar   .set_label(r"$\frac{p(x)}{{\rm max}~p(x)}$", fontsize=25, labelpad=50, rotation=0, va="center")
    #
    #  Save and show plot
    #
    if len(savefig) > 0 :
        plt.savefig(savefig, bbox_inches="tight")
    else :
        plt.show()


def plot_2D_ratios (datapoints_num, datapoints_den, weights_num=None, weights_den=None, label="", savefig="", is_latent=False, num_bins=20) :
    
    #global observables, num_observables, observable_limits, transformed_observable_limits, int_observables, log_observables   # VBFZ-tag
    observables      , num_observables = VBFZ.observables      , VBFZ.num_observables    # NB-tag
    observable_limits, int_observables = VBFZ.observable_limits, VBFZ.int_observables    # NB-tag
    transformed_observable_limits = VBFZ.transformed_observable_limits    # NB-tag
    #
    #  If no weights provided then assume uniform
    #
    if type(weights_num) == type(None) : weights_num = np.ones(shape=(datapoints_num.shape[0],))
    if type(weights_den) == type(None) : weights_den = np.ones(shape=(datapoints_den.shape[0],))
    #
    #  Save the list of indices to plot (to make sure all loops are over consistent sets)
    #
    obs_to_plot = get_obs_for_2D_plot (observables)
    #
    #  Make plot
    #
    fig = plt.figure(figsize=(20, 14))
    vmin = 1e-5
    for obs_idx_x, obs_x, obs_idx_y, obs_y in obs_to_plot :
        xlo    = obs_idx_x / (num_observables-1)    #  Get axis x coordinates
        xwidth = 1.        / (num_observables-1)
        ylo     = (num_observables-obs_idx_y-1) / (num_observables-1)   #  Get axis y coordinates
        yheight = 1.                            / (num_observables-1)
        #
        #  Create axis
        #
        ax = fig.add_axes([xlo, ylo, 0.95*xwidth, 0.95*yheight])
        #
        #  Format log axes
        #
        if not is_latent :
            if obs_x in log_observables : ax.set_xscale("function", functions=log_axis_functions )
            if obs_y in log_observables : ax.set_yscale("function", functions=log_axis_functions )
        #
        #  Draw axis ticks and labels
        #
        ax.set_xticks(get_obs_ticks(obs_x, is_latent=is_latent))
        ax.set_yticks(get_obs_ticks(obs_y, is_latent=is_latent))
        if obs_idx_y == num_observables-1 : 
            ax.get_xaxis().set_ticklabels(get_obs_ticklabels(obs_x, is_latent=is_latent))
            ax.set_xlabel(get_obs_label(obs_x).replace("  [","\n["), fontsize=19, labelpad=20, va="top", ha="center")
        else :
            ax.get_xaxis().set_ticklabels([])
        if obs_idx_x == 0 : 
            ax.get_yaxis().set_ticklabels(get_obs_ticklabels(obs_y, is_latent=is_latent))
            ax.set_ylabel(get_obs_label(obs_y).replace("  [","\n["), fontsize=19, labelpad=20, rotation=0, va="center", ha="right")
        else :
            ax.get_yaxis().set_ticklabels([])
        #
        #  Format tick params
        #
        ax.tick_params(which="both", right=True, top=True, direction="in", labelsize=15)
        #
        #  Draw histogram
        #
        bins_x, bins_y = get_bins(obs_x, is_latent=is_latent, num_bins=num_bins), get_bins(obs_y, is_latent=is_latent, num_bins=num_bins)
        X, Y, ratio, ratio_err = plot.get_ratio_2D (datapoints_num[:,obs_idx_x], datapoints_num[:,obs_idx_y],
                                                    datapoints_den[:,obs_idx_x], datapoints_den[:,obs_idx_y],
                                                    bins_x, bins_y, weights1=weights_num, weights2=weights_den)            
        im = ax.pcolormesh(X, Y, ratio.transpose()-1, cmap=custom_colormap, vmin=-0.3, vmax=0.3)
        #
        #  Draw label
        #
        if (obs_idx_x==0) and (obs_idx_y==1) and len(label) > 0 :
            ax.text(0, 1.2, label, weight="bold", ha="left", va="bottom", transform=ax.transAxes, fontsize=21)
    #
    #  Draw colour bar
    #
    cbar_ax = fig.add_axes([0.76, 0.5, 0.03, 0.45])
    cbar    = fig.colorbar(im, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=14)
    cbar   .set_ticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    cbar   .set_label(r"$\frac{p(x)}{{\rm max}~p(x)}$", fontsize=25, labelpad=50, rotation=0, va="center")
    #
    #  Save and show plot
    #
    if len(savefig) > 0 :
        plt.savefig(savefig, bbox_inches="tight")
    else :
        plt.show()


def plot_1D_projections (datapoints_num, datapoints_den, weights_num=None, weights_den=None, savefig="", is_latent=False, num_bins=20, max_cols=6) :
    """plot the 1D projections of the datapoints provided"""
    
    #global observables, num_observables, observable_limits, transformed_observable_limits, int_observables, log_observables   # VBFZ-tag
    observables      , num_observables = VBFZ.observables      , VBFZ.num_observables    # NB-tag
    observable_limits, int_observables = VBFZ.observable_limits, VBFZ.int_observables    # NB-tag
    transformed_observable_limits = VBFZ.transformed_observable_limits    # NB-tag
    #
    #  If no weights provided then assume uniform
    #
    if type(weights_num) == type(None) : weights_num = np.ones(shape=(datapoints_num.shape[0],))
    if type(weights_den) == type(None) : weights_den = np.ones(shape=(datapoints_den.shape[0],))
    #
    #  Calculate out plot dimensions and create figure
    #
    num_cols = np.min([max_cols, num_observables])
    num_rows = math.ceil(num_observables/num_cols)
    fig = plt.figure(figsize=(2*num_cols, 6*num_rows))
    #
    #  Loop over subplots
    #
    axes1, axes2 = [], []
    ymins, ymaxs = [], []
    for row_idx in range(num_rows) :
        for col_idx in range(num_cols) :
            obs_idx = num_cols*row_idx + col_idx
            if obs_idx >= num_observables : continue
            observable = observables[obs_idx]
            #
            #  Get axis co-ordinates
            #
            xlo, xwidth  = col_idx/num_cols, 1./num_cols
            ylo, yheight = 1. - (1+row_idx)/num_rows, 1./num_rows
            #
            #
            #  Get values of distributions
            #
            #  get binning
            bins       = get_adaptive_bins(datapoints_num[:,obs_idx], weights_num, obs=observable, num_increments=30, max_err=0.02, is_latent=is_latent)
            bin_widths = bins[1:] - bins[:-1]
            #  numerator histo values
            hvals_num, _ = np.histogram(datapoints_num[:,obs_idx], bins=bins, weights=weights_num            )
            herrs_num, _ = np.histogram(datapoints_num[:,obs_idx], bins=bins, weights=weights_num*weights_num)
            herrs_num    = np.sqrt(herrs_num)
            hvals_num, herrs_num = hvals_num/np.sum(weights_num)/bin_widths, herrs_num/np.sum(weights_num)/bin_widths
            norm_vals    = np.max(hvals_num)
            hvals_num, herrs_num = hvals_num / norm_vals, herrs_num / norm_vals
            #  denominator histo values
            hvals_den, _ = np.histogram(datapoints_den[:,obs_idx], bins=bins, weights=weights_den            )
            herrs_den, _ = np.histogram(datapoints_den[:,obs_idx], bins=bins, weights=weights_den*weights_den)
            herrs_den    = np.sqrt(herrs_den)
            hvals_den, herrs_den = hvals_den/np.sum(weights_den)/bin_widths, herrs_den/np.sum(weights_den)/bin_widths
            hvals_den, herrs_den = hvals_den / norm_vals, herrs_den / norm_vals
            #  histograms expressed as lines
            plot_x, plot_y_num, plot_ey_num = plot.histo_to_line(bins, hvals_num, herrs_num)
            _     , plot_y_den, plot_ey_den = plot.histo_to_line(bins, hvals_den, herrs_den)
            #
            #  Create absolute distribution plot (top panel of each observable)
            #
            ax1 = fig.add_axes([xlo, ylo+0.6*yheight, 0.95*xwidth, 0.38*yheight])
            ax1.plot(plot_x, plot_y_num, "-", color="k"      , linewidth=2, label="MG5 events")
            ax1.fill_between(plot_x, plot_y_num-plot_ey_num, plot_y_num+plot_ey_num, color="lightgrey", alpha=1)
            ax1.plot(plot_x, plot_y_den, "-", color="darkred", linewidth=2, label="Samples from density model")
            ax1.fill_between(plot_x, plot_y_den-plot_ey_den, plot_y_den+plot_ey_den, color="red", alpha=0.2)
            ax1.set_yscale("log")
            #
            # Save ymin, ymax and top axis for this observable
            #
            ymin, ymax = np.min([plot_y_num-plot_ey_num, plot_y_den-plot_ey_den]), np.max([plot_y_num+plot_ey_num, plot_y_den+plot_ey_den])
            ymins.append(ymin)
            ymaxs.append(ymax)
            axes1.append(ax1)
            #
            #  Create ratio plot (bottom panel of each observable) and save it
            #
            plot_ey_diff = np.sqrt(plot_ey_num*plot_ey_num + plot_ey_den*plot_ey_den)
            ax2 = fig.add_axes([xlo, ylo+0.2*yheight, 0.95*xwidth, 0.38*yheight])
            ax2.axhline(0, c="darkred", linewidth=2)
            ax2.fill_between(plot_x, -safe_divide(plot_ey_den, plot_y_den), safe_divide(plot_ey_den, plot_y_den), color="red", alpha=0.2)
            ax2.plot(plot_x, safe_divide(plot_y_num-plot_y_den, plot_y_den), c="k", linewidth=2)
            ax2.fill_between(plot_x, safe_divide(plot_y_num-plot_ey_diff-plot_y_den, plot_y_den), safe_divide(plot_y_num+plot_ey_diff-plot_y_den, plot_y_den), color="lightgrey", alpha=0.5)
            axes2.append(ax2)
            #
            #  Set ylim and draw horizontal reference lines
            #
            ax2.set_ylim([-0.12, 0.12])
            for h in [-0.1, -0.05, 0.05, 0.1] :
                ax2.axhline(h, linestyle="--", c="grey", linewidth=0.5)
            #  
            #  Set x limits and scale
            #  
            print(observable, bins[0], bins[-1])
            ax1.set_xlim([bins[0], bins[-1]])
            ax2.set_xlim([bins[0], bins[-1]])
            if not is_latent :
                if observable in log_observables :
                    ax1.set_xscale("function", functions=log_axis_functions )
                    ax2.set_xscale("function", functions=log_axis_functions )
            #
            #  Set axis ticks
            #   
            if col_idx > 0 :
                ax1.get_yaxis().set_ticklabels([])
                ax2.get_yaxis().set_ticklabels([])
            ax1.set_xticks(get_obs_ticks(observable, is_latent=is_latent))
            ax2.set_xticks(get_obs_ticks(observable, is_latent=is_latent))
            ax1.get_xaxis().set_ticklabels([])
            ax2.get_xaxis().set_ticklabels(get_obs_ticklabels(observable, is_latent=is_latent))
            #   
            #  Set axis labels
            #   
            ax2.set_xlabel(get_obs_label(observable), fontsize=19, labelpad=20)
            if col_idx == 0 : 
                ax1.set_ylabel("Normalised\nentries", fontsize=19, labelpad=75, rotation=0, va="center")
                ax2.set_ylabel("Ratio to\ndensity\nmodel", fontsize=19, labelpad=65, rotation=0, va="center")
                ax2.set_yticks     ([-0.1, -0.05, 0, 0.05, 0.1])
                ax2.set_yticklabels([r"$-10\%$", r"$-5\%$", r"$0$", r"$+5\%$", r"$+10\%$"])
            #  
            #  Set tick params
            #  
            ax1.tick_params(which="both", right=True, top=True, direction="in", labelsize=15)
            ax2.tick_params(which="both", right=True, top=True, direction="in", labelsize=15)
    #
    #  Set consistent axis y lims
    #
    ymin, ymax = np.min([y for y in ymins if y > 0])/2., 2.*np.max(ymaxs)
    for ax in axes1 :
        ax.set_ylim([ymin, ymax])
    #
    #  Set y-axis ticks and legend
    #
    axes1[0].legend(loc=(0, 1.05), frameon=True, edgecolor="white", facecolor="white", ncol=2, fontsize=17)
    #
    #  Save and show figure
    #
    if len(savefig) > 0 :
        plt.savefig(savefig, bbox_inches="tight")
    else :
        plt.show()


def plot_model_training_curves (model, savefig="", max_cols=6) :
    observables, num_observables = VBFZ.observables, VBFZ.num_observables
    #
    num_cols = np.min([max_cols, num_observables])
    num_rows = math.ceil(num_observables/num_cols)
    fig = plt.figure(figsize=(3*num_cols, 5*num_rows))
    #
    #  Loop over subplots
    #
    axes = []
    for row_idx in range(num_rows) :
        for col_idx in range(num_cols) :
            obs_idx = num_cols*row_idx + col_idx
            if obs_idx >= num_observables : continue
            observable = observables[obs_idx]
            #
            #  Get axis co-ordinates
            #
            xlo, xwidth  = col_idx/num_cols, 1./num_cols
            ylo, yheight = 1. - (1+row_idx)/num_rows, 1./num_rows
            #   
            #  Set axis labels
            #   
            ax = fig.add_axes([xlo+0.2*xwidth, ylo, 0.75*xwidth, .7*yheight])
            ax.set_xlabel(get_obs_label(observable), fontsize=19, labelpad=20)
            if col_idx == 0 : 
                ax.set_ylabel(r"$- \log \mathcal{L}(\vec x)$", fontsize=19, labelpad=75, rotation=0, va="center")
            axes.append(ax)
            #
            #  Plot curve
            #
            if hasattr(model.likelihood_models[obs_idx].model, "monitor_record") is False : continue
            training_profile = model.likelihood_models[obs_idx].model.monitor_record
            ax.plot(training_profile, c="blue", lw=2, alpha=0.8, label="Training profile")
            print(obs_idx, observable, "-logL = ", np.min(training_profile))
            if hasattr(model.likelihood_models[obs_idx].model, "lr_record") is False : continue
            is_first = True
            for (epoch, new_lr) in model.likelihood_models[obs_idx].model.lr_record :
                ax.axvline(epoch, ls="--", c="k")
                label = r"L.R. $\times = 0.5$" if is_first else None
                is_first = False
            ax.set_yscale("symlog", linthresh=1e-11)
            print(observable, [np.min(training_profile), np.max(training_profile[-int(0.2*len(training_profile)):])])
            ax.set_ylim([np.min(training_profile), np.max(training_profile[-int(0.2*len(training_profile)):])])
    axes[0].legend(loc=(0, 1.05), frameon=True, edgecolor="white", facecolor="white", ncol=2, fontsize=17)
    #
    #  Save and show figure
    #
    if len(savefig) > 0 :
        plt.savefig(savefig, bbox_inches="tight")
    plt.show()


#====================================#
#  N. Fallback to running as script  #
#====================================#

if __name__ == "__main__" :
    parse_args ()
    train_0D_density_model.load_settings(config_fname)
    train_0D_density_model.print_settings()
    train_0D_density_model.VBFZ_setup()

    data_table = VBFZ.load_table(train_0D_density_model.input_fname)
    whitening_funcs, true_data, true_data_weights, transformed_data, white_data = train_0D_density_model.get_original_and_projected_data_as_dict (data_table)
    whitening_funcs = pickle.load(open(train_0D_density_model.load_whitening_funcs, "rb"))
    true_data, true_data_weights, transformed_data, white_data = true_data[1.], true_data_weights[1.], transformed_data[1.], white_data[1.]

    model = load_model(train_0D_density_model.save_model_dir)
    plot_model_training_curves(model, savefig=f"{train_0D_density_model.save_model_dir}/Training_curves.pdf")

    fake_white_datapoints, fake_transformed_datapoints, fake_datapoints = generate(model, num_evts=num_evts, data_table=data_table, whitening_funcs=whitening_funcs, num_proc=num_proc)

    plot_2D_ratios(fake_white_datapoints, white_data, weights_den=true_data_weights, is_latent=True, 
                               label="Samples from density model / MG5 events (latent space)",
                               savefig=f"{train_0D_density_model.save_model_dir}/2D_ratios_latent.pdf")

    plot_2D_projections(white_data, weights=true_data_weights, is_latent=True, 
                        label="MG5 events (latent space)",
                        savefig=f"{train_0D_density_model.save_model_dir}/2D_dist_MG5_latent.pdf")

    plot_2D_projections(fake_white_datapoints, is_latent=True, 
                        label="Samples from density model (latent space)",
                        savefig=f"{train_0D_density_model.save_model_dir}/2D_dist_model_latent.pdf")

    plot_2D_ratios(fake_datapoints, true_data, weights_den=true_data_weights, is_latent=False, 
                   label="Samples from density model / MG5 events (physical space)",
                   savefig=f"{train_0D_density_model.save_model_dir}/2D_ratios_physical.pdf")

    plot_2D_projections(true_data, weights=true_data_weights, is_latent=False, 
                        label="MG5 events (physical space)",
                        savefig=f"{train_0D_density_model.save_model_dir}/2D_dist_MG5_physical.pdf",
                        vmin=1e-4)

    plot_2D_projections(fake_datapoints, is_latent=False, 
                        label="Samples from density model (physical space)",
                        savefig=f"{train_0D_density_model.save_model_dir}/2D_dist_model_physical.pdf",
                        vmin=1e-4)

    plot_1D_projections(true_data, fake_datapoints, weights_num=true_data_weights, is_latent=False,
                        savefig=f"{train_0D_density_model.save_model_dir}/1D_dist_physical.pdf")

    plot_1D_projections(white_data, fake_white_datapoints, weights_num=true_data_weights, is_latent=True,
                        savefig=f"{train_0D_density_model.save_model_dir}/1D_dist_latent.pdf")

