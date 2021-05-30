#   ==================================================================================
#   Brief : Implement configuration details common to all analyses on the VBFZ dataset
#   Author: Stephen Menary (sbmenary@gmail.com)
#   ==================================================================================


#  ================
#  Required imports
#  ================

import numpy as np

from matplotlib import pyplot as plt, colors

from .data_preparation import DataTable


#  ===========
#  Data config
#  ===========

observable_limits = {}
observable_limits ["m_ll"      ] = [75    , 105  ]
observable_limits ["pT_ll"     ] = [0     , 900  ]
observable_limits ["theta_ll"  ] = [0     , np.pi]
observable_limits ["rap_ll"    ] = [0     , 2.2  ]
observable_limits ["m_jj"      ] = [150   , 5000 ]
observable_limits ["pT_jj"     ] = [0     , 900  ]
observable_limits ["theta_jj"  ] = [0     , np.pi]
observable_limits ["rap_jj"    ] = [0     , 4.4  ]
observable_limits ["pT_j1"     ] = [60    , 1200 ]
observable_limits ["pT_j2"     ] = [40    , 1200 ]
observable_limits ["Dy_j_j"    ] = [0     , 8.8  ]
observable_limits ["Dphi_j_j"  ] = [-np.pi, np.pi]
observable_limits ["N_jets"    ] = [2     , 5    ]
observable_limits ["N_gap_jets"] = [0     , 2    ]

transformed_observable_limits = {k:[x[0], x[1]] for k,x in observable_limits.items()}
#transformed_observable_limits ["N_jets" ] = [0, 5]
#transformed_observable_limits ["pT_j1"  ] = [0, 1200]
#transformed_observable_limits ["pT_jj"  ] = [0, 1]
#transformed_observable_limits ["Dy_j_j" ] = [0, 1.5]

all_observables  = sorted([obs for obs in observable_limits])      #  Persistent record of all observables available
int_observables  = ["N_jets", "N_gap_jets"]                        #  Persistent record of which observables are integers
observables      = sorted([obs for obs in observable_limits])      #  Modifiable configuration variable, labels observables considered for a given analysis
num_observables  = len(observables)                                #  Modifiable configuration variable, update concurrently with observables to avoid mismatch
observable_types = [float if obs not in int_observables else int for obs in observables]

obs_labels = {}
obs_labels ["Dphi_j_j"  ] = r"$\Delta\phi\left(j,j\right)$"
obs_labels ["Dy_j_j"    ] = r"$\Delta y\left(j,j\right)$"
obs_labels ["N_gap_jets"] = r"$N_{\rm gap jet}$"
obs_labels ["N_jets"    ] = r"$N_{\rm jet}$"
obs_labels ["m_jj"      ] = r"$m_{jj}$  [TeV]"
obs_labels ["m_ll"      ] = r"$m_{ll}$  [GeV]"
obs_labels ["pT_j1"     ] = r"$p_{T}^{j1}$  [GeV]"
obs_labels ["pT_j2"     ] = r"$p_{T}^{j2}$  [GeV]"
obs_labels ["pT_jj"     ] = r"$p_{T}^{jj}$  [GeV]"
obs_labels ["pT_ll"     ] = r"$p_{T}^{ll}$  [GeV]"
obs_labels ["rap_jj"    ] = r"$|y^{jj}|$"
obs_labels ["rap_ll"    ] = r"$|y^{ll}|$"
obs_labels ["theta_jj"  ] = r"$|\theta^{jj}|$"
obs_labels ["theta_ll"  ] = r"$|\theta^{ll}|$"

transformed_obs_labels = {k:x for k,x in obs_labels.items()}
transformed_obs_labels ["Dy_j_j"] = r"$\Delta y\left(j,j\right)'$"
transformed_obs_labels ["N_jets"] = r"$N_{\rm jet}'$"
transformed_obs_labels ["pT_j1" ] = r"$p_{T}^{j1}'$"
transformed_obs_labels ["pT_jj" ] = r"$p_{T}^{jj}'$"

obs_ticks = {}
obs_ticks ["Dphi_j_j"  ] = [-2, 0, 2]
obs_ticks ["Dy_j_j"    ] = [2, 4, 6, 8]
obs_ticks ["N_gap_jets"] = [0, 1, 2]
obs_ticks ["N_jets"    ] = [2, 3, 4, 5]
obs_ticks ["m_jj"      ] = [1000, 2500, 4000]
obs_ticks ["m_ll"      ] = [80, 90, 100]
obs_ticks ["pT_j1"     ] = [300, 600, 900]
obs_ticks ["pT_j2"     ] = [300, 600, 900]
obs_ticks ["pT_jj"     ] = [150, 450, 750]
obs_ticks ["pT_ll"     ] = [150, 450, 750]
obs_ticks ["rap_jj"    ] = [1, 2.5, 4]
obs_ticks ["rap_ll"    ] = [0.3, 1.1, 1.9]
obs_ticks ["theta_jj"  ] = [np.pi/4., np.pi/2., 3.*np.pi/4.]
obs_ticks ["theta_ll"  ] = [np.pi/4., np.pi/2., 3.*np.pi/4.]

transformed_obs_ticks = {k:[x[0], x[1]] for k,x in obs_ticks.items()}
transformed_obs_ticks ["Dy_j_j"] = [0.2, 0.5, 0.8]
transformed_obs_ticks ["N_jets"] = [0, 1, 2, 3, 4, 5]
transformed_obs_ticks ["pT_j1" ] = [0.2, 0.5, 0.8]
transformed_obs_ticks ["pT_jj" ] = [0.2, 0.5, 0.8]

obs_ticklabels = {}
obs_ticklabels ["Dphi_j_j"  ] = ["-2", "0", "2"]
obs_ticklabels ["Dy_j_j"    ] = ["2", "4", "6", "8"]
obs_ticklabels ["N_gap_jets"] = ["0", "1", "2"]
obs_ticklabels ["N_jets"    ] = ["2", "3", "4", "5"]
obs_ticklabels ["m_jj"      ] = ["1" , "2.5", "4"]
obs_ticklabels ["m_ll"      ] = ["80", "90", "100"]
obs_ticklabels ["pT_j1"     ] = ["300", "600", "900"]
obs_ticklabels ["pT_j2"     ] = ["300", "600", "900"]
obs_ticklabels ["pT_jj"     ] = ["150", "450", "750"]
obs_ticklabels ["pT_ll"     ] = ["150", "450", "750"]
obs_ticklabels ["rap_jj"    ] = ["1", "2.5", "4"]
obs_ticklabels ["rap_ll"    ] = ["0.3", "1.1", "1.9"]
obs_ticklabels ["theta_jj"  ] = [r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$"]
obs_ticklabels ["theta_ll"  ] = [r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$"]

transformed_obs_ticklabels = {k:[x[0], x[1]] for k,x in obs_ticklabels.items()}
#transformed_obs_ticklabels ["Dy_j_j"] = ["0.2", "0.5", "0.8"]
transformed_obs_ticklabels ["N_jets"] = ["0", "1", "2", "3", "4", "5"]
transformed_obs_ticklabels ["pT_j1" ] = None
transformed_obs_ticklabels ["pT_jj" ] = ["0.2", "0.5", "0.8"]


def configure (remove_more_observables, reverse=False) :
    """Select observables for analysis by configuring global variables observables, num_observables and observable_types"""
    global remove_observables
    remove_observables = remove_more_observables
    if "theta_ll" not in remove_observables : remove_observables.append("theta_ll")
    if "theta_jj" not in remove_observables : remove_observables.append("theta_jj")
    global observables, num_observables, observable_types
    observables = sorted([obs for obs in observable_limits if obs not in remove_observables])
    if reverse : observables = observables[::-1]
    num_observables  = len(observables)
    observable_types = [float if obs not in int_observables else int for obs in observables]

def get_obs_label (obs) :
    if obs in obs_labels :
        return obs_labels[obs]
    return obs         

def get_obs_ticks (obs) :
    if obs in obs_ticks :
        return obs_ticks[obs]
    raise RuntimeWarning(f"WARNING - no obs_ticks found for observable {obs}")
    return []

def get_obs_ticklabels (obs) :
    if obs in obs_ticklabels :
        return obs_ticklabels[obs]
    if obs in obs_ticks :
        return [str(x) for x in obs_ticks[obs]]
    raise RuntimeWarning(f"WARNING - no obs_ticklabels found for observable {obs}")
    return []

def get_transformed_obs_label (obs) :
    if obs in transformed_obs_labels :
        return transformed_obs_labels[obs]
    return obs         

def get_transformed_obs_ticks (obs) :
    if obs in transformed_obs_ticks :
        return transformed_obs_ticks[obs]
    raise RuntimeWarning(f"WARNING - no transformed_obs_ticks found for observable {obs}")
    return []

def get_transformed_obs_ticklabels (obs) :
    if obs in transformed_obs_ticklabels :
        return transformed_obs_ticklabels[obs]
    if obs in transformed_obs_ticks :
        return [str(x) for x in transformed_obs_ticks[obs]]
    raise RuntimeWarning(f"WARNING - no transformed_obs_ticklabels found for observable {obs}")
    return []


def load_table (input_fname, transform=False) :
    """Load a new data_table, select and order the observables as configured, transform and return"""
    #  Create new table
    print(f"Loading events from file {input_fname}")
    data_table = DataTable(input_fname)
    print(f" -- Table created with {data_table.get_num_events()} events")
    #  Cut events which fall outside observable limits
    for observable, limits in observable_limits.items() :
        print(f" -- filtering observable {observable} between {limits[0]} and {limits[1]}")
        data_table.filter(observable, limits[0], limits[1])
        print(f" -- {data_table.get_num_events()} events survived")
    #  Transform observables
    if transform :
        print(" -- transforming observables")
        data_table.data = transform_observables_fwd(data_table.data, data_table.keys)
    #  Remove unwanted columns
    global remove_observables
    for observable in remove_observables :
        print(f" -- removing observable {observable}")
        data_table.remove_column(observable)
    #  Order observables
    print(" -- ordering observables")
    data_table.reorder(*observables)
    #  Print summary
    data_table.print_summary()
    #  Return table
    return data_table


def plot_2D_projections (datapoints, weights=None, label="", savefig="") :
    """plot the 2D projections of the datapoints provided"""
    
    global observables, num_observables, observable_limits, int_observables
    
    #  If no weights provided then assume uniform
    if type(weights) == type(None) :
        weights = np.ones(shape=(datapoints.shape[0],))
    
    #  Get histo bins
    get_bins = {}
    for obs_idx_x, obs_x in enumerate(observables) :
        obs_lims_x = observable_limits[obs_x]
        num_bins_x = 20
        if obs_x in int_observables : get_bins[obs_idx_x] = np.linspace(obs_lims_x[0]-0.5, obs_lims_x[1]+0.5, 2+(obs_lims_x[1]-obs_lims_x[0]))
        else                        : get_bins[obs_idx_x] = np.linspace(obs_lims_x[0]    , obs_lims_x[1]    , num_bins_x+1)

    #  First figure out our colour axis limits
    #     whilst we're at it, save the list of indices to plot (to make sure all loops are over consistent sets)
    vmin = 1e30
    observables_for_x, observables_for_y = [], []
    for obs_idx_x, obs_x in enumerate(observables) :
        #  Don't plot observable -1 on x axis
        if obs_idx_x == num_observables-1 : continue
        observables_for_x.append(obs_idx_x)
        for obs_idx_y, obs_y in enumerate(observables) :
            #  Don't plot observable 0 or above diagonal on y axis
            if obs_idx_y == 0         : continue
            if obs_idx_y <= obs_idx_x : continue
            observables_for_y.append(obs_idx_y)
            #  Get histo limits
            bins_x, bins_y = get_bins[obs_idx_x], get_bins[obs_idx_y]
            vals, _, _ = np.histogram2d(datapoints[:,obs_idx_x], datapoints[:,obs_idx_y], weights=weights, bins=(bins_x, bins_y))
            vals = vals.flatten()
            vals_min, vals_max = np.nanmin([v for v in vals if v > 0]), np.nanmax([v for v in vals if v > 0])
            vmin = np.nanmin([vmin, vals_min/vals_max])

    #  Make plot
    fig = plt.figure(figsize=(20, 14))
    vmin = 1e-5

    for obs_idx_x, obs_x in enumerate(observables) :

        #  Check whether we want to plot observable x
        if obs_idx_x not in observables_for_x : continue

        #  Get axis x coordinates
        xlo    = obs_idx_x / (num_observables-1)
        xwidth = 1.        / (num_observables-1)

        for obs_idx_y, obs_y in enumerate(observables) :

            #  Check whether we want to plot observable y
            if obs_idx_y not in observables_for_y : continue
            if obs_idx_y <= obs_idx_x : continue

            #  Get axis y coordinates
            ylo     = (num_observables-obs_idx_y-1) / (num_observables-1)
            yheight = 1.                            / (num_observables-1)

            #  Create axis
            ax = fig.add_axes([xlo, ylo, 0.95*xwidth, 0.95*yheight])

            #  Draw y-axis label
            if obs_idx_x != 0 : ax.get_yaxis().set_ticklabels([])
            else              : ax.set_ylabel(get_obs_label(obs_y).replace("  [","\n["), fontsize=19, labelpad=20, rotation=0, va="center", ha="right")
            
            #  Draw x-axis label
            if obs_idx_y != num_observables-1 : ax.get_xaxis().set_ticklabels([])
            else                              : ax.set_xlabel(get_obs_label(obs_x).replace("  [","\n"), fontsize=19, labelpad=20, va="top", ha="center")
            
            #  Draw axis ticks
            ax.set_xticks(get_obs_ticks(obs_x))
            ax.set_yticks(get_obs_ticks(obs_y))
            
            #  Draw tick labels
            if obs_idx_y == num_observables-1 : ax.get_xaxis().set_ticklabels(get_obs_ticklabels(obs_x))
            if obs_idx_x == 0                 : ax.get_yaxis().set_ticklabels(get_obs_ticklabels(obs_y))
        
            #  Format tick params
            ax.tick_params(which="both", right=True, top=True, direction="in", labelsize=15)

            #  Draw histogram
            bins_x, bins_y = get_bins[obs_idx_x], get_bins[obs_idx_y]
            _, _, _, patches = ax.hist2d(datapoints[:,obs_idx_x], datapoints[:,obs_idx_y], weights=weights/vals_max, bins=(bins_x, bins_y),
                                      vmin=vmin, vmax=1, norm=colors.LogNorm())
            
            #  Draw label
            if (obs_idx_x==0) and (obs_idx_y==1) and len(label) > 0 :
                ax.text(0, 1.2, label, weight="bold", ha="left", va="bottom", transform=ax.transAxes, fontsize=21)
                

    cbar_ax = fig.add_axes([0.76, 0.5, 0.03, 0.45])
    cbar    = fig.colorbar(patches, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=14)
    cbar   .set_ticks([1, 0.1, 0.01, 0.001, 0.0001, 1e-5])
    cbar   .set_label(r"$\frac{p(x)}{{\rm max}~p(x)}$", fontsize=25, labelpad=50, rotation=0, va="center")

    if len(savefig) > 0 :
        plt.savefig(savefig, bbox_inches="tight")
    plt.show()


def transform_observables_back (data_array, keys) :
    """Transform a 2D array data_array with observables indexed by keys from the transformed basis onto the physical one"""
    return data_array.copy()
    #  Make copy of data_array to modify
    data_array = data_array.copy()
    #  Get indices for observables
    #idx_Dy_j_j     = keys.index("Dy_j_j"    )
    idx_N_jets     = keys.index("N_jets"    )
    idx_N_gap_jets = keys.index("N_gap_jets")
    idx_pT_j1      = keys.index("pT_j1"     )
    idx_pT_j2      = keys.index("pT_j2"     )
    idx_pT_jj      = keys.index("pT_jj"     )
    #idx_rap_jj     = keys.index("rap_jj"    )
    #  Undo N_jets transformation
    N_gap_jets = data_array[:,idx_N_gap_jets].copy()
    data_array[:,idx_N_jets] = data_array[:,idx_N_jets] + N_gap_jets
    #  Undo pT_j1 transformation
    pT_j2 = data_array[:,idx_pT_j2].copy()
    data_array[:,idx_pT_j1 ] = data_array[:,idx_pT_j1] + pT_j2
    #  Undo pT_jj transformation
    pT_j1 = data_array[:,idx_pT_j1].copy()
    data_array[:,idx_pT_jj ] = data_array[:,idx_pT_jj ]  * (pT_j1 + pT_j2)
    #  Undo pT_jj transformation
    #rap_jj = data_array[:,idx_rap_jj].copy()
    #data_array[:,idx_Dy_j_j] = data_array[:,idx_Dy_j_j] * (8.8 - 2*rap_jj)
    #  Return transformed data
    return data_array


def transform_observables_fwd (data_array, keys) :
    """Transform a 2D array data_array with observables indexed by keys from the physical basis onto the transformed one"""
    return data_array.copy()
    #  Ensure all required observables are present
    #for required_key in ["Dy_j_j", "N_jets", "N_gap_jets", "pT_j1", "pT_j2", "pT_jj", "rap_jj"] :
    for required_key in ["N_jets", "N_gap_jets", "pT_j1", "pT_j2", "pT_jj"] :
        assert required_key in keys, f"Required key '{required_key}' not in list provided: " + ", ".join(keys)
    #  Make copy of data_array to modify
    data_array = data_array.copy()
    #  Get indices for observables
    #idx_Dy_j_j     = keys.index("Dy_j_j"    )
    idx_N_jets     = keys.index("N_jets"    )
    idx_N_gap_jets = keys.index("N_gap_jets")
    idx_pT_j1      = keys.index("pT_j1"     )
    idx_pT_j2      = keys.index("pT_j2"     )
    idx_pT_jj      = keys.index("pT_jj"     )
    #idx_rap_jj     = keys.index("rap_jj"    )
    #  Get arrays of observable values
    #Dy_j_j     = data_array[:,idx_Dy_j_j    ].copy()
    N_jets     = data_array[:,idx_N_jets    ].copy()
    N_gap_jets = data_array[:,idx_N_gap_jets].copy()
    pT_j1      = data_array[:,idx_pT_j1     ].copy()
    pT_j2      = data_array[:,idx_pT_j2     ].copy()
    pT_jj      = data_array[:,idx_pT_jj     ].copy()
    #rap_jj     = data_array[:,idx_rap_jj    ].copy()
    #  Set transformed observable values
    data_array[:,idx_N_jets] = N_jets - N_gap_jets          #  Transformed x bound between [0, infty]  (enforces N_jets >= N_gap_jets)
    data_array[:,idx_pT_j1 ] = pT_j1 - pT_j2                #  Transformed x bound between [0, infty]  (enforces pT_j1  >= pT_j2)
    data_array[:,idx_pT_jj ] = pT_jj  / (pT_j1 + pT_j2)     #  Transformed x bound between [0, 1]      (enforces pT_jj  <= pT_j1 + pT_j2)
    #data_array[:,idx_Dy_j_j] = Dy_j_j / (8.8 - 2*rap_jj)    #  Transformed x bound between [0, infty]  (enforces Dy_j_j <= 8.8 - 2*rap_jj)
    #  Return transformed data
    return data_array

