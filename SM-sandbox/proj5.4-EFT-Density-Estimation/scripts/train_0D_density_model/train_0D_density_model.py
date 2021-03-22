#======================================#
#   Brief:  Train a likelihood model   #
#   Author: stmenary@cern.ch           #
#======================================#


#=======================#
#  1. Required imports  #
#=======================#

print("Importing standard library")
import os, sys, time

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
from backends.utils            import make_sure_dir_exists_for_filename, joint_shuffle

from backends import plot as plot, density_model as density_model, VBFZ_analysis as VBFZ
