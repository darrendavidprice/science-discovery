# ====================================================================================================
#  Brief:  Utility functions
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================


from os import path

import numpy as np


def set_numpy_random_seed (rnd_seed=0) :
	np.random.seed(rnd_seed)


def file_exists (fname) :
	return path.isfile(fname)


def dir_exists (dirname) :
	return path.isdir(dirname)


def path_exists (fname) :
	return path.exists(fname)