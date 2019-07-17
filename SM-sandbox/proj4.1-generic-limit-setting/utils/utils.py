##  =======================================================================================================================
##  Brief: module of utility functions
##  Author: Stephen Menary
##  Email: sbmenary@gmail.com
##  =======================================================================================================================

import sys
import os
import ast
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


#  ========================================================
#  Globals
#  ========================================================

plot_file = None


#  ========================================================
#  Messaging
#  ========================================================

def print_message (typ, origin, message) :
	print (f"{typ}\t\t{origin}\t\t{message}")

def info (origin, message) :
	print_message("INFO", origin, message)

def error (origin, message) :
	print_message("ERROR", origin, message)

def warning (origin, message) :
	print_message("WARNING", origin, message)

def fatal (origin, message, code=0) :
	print_message("FATAL", origin, message)
	sys.exit(code)


#  ========================================================
#  File Helpers
#  ========================================================

def path_exists (fname) :
	return os.path.exists(fname)

def is_file (fname) :
	return os.path.isfile(fname)

def is_dir (fname) :
	return os.path.isdir(fname)


#  ========================================================
#  Plotting
#  ========================================================


def set_mpl_style (stylename="utils/default_plot_style.mplstyle") :
	try :
		plt.style.use(stylename)
	except OSError :
		error("utils.set_style()", "style {0} could not be found - available styles are: {1}.".format(stylename, ", ".join(plt.style.available)))

def close_plots_pdf () :
	global plot_file
	if type(plot_file) is not PdfPages : return
	plot_file.close()
	plot_file = None

def save_figure (fig) :
	global plot_file
	if type(plot_file) is not PdfPages : return
	fig.savefig(plot_file, format='pdf')

def open_plots_pdf (fname) :
	close_plots_pdf()
	global plot_file
	if type(fname) is str :
		if fname[-4:] != ".pdf" : fname = fname + ".pdf"
		info("utils.open_plots_pdf()", "opening pdf file {0}".format(fname))
		plot_file = PdfPages(fname)
	else :
		raise ValueError("utils.open_plots_pdf()", f"filename of type {type(fname)} provided where string expected")


#  ========================================================
#  File Access
#  ========================================================

def open_from_pickle (fname, specific_keys=None, **kwargs) :
	if path_exists(fname) is False : return False, {}
	unpickled_file = pickle.load(open(fname, "rb"))
	for option, value in kwargs.items() :
		if option not in unpickled_file : return False, {}
		if unpickled_file[option] != value : return False, {}
	if specific_keys == None :
		return True, unpickled_file
	ret = {}
	for key in specific_keys :
		if key not in unpickled_file : return False, {}
		ret[key] = unpickled_file[key]
	return True, ret

def save_to_pickle (fname, dictionary, **kwargs) :
	if path_exists(fname) is True :
		warning("utils.save_to_file()", f"Path {fname} already exists and will be overwritten")
	to_save = {}
	for option, value in dictionary.items() :
		to_save[option] = value
	for option, value in kwargs.items() :
		to_save[option] = value
	pickle.dump(to_save, open(fname,"w+b"))


#  ========================================================
#  General
#  ========================================================

def string_to_object (string) :
	try :
		string = ast.literal_eval(string)
		return string
	except Exception as e :
		error("utils.string_to_object()", "Problem converting string to python object. Printing error:")
		print("========================================\n", e, "\n========================================")
		raise ValueError(f"utils.string_to_object(): could not evaluate entry \"{string}\" as a literal python object")

