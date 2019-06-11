# ====================================================================================================
#  Brief:  Plotting helper functions
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.stats as stats

import Confidence.messaging as msg


#  Dlobal .pdf document to store all plots
document = None


#  Close the save file
def close_save_file () :
	global document
	if type(document) is not PdfPages : return
	document.close()


#  Set the save file (plots created using save=True will be saved here)
def set_save_file ( fname_ ) :
	close_save_file()
	global document
	if type(fname_) is str :
		if fname_[-4:] != ".pdf" : fname_ = fname_ + ".pdf"
		msg.info("Plotting.plotting.set_save_file", "Opening pdf file {0}".format(fname_), verbose_level=0)
		document = PdfPages(fname_)
	else :
		msg.error("Plotting.plotting.set_save_file", "Filename must be a str")


#  Set custom plot style
#
def set_style (stylename="Confidence/Plotting/my_style.mplstyle") :
	try : plt.style.use(stylename)
	except OSError :
		msg.error("Plotting.plotting.set_style","Style {0} could not be found. Available styles are {1}.".format(stylename,plt.style.available))


#  Turn bin edges into the x-axis of a histogram-like graph
#
def bin_edges_to_histogram_x (bins) :
	x_new = [ bins[0], bins[0] ]
	for i in range(len(bins)-2) :
		x_new.append(bins[1+i])
		x_new.append(x_new[-1])
	x_new.append (bins[-1])
	x_new.append (bins[-1])
	return x_new


#  Turn bin contents into the y-axis of a histogram-like graph
#
def bin_contents_to_histogram_y (values) :
	y_new = [0.]
	for i in range(len(values)) :
		y_new.append(values[i])
		y_new.append(y_new[-1])
	y_new.append(0.)
	return y_new


#  Turn bin edges and contents from TH1-like format into the y-axis of a histogram-like graph
#
def bin_edges_and_contents_to_histogram (bins, values) :
	return bin_edges_to_histogram_x(bins), bin_contents_to_histogram_y(values)


#  Create histogram from raw data
#
def create_histogram (n_bins, raw_data, **kwargs) :
	x_min, x_max = min(raw_data), max(raw_data)
	pad_x = kwargs.get("pad_x",0.)
	if "xmin" in kwargs : x_min = kwargs["xmin"]
	if "xmax" in kwargs : x_max = kwargs["xmax"]
	x_range = x_max - x_min
	x_min, x_max = x_min - pad_x*x_range, x_max + pad_x*x_range
	x_range = x_max - x_min
	bin_width = x_range / n_bins
	bins = [x_min + i*bin_width for i in range(1+n_bins)]
	y = np.zeros(shape=(n_bins))
	for datum in raw_data :
		for i in range(n_bins) :
			if datum < bins[i] : continue
			if datum > bins[i+1] : continue
			y[i] = y[i] + 1
	x_new = bin_edges_to_histogram_x(bins)
	y_new = bin_contents_to_histogram_y(y)
	return x_new, y_new


#  Simple plot of histogrammed data
#
def plot_histo_from_data (n_bins, raw_data, **kwargs) :
	fig  = plt.figure(figsize=(7,7))
	ax = fig.add_subplot(111)
	labels = kwargs.get("labels",[])
	if len(labels) != len(raw_data) : labels = ["unknown" for i in range(len(raw_data))]
	for datum, label in zip(raw_data, labels) :
		x_new, y_new = create_histogram(n_bins, datum, **kwargs)
		ax.plot(x_new, y_new, marker=None, linestyle='-', linewidth=2, label=label)
		ax.set_xlim(x_new[0],x_new[-1])
		if kwargs.get("overlay_gauss",False) is True :
			N = (x_new[-1]-x_new[0]) * float(len(datum)) / float(n_bins)
			ax.plot(x_new, N*stats.norm.pdf(x_new), '-', c='k', linewidth=2, label="Gauss(0,1)")
	if kwargs.get("logx",False) is True : ax.set_xscale("log")
	if kwargs.get("logy",False) is True : ax.set_yscale("log")
	if "xlabel" in kwargs : plt.xlabel(kwargs["xlabel"])
	if "ylabel" in kwargs : plt.ylabel(kwargs["ylabel"])
	ymin, ymax = ax.get_ylim()
	ymin, ymax = kwargs.get("ymin",ymin), kwargs.get("ymax",ymax)
	ax.set_ylim(ymin, ymax)
	plt.legend(loc="best")
	plt.show()
	if type(document) is PdfPages :
		fig.savefig(document, format='pdf')
	plt.close(fig)