##  =======================================================================================================================
##  Brief :  module of plotting functions
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  =======================================================================================================================

import matplotlib.pyplot               as     plt
from   matplotlib.backends.backend_pdf import PdfPages

import utils2.utils.utils                 as utils
import utils2.utils.globals_and_fallbacks as glob


def set_mpl_style (stylename="utils2/utils/default_plot_style.mplstyle") :
	try :
		plt.style.use(stylename)
	except OSError :
		utils.error("utils.set_style()", "style {0} could not be found - available styles are: {1}.".format(stylename, ", ".join(plt.style.available)))

def close_plots_pdf () :
	if type(glob.plot_file) is not PdfPages : return
	glob.plot_file.close()
	glob.plot_file = None

def save_figure (fig) :
	if type(glob.plot_file) is not PdfPages : return
	fig.savefig(glob.plot_file, format='pdf')

def open_plots_pdf (fname) :
	close_plots_pdf()
	if type(fname) is str :
		if fname[-4:] != ".pdf" : fname = fname + ".pdf"
		utils.info("utils.open_plots_pdf()", f"opening pdf file {fname}")
		glob.plot_file = PdfPages(fname)
	else :
		raise ValueError("utils.open_plots_pdf()", f"filename of type {type(fname)} provided where string expected")

