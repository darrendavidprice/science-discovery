# ====================================================================================================
#  Brief: open HEPdata or ROOT file(s) and plot the ratios between matching tables using the given
#           --num and --den tags to label numerators and denominators
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================


import sys, os, getopt, shutil, math
import numpy as np
import HEP_data_utils.messaging as msg
import HEP_data_utils.plotting as plotter
import HEP_data_utils.general_helpers as hlp
import HEP_data_utils.HEP_data_helpers as HD
import HEP_data_utils.ROOT_helpers as RT
from HEP_data_utils.DistributionContainer import DistributionContainer


#  Brief: print help
def print_help () :
	msg.info("plot_ratio_between_distributions.py:print_help","Usage is: python3 plot_ratio_between_distributions.py --num <tag to label numerators> --den <tag to label denominators> <input-files>")
	msg.info("plot_ratio_between_distributions.py:print_help","If input files/directory contains a submission.yaml file, all other inputs will be ignored")
	msg.info("plot_ratio_between_distributions.py:print_help","I assume that you follow the format instructions provided at https://hepdata-submission.readthedocs.io/en/latest/introduction.html")
	msg.info("plot_ratio_between_distributions.py:print_help","Optional arguments are:")
	msg.info("plot_ratio_between_distributions.py:print_help","       -h, --help\t\tPrint this help message and close")
	msg.info("plot_ratio_between_distributions.py:print_help","       -v, --verbosity\t\tSet HEP_data_utils.messaging.VERBOSE_LEVEL as {-1, 0, 1, 2} (-1 by default)")
	msg.info("plot_ratio_between_distributions.py:print_help","       -r, --recursive\t\tAllow recursive searching of directories. Recursion stops if submission.yaml file is found")
	msg.info("plot_ratio_between_distributions.py:print_help","       -s, --save\t\tSave plots to the file provided")
	msg.info("plot_ratio_between_distributions.py:print_help","       --show\t\t\tShow plots to the screen")
	msg.info("plot_ratio_between_distributions.py:print_help","N.B. you can validate your yaml files using the following package: https://github.com/HEPData/hepdata-validator")


#  Brief: parse command line arguments and check for errors
def parse_inputs ( argv_ ) :
	#  Get arguments
	try :
		opts, rest = getopt.getopt(argv_,"hrv:s:",["help","recursive","show","save=","verbosity=","num=","den="])
	except getopt.GetoptError as err :
		msg.error("plot_ratio_between_distributions.py","The following error was thrown whilst parsing command-line arguments")
		print(">>>>>>>>\n",err,"\n<<<<<<<<")
		msg.error("plot_ratio_between_distributions.py","Falling back to to --help...")
		print_help()
		msg.fatal("plot_ratio_between_distributions.py","Command-line arguments not recognised.")
	#  Parse arguments
	do_recurse = False
	do_show = False
	save_file = ""
	num_tag, den_tag = None, None
	for opt, arg in opts:
		if opt in ['-h',"--help"] :
			print_help()
			sys.exit(0)
		if opt in ['-r',"--recursive",] :
			msg.info("plot_ratio_between_distributions.py","Config: using recursion if needed",verbose_level=0)
			do_recurse = True
		if opt in ["--num",] :
			num_tag = str(arg)
			msg.info("plot_ratio_between_distributions.py","Config: numerators will be identified using the tag {0}".format(num_tag),verbose_level=0)
		if opt in ["--den",] :
			den_tag = str(arg)
			msg.info("plot_ratio_between_distributions.py","Config: denominators will be identified using the tag {0}".format(den_tag),verbose_level=0)
		if opt in ["--show"] :
			msg.info("plot_contents_of_yaml.py","Config: showing all distributions found",verbose_level=0)
			do_show = True
		if opt in ['-s',"--save"] :
			save_file = str(arg)
			if save_file[-4:] != ".pdf" : save_file = save_file + ".pdf"
			msg.info("plot_contents_of_yaml.py","Config: saving plots to {0}".format(save_file),verbose_level=0)
		if opt in ['-v',"--verbosity"] :
			msg.info("plot_ratio_between_distributions.py","Config: setting verbosity to {0}".format(arg),verbose_level=0)
			try : msg.VERBOSE_LEVEL = int(arg)
			except : msg.fatal("plot_ratio_between_distributions.py","Could not cast verbosity level {0} to integer".format(arg))
	yaml_files = hlp.keep_only_yaml_files(argv_,recurse=do_recurse)
	root_files = hlp.keep_only_root_files(argv_,recurse=do_recurse)
	if num_tag is None :
		num_tag = "measured"
		msg.warning("plot_ratio_between_distributions.py","No --num provided, falling back to \"{0}\"".format(num_tag))
	if den_tag is None :
		den_tag = "expected"
		msg.warning("plot_ratio_between_distributions.py","No --den provided, falling back to \"{0}\"".format(den_tag))
	#  Return
	return num_tag, den_tag, do_show, save_file, yaml_files, root_files


# Brief: return the chi^2 between two distributions, using the uncertainty amplitudes in the inside direction
#        - assumes that the x-axes are identical
#        - assumes that bin-to-bin correlation is zero
def get_chi2 ( y1 , ey1_lo , ey1_hi , y2 , ey2_lo , ey2_hi ) :
	if len(y1) != len(y2) : return None
	if len(ey1_lo) != len(y1) or len(ey1_hi) != len(y1) : return None
	if len(ey2_lo) != len(y2) or len(ey2_hi) != len(y2) : return None
	chi2 = 0.
	for i in range(len(y1)) :
		res = y2[i] - y1[i]
		if res > 0 : err2 = ey1_hi[i]*ey1_hi[i] + ey2_lo[i]*ey2_lo[i]
		else : err2 = ey1_lo[i]*ey1_lo[i] + ey2_hi[i]*ey2_hi[i]
		chi2 = chi2 + res*res/err2
	return chi2



#  Brief: plot ratio of 1D distributions from HEPDataTables
def plot_ratio ( table_num_ , table_den_ , **kwargs ) :
	x_n, y_n, [ex_lo_n,ex_hi_n], [ey_lo_n,ey_hi_n], labels, keys_num = plotter.get_1D_distribution(table_num_)
	x_d, y_d, [ex_lo_d,ex_hi_d], [ey_lo_d,ey_hi_d], labels, keys_den = plotter.get_1D_distribution(table_den_)
	chi2 = get_chi2(y_n,ey_lo_n,ey_hi_n,y_d,ey_lo_d,ey_hi_d)
	for i in range(len(x_n)) :
		if x_n[i] == x_d[i] : continue
		msg.error("plot_ratio","Arguments do not have the same binning")
		raise ValueError("Ratio of distributions with bin centres at {0} and {1}",x_n,x_d) 
	fig = plotter.plt.figure(figsize=(10,10))
	ax1 = fig.add_subplot(211)
	legend_char_width = 53
	str_num_legend = "TABLE 1 ( " + " + ".join(keys_num) + " )"
	str_num_legend = "\n".join([str_num_legend[legend_char_width*i:min(len(str_num_legend),legend_char_width*(i+1))] for i in range(int(len(str_num_legend)/legend_char_width)+1)])
	str_den_legend = "TABLE 2 ( " + " + ".join(keys_den) + " )"
	str_den_legend = "\n".join([str_den_legend[legend_char_width*i:min(len(str_den_legend),legend_char_width*(i+1))] for i in range(int(len(str_den_legend)/legend_char_width)+1)])
	ax1.errorbar(x_n, y_n, yerr=[ey_lo_n,ey_hi_n], xerr=[ex_lo_n,ex_hi_n], c='k', linewidth=2, linestyle='None', alpha=0.8, label=str_num_legend)
	ax1.errorbar(x_d, y_d, yerr=[ey_lo_d,ey_hi_d], xerr=[ex_lo_d,ex_hi_d], c='r', linewidth=4, linestyle='None', alpha=0.4, label=str_den_legend)
	ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plotter.plt.subplots_adjust(left=0.1, right=0.5, top=0.95, bottom=0.4)
	plotter.plt.ylabel("Values")
	plotter.plt.ylabel("Values")
	xlim = kwargs.get("xlim",[x_d[0]-np.fabs(ex_lo_n[0]),x_d[-1]+np.fabs(ex_hi_n[-1])])
	ymin, ymax = plotter.plt.gca().get_ylim()
	ax1.axis(xmin=xlim[0],xmax=xlim[1])
	try :
		plotter.plt.text( xlim[0] , 1.19*ymax - 0.19*ymin , ("$\\bf{TABLE}$ $\\bf{1:}$  "+table_num_._dep_var._name).replace("\\\\","\\").replace(r"\text{",r"{\rm ") )
		plotter.plt.text( xlim[0] , 1.08*ymax - 0.08*ymin , ("$\\bf{TABLE}$ $\\bf{2:}$  "+table_den_._dep_var._name).replace("\\\\","\\").replace(r"\text{",r"{\rm ") )
	except : msg.warning("plot_ratio","could not render observable name - no title given to plot")
	plotter.plt.grid()
	ax2 = fig.add_subplot(212)
	ax2.errorbar(x_n, y_n/y_d, yerr=[ey_lo_n/y_d,ey_hi_n/y_d], xerr=[ex_lo_n,ex_hi_n], c='k', linewidth=2, linestyle='None', alpha=0.8)
	ax2.errorbar(x_d, y_d/y_d, yerr=[ey_lo_d/y_d,ey_hi_d/y_d], xerr=[ex_lo_d,ex_hi_d], c='r', linewidth=4, linestyle='None', alpha=0.4)
	box = ax2.get_position()
	ax2.set_position([box.x0, box.y0, box.width*0.4, box.height])
	ax2.axis(xmin=xlim[0],xmax=xlim[1])
	plotter.plt.ylabel("Ratio  $\\bf{vs.}$  TABLE 2")
	try : plotter.plt.xlabel(kwargs.get("xlabel",table_den_._indep_vars[0].name().replace("\\\\","\\").replace(r"\text{",r"{\rm ")))
	except : plotter.plt.xlabel("<error reading xlabel")
	plotter.plt.subplots_adjust(left=0.1, right=0.5, top=0.92, bottom=0.4)
	plotter.plt.grid()
	if kwargs.get("show",False) :
		plotter.plt.show()
	if kwargs.get("save",False) :
		fig.savefig ( plotter.document , format='pdf' )
		plotter.plt.close(fig)
	return chi2, y_n/y_d


#  Brief: return True if two arrays look very similar within some margin of error
def do_arrays_look_similar ( array1 , array2 , **kwargs ) :
	if array1.shape != array2.shape : return False
	value_margin = kwargs.get("value_margin",1e-4)
	zero_margin = kwargs.get("zero_margin",None)
	array1, array2 = array1.flatten(), array2.flatten()
	ratio = np.zeros(shape=(len(array1)))
	for i in range(len(ratio)) :
		if array2[i] == 0 :
			if array1[i] == 0 : ratio[i] = 1
			else : ratio[i] = 0
			continue
		ratio[i] = array1[i] / array2[i]
	is_similar = True
	max1, max2 = max(np.fabs(array1.astype(np.float32))), max(np.fabs(array2.astype(np.float32)))
	for idx in range(len(ratio)) :
		if ratio[idx] > 1 - value_margin and ratio[idx] < 1 + value_margin : continue
		if zero_margin is not None and math.fabs(array1[idx]) / max1 < zero_margin : continue     #  Don't count zero entries, defined as less than zero_margin of max
		if zero_margin is not None and math.fabs(array2[idx]) / max2 < zero_margin : continue     #  Don't count zero entries, defined as less than zero_margin of max
		is_similar = False
	return is_similar


#  Brief: return e.g. [True,True,False] if the binnings are similar
def do_bins_look_similar ( table1 , table2 , **kwargs ) :
	if table1.n_indep_vars() != table2.n_indep_vars() : return [False]
	bins_match = []
	for var_idx1 in range(table1.n_indep_vars()) :
		match = False
		indep_var_1 = table1._indep_vars[var_idx1]
		for var_idx2 in range(table2.n_indep_vars()) :
			indep_var_2 = table2._indep_vars[var_idx2]
			if not do_arrays_look_similar(indep_var_1._bin_centers,indep_var_2._bin_centers) : continue
			if not do_arrays_look_similar(indep_var_1._bin_widths_lo,indep_var_2._bin_widths_lo,value_margin=1e-3,zero_margin=1e-5) : continue
			if not do_arrays_look_similar(indep_var_1._bin_widths_hi,indep_var_2._bin_widths_hi,value_margin=1e-3,zero_margin=1e-5) : continue
			match = True
		bins_match.append(match)
	return bins_match


def has_matching_bins ( table1 , table2 ) :
	if table1.n_indep_vars() != table2.n_indep_vars() : return False
	if len(table1._dep_var) != len(table2._dep_var) : return False
	if False in do_bins_look_similar(table1,table2) : return False
	return True


#  =================================== #
#  ====    Brief: main program    ==== #
#  =================================== #
if __name__ == "__main__" :
				#
				#  Welcome
				#
	msg.info("plot_ratio_between_distributions.py","Running program")
				#
				#  Get input files and settings
				#
	num_tag, den_tag, do_show, save_file, yamls_to_load, roots_to_load = parse_inputs(sys.argv[1:])
	do_save = len(save_file) > 0
	if do_save : plotter.set_save_file(save_file)
				#
				#  Load input files
				#
	my_tables = DistributionContainer("my_tables")
	HD.load_yaml_files_from_list(my_tables,yamls_to_load)
	RT.load_root_files_from_list(my_tables,roots_to_load)
				#
				#  Get numerator and denominator distributions
				#
	num_dists, den_dists = [], []
	for key, dist in my_tables._1D_distributions.items() :
		if num_tag in key or num_tag in dist._dep_var._name :
			num_dists.append(dist)
		if den_tag in key or den_tag in dist._dep_var._name :
			den_dists.append(dist)
				#
				#  Get numerator and denominator distributions
				#
	for num_dist in num_dists :
		for den_dist in den_dists :
			if not has_matching_bins ( num_dist , den_dist ) : continue
			print("=====================================================================================")
			print("===    PLOTTING THE FOLLOWING RATIO  ( 1 divided by 2 )")
			print("===    1.  ",num_dist._dep_var._name)
			print("===    2.  ",den_dist._dep_var._name)
			try :
				chi2, bin_ratios = plot_ratio(num_dist,den_dist,show=do_show,save=do_save)
				print("===    chi2 = {:.4f}".format(chi2))
				print("===    bin ratios = ",["{:.4f}".format(x) for x in bin_ratios])
			except Exception as e :
				print(e)
				msg.error("HEP_data_utils.data_structures.DistributionContainer.plot_ratio","Error when plotting ratio... skipping")
			print("=====================================================================================")
				#
				#  Close save file
				#
	if do_save : plotter.close_save_file()
				#
				#  Goodbye
				#
	msg.info("plot_ratio_between_distributions.py","Program reached the end without crashing and will close :) Have a nice day...")
				#
				#
