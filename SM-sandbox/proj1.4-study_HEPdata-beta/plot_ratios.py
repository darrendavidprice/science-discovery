# ====================================================================================================
#  Brief: open HEPdata or ROOT file(s) and plot the ratios between matching tables using the given
#           --num and --den tags to label numerators and denominators
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================


import sys, os, getopt, shutil
import numpy as np
import HEP_data_utils.messaging as msg
import HEP_data_utils.plotting as plotter
import HEP_data_utils.general_helpers as hlp
import HEP_data_utils.HEP_data_helpers as HD
import HEP_data_utils.ROOT_helpers as RT
from HEP_data_utils.DistributionContainer import DistributionContainer


#  Brief: print help
def print_help () :
	msg.info("plot_ratios.py","Usage: python3 plot_ratios.py --num <numerator tag> --den <denominator tag> <input-files>")
	msg.info("plot_ratios.py","If input files/directory contains a submission.yaml file, all other inputs will be ignored")
	msg.info("plot_ratios.py","I assume that you follow the format instructions provided at")
	msg.info("plot_ratios.py","    https://hepdata-submission.readthedocs.io/en/latest/introduction.html")
	msg.info("plot_ratios.py","Optional arguments are:")
	msg.info("plot_ratios.py","     -h, --help\tPrint this help message and close")
	msg.info("plot_ratios.py","     -v, --verbosity\tSet VERBOSE_LEVEL {-1, 0, 1, 2} (-1 by default)")
	msg.info("plot_ratios.py","     -r, --recursive\tAllow recursive searching of directories")
	msg.info("plot_ratios.py","                    \tRecursion stops if submission.yaml file is found")
	msg.info("plot_ratios.py","     -s, --save\t\tSave plots to the file provided")
	msg.info("plot_ratios.py","     --show\t\tShow plots to the screen")
	msg.info("plot_ratios.py","N.B. you can validate your yaml file format using the package:")
	msg.info("plot_ratios.py","    https://github.com/HEPData/hepdata-validator")


#  Brief: parse command line arguments and check for errors
def parse_inputs ( argv_ ) :
	#  Get arguments
	try :
		opts, rest = getopt.getopt(argv_,"hrv:s:",["help","recursive","show","save=","verbosity=","num=","den="])
	except getopt.GetoptError as err :
		msg.error("plot_ratios.py","The following error was thrown whilst parsing command-line arguments")
		print(">>>>>>>>\n",err,"\n<<<<<<<<")
		msg.error("plot_ratios.py","Falling back to to --help...")
		print_help()
		msg.fatal("plot_ratios.py","Command-line arguments not recognised.")
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
			msg.info("plot_ratios.py","Config: using recursion if needed",verbose_level=0)
			do_recurse = True
		if opt in ["--num",] :
			num_tag = str(arg)
			msg.info("plot_ratios.py","Config: numerators will be identified using the tag {0}".format(num_tag),verbose_level=0)
		if opt in ["--den",] :
			den_tag = str(arg)
			msg.info("plot_ratios.py","Config: denominators will be identified using the tag {0}".format(den_tag),verbose_level=0)
		if opt in ["--show"] :
			msg.info("plot_contents_of_yaml.py","Config: showing all distributions found",verbose_level=0)
			do_show = True
		if opt in ['-s',"--save"] :
			save_file = str(arg)
			if save_file[-4:] != ".pdf" : save_file = save_file + ".pdf"
			msg.info("plot_contents_of_yaml.py","Config: saving plots to {0}".format(save_file),verbose_level=0)
		if opt in ['-v',"--verbosity"] :
			msg.info("plot_ratios.py","Config: setting verbosity to {0}".format(arg),verbose_level=0)
			try : msg.VERBOSE_LEVEL = int(arg)
			except : msg.fatal("plot_ratios.py","Could not cast verbosity level {0} to integer".format(arg))
	yaml_files = hlp.keep_only_yaml_files(argv_,recurse=do_recurse)
	root_files = hlp.keep_only_root_files(argv_,recurse=do_recurse)
	if num_tag is None :
		num_tag = "measured"
		msg.warning("plot_ratios.py","No --num provided, falling back to \"{0}\"".format(num_tag))
	if den_tag is None :
		den_tag = "expected"
		msg.warning("plot_ratios.py","No --den provided, falling back to \"{0}\"".format(den_tag))
	#  Return
	return num_tag, den_tag, do_show, save_file, yaml_files, root_files


#  Brief: plot ratio of 1D distributions from HEPDataTables
def plot_ratio_1D ( table_num_ , table_den_ , **kwargs ) :
	x_n, y_n, [ex_lo_n,ex_hi_n], [ey_lo_n,ey_hi_n], labels, keys_num = plotter.get_1D_distribution(table_num_)
	x_d, y_d, [ex_lo_d,ex_hi_d], [ey_lo_d,ey_hi_d], labels, keys_den = plotter.get_1D_distribution(table_den_)
	chi2 = hlp.get_chi2(y_n,ey_lo_n,ey_hi_n,y_d,ey_lo_d,ey_hi_d)
	for i in range(len(x_n)) :
		if x_n[i] == x_d[i] : continue
		msg.error("plot_ratio_1D","Arguments do not have the same binning")
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
	ymin, ymax = ax1.get_ylim()
	ax1.axis(xmin=xlim[0],xmax=xlim[1])
	try :
		plotter.plt.text( xlim[0] , 1.19*ymax - 0.19*ymin , ("$\\bf{TABLE}$ $\\bf{1:}$  "+table_num_._dep_var._name).replace("\\\\","\\").replace(r"\text{",r"{\rm ") )
		plotter.plt.text( xlim[0] , 1.08*ymax - 0.08*ymin , ("$\\bf{TABLE}$ $\\bf{2:}$  "+table_den_._dep_var._name).replace("\\\\","\\").replace(r"\text{",r"{\rm ") )
	except : msg.warning("plot_ratio_1D","could not render observable name - no title given to plot")
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


#  Brief: make plot of 2D values using inputs in value and label format
def make_2D_plot_from_values ( ax , v , x_label, y_label , **kwargs ) :
	try : plotter.plt.xlabel(x_label)
	except Exception : pass
	try : plotter.plt.ylabel(y_label)
	except Exception : pass
	y_central_value = kwargs.get("y_central_value",0.)
	max_val = max([np.fabs(val-y_central_value) for val in v.flatten()])
	vmin, vmax = y_central_value - max_val, y_central_value + max_val
	ax.imshow(v.transpose(),cmap="bwr",vmin=vmin,vmax=vmax)
	precision = kwargs.get("flt_precision",2)
	labels_x, labels_y = kwargs.get("labels_x",[]), kwargs.get("labels_y",[])
	if (len(labels_x),len(labels_y)) == v.shape :
		for i in range(len(labels_x)) :
			for j in range(len(labels_y)) :
				ax.text(i, j, "{0:.{1}f}".format(v[i, j],precision), ha="center", va="center", color="k",fontsize="small")
	ax.set_xticks(np.arange(len(labels_x)))
	ax.set_xticklabels(labels_x,rotation=90)
	ax.set_yticks(np.arange(len(labels_y)))
	ax.set_yticklabels(labels_y)
	plotter.plt.title(kwargs.get("title",""))



#  Brief: plot ratio of 2D distributions from HEPDataTables
def plot_ratio_2D ( table_num_ , table_den_ , **kwargs ) :
	labels_x, labels_y, values_den = plotter.get_2D_distribution(table_den_)
	labels_x, labels_y, values_num = plotter.get_2D_distribution(table_num_)
	values = np.divide(values_num,values_den)
	indep_var = table_num_._indep_vars
	fig = plotter.plt.figure(figsize=(20,40))
	#		ADD PLOT OF TABLE 1
	ax = fig.add_subplot(421)
	plotter.plot_2D_distribution_on_current_axes ( table_num_ , fontsize="small" , **kwargs )
	try : plotter.plt.title(("$\\bf{TABLE}$ $\\bf{1:}$  "+table_num_._dep_var._name).replace("\\\\","\\").replace(r"\text{",r"{\rm "))
	except : msg.warning("plot_ratio_2D","could not render observable name for TABLE 1 - no title given to plot")
	#		ADD PLOT OF TABLE 2
	ax = fig.add_subplot(422)
	plotter.plot_2D_distribution_on_current_axes ( table_den_ , fontsize="small" , **kwargs )
	try : plotter.plt.title(("$\\bf{TABLE}$ $\\bf{1:}$  "+table_den_._dep_var._name).replace("\\\\","\\").replace(r"\text{",r"{\rm "))
	except : msg.warning("plot_ratio_2D","could not render observable name for TABLE 1 - no title given to plot")
	#		ADD PLOT OF RATIO
	x_label, y_label = "", ""
	try : x_label = kwargs.get("x_label",indep_var[0].name().replace("\\\\","\\").replace(r"\text{",r"{\rm "))
	except Exception : pass
	try : y_label = kwargs.get("y_label",indep_var[1].name().replace("\\\\","\\").replace(r"\text{",r"{\rm "))
	except Exception : pass
	ax = fig.add_subplot(423)
	make_2D_plot_from_values ( ax , values , x_label, y_label , title="$\\bf{TABLE 1}$  /  $\\bf{TABLE 2}$" , y_central_value=1., labels_x=labels_x, labels_y=labels_y, **kwargs )
	#		ADD PLOTS OF TABLE 1 RELATIVE UNCERTAINTIES
	[ey_lo_n, ey_hi_n], err_keys_n = plotter.get_table_errors(table_num_)
	rel_err_hi_n, rel_err_lo_n = np.divide(ey_hi_n,values_num), np.divide(ey_lo_n,values_num)
	ax = fig.add_subplot(425)
	make_2D_plot_from_values ( ax , rel_err_hi_n , x_label, y_label , title="TABLE 1 relative uncertainty (up, combined sources)"   , y_central_value=0., labels_x=labels_x, labels_y=labels_y, **kwargs )
	ax = fig.add_subplot(426)
	make_2D_plot_from_values ( ax , rel_err_lo_n , x_label, y_label , title="TABLE 1 relative uncertainty (down, combined sources)" , y_central_value=0., labels_x=labels_x, labels_y=labels_y, **kwargs )
	#		ADD PLOTS OF TABLE 2 RELATIVE UNCERTAINTIES
	[ey_lo_d, ey_hi_d], err_keys_d = plotter.get_table_errors(table_den_)
	rel_err_hi_d, rel_err_lo_d = np.divide(ey_hi_d,values_den), np.divide(ey_lo_d,values_den)
	ax = fig.add_subplot(427)
	make_2D_plot_from_values ( ax , rel_err_hi_d , x_label, y_label , title="TABLE 2 relative uncertainty (up, combined sources)"   , y_central_value=0., labels_x=labels_x, labels_y=labels_y, **kwargs )
	ax = fig.add_subplot(428)
	make_2D_plot_from_values ( ax , rel_err_lo_d , x_label, y_label , title="TABLE 2 relative uncertainty (down, combined sources)" , y_central_value=0., labels_x=labels_x, labels_y=labels_y, **kwargs )
	#		Finish up
	if kwargs.get("show",False) :
		plotter.plt.show()
	if kwargs.get("save",False) :
		fig.savefig ( plotter.document , format='pdf' )
	plotter.plt.close(fig)
	#		Return chi2 and Table1/Table2 values
	chi2 = hlp.get_chi2(values_num,ey_lo_n,ey_hi_n,values_den,ey_lo_d,ey_hi_d)
	return chi2, values.flatten()


#  Brief: plot ratio of two tables
def plot_ratio ( table_num_ , table_den_ , **kwargs ) :
	n_dim = table_num_.n_indep_vars()
	if n_dim == 1 : return plot_ratio_1D(num_dist,den_dist,**kwargs)
	if n_dim == 2 : return plot_ratio_2D(num_dist,den_dist,**kwargs)
	raise TypeError("Cannot plot the ratio between tables with {0} independent variables".format(n_dim))


#  Brief: print a nice output describing the ratio between the two tables, and 
def plot_and_print ( num_dist , den_dist , **kwargs ) :
	if not HD.has_matching_bins ( num_dist , den_dist ) : return
	n_dim = num_dist.n_indep_vars()
	print("=====================================================================================")
	print("===    PLOTTING THE FOLLOWING RATIO  ( 1 divided by 2 )")
	print("===    1.  ",num_dist._dep_var._name)
	print("===    2.  ",den_dist._dep_var._name)
	if n_dim == 1 :
		print("===    as a function of:  ",num_dist._indep_vars[0]._name)
	if n_dim == 2 :
		print("===    as a function of:  ",num_dist._indep_vars[0]._name)
		print("===                       ",num_dist._indep_vars[1]._name)
	try :
		chi2, bin_ratios = plot_ratio(num_dist,den_dist,**kwargs)
		print("===    chi2 = {:.4f}".format(chi2))
		print("===    bin ratios = ",["{:.4f}".format(x) for x in bin_ratios])
	except Exception as e :
		print(e)
		msg.error("plot_ratios","Error when plotting ratio... skipping")
	print("=====================================================================================")


#  =================================== #
#  ====    Brief: main program    ==== #
#  =================================== #
if __name__ == "__main__" :
				#
				#  Welcome
				#
	msg.info("plot_ratios.py","Running program")
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
				#  Get numerator and denominator tables
				#
	num_dists, den_dists = [], []
	for d in [ my_tables._inclusive_distributions , my_tables._1D_distributions , my_tables._2D_distributions , my_tables._ND_distributions ] : 
		for key, dist in d.items() :
			if num_tag in key or num_tag in dist._dep_var._name : num_dists.append(dist)
			if den_tag in key or den_tag in dist._dep_var._name : den_dists.append(dist)
				#
				#  Look at ratios when numerator/denominator table natch found
				#
	for num_dist in num_dists :
		for den_dist in den_dists :
			plot_and_print ( num_dist , den_dist , show=do_show , save=do_save )
				#
				#  Close save file
				#
	if do_save : plotter.close_save_file()
				#
				#  Goodbye
				#
	msg.info("plot_ratios.py","Program reached the end without crashing and will close :) Have a nice day...")
				#
				#
