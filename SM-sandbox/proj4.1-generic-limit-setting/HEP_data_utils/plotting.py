# ====================================================================================================
#  Brief: functions for plotting the contents of HEP_data_utils.data_structures.SubmissionFileTable
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted, ns

import HEP_data_utils.messaging as msg


#  Brief: global pdf to store all plots
document = None


#  Brief: set the save file (plots created using save=True will be saved here)
def set_save_file ( fname_ ) :
	global document
	if type(fname_) is str :
		if type(document) is PdfPages : document.close()
		if fname_[-4:] != ".pdf" : fname_ = fname_ + ".pdf"
		msg.info("HEP_data_utils.plotting.set_save_file","Opening pdf file {0}".format(fname_),verbose_level=0)
		document = PdfPages(fname_)
	else : msg.error("HEP_data_utils.plotting.set_save_file","Filename must be a str")


#  Brief: close the save file
def close_save_file () :
	global document
	if type(document) is not PdfPages : return
	document.close()


#  Brief: close the save file
def get_table_errors ( table_ , errs_ = [] ) :
	if type(errs_) is str : errs_ = [errs_]
	if type(errs_) is None : errs_ = []
	keys = []
	dep_var = table_._dep_var
	original_shape = dep_var._values.shape
	y = dep_var._values.flatten()
	ey_lo, ey_hi = np.zeros(shape=y.shape), np.zeros(shape=y.shape)
	for key in dep_var._symerrors :
		if len(errs_) > 0 and key not in errs_ : continue
		errs = dep_var._symerrors[key].flatten()
		keys.append(key)
		for i in range(0,len(errs)) :
			ey_lo[i] = ey_lo[i] + errs[i]*errs[i]
			ey_hi[i] = ey_hi[i] + errs[i]*errs[i]
	for key in dep_var._asymerrors_up :
		if len(errs_) > 0 and key not in errs_ : continue
		errs1, errs2 = dep_var._asymerrors_up[key].flatten(), dep_var._asymerrors_dn[key].flatten()
		keys.append(key)
		for i in range(0,len(errs1)) :
			err1 = errs1[i]
			err2 = errs2[i]
			if err1 > 0 : ey_hi[i] = ey_hi[i] + err1*err1
			else : ey_lo[i] = ey_lo[i] + err1*err1
			if err2 > 0 : ey_hi[i] = ey_hi[i] + err2*err2
			else : ey_lo[i] = ey_lo[i] + err2*err2
	ey_lo, ey_hi = np.sqrt(ey_lo), np.sqrt(ey_hi)
	return [ey_lo.reshape(original_shape),ey_hi.reshape(original_shape)], keys


#  Brief: open a 1D distribution and turn it into plottable data
def get_1D_distribution ( table_ , errs_ = [] ) :
	if type(errs_) is str : errs_ = [errs_]
	dep_var = table_._dep_var
	indep_var = table_._indep_vars[0]
	use_labels = True
	for label in indep_var._bin_labels :
		label = str(label)
		if len(label) > 0 : continue 
		use_labels = False
	[ey_lo, ey_hi], keys = get_table_errors(table_,errs_)
	return indep_var._bin_centers, dep_var._values, [indep_var._bin_widths_lo,indep_var._bin_widths_hi], [ey_lo,ey_hi], use_labels, keys


#  Brief: plot 1D distribution from HEPDataTable table_
def plot_1D_distribution ( table_ , **kwargs ) :
	if table_.n_indep_vars() != 1 :
		msg.error("HEP_data_utils.plotting.plot_1D_distribution","Table has {0} independent variables where 1 was expected".format(table_.n_indep_vars()))
		return
	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot(111)
	legend_char_width = 53
	specific_errors = []
	for requested_error in kwargs.get("errors",[]) :
		if requested_error not in table_._dep_var._symerrors and requested_error not in table_._dep_var._asymerrors_up :
			msg.warning("HEP_data_utils.plotting.plot_1D_distribution","Specified error {0} not found in dependent variable {1}. Ignoring.".format(requested_error,table_._dep_var._name))
			continue
		specific_errors.append(requested_error)
	if len(specific_errors) > 0 :
		x, y, [ex_lo,ex_hi], [ey_lo,ey_hi], labels, keys = get_1D_distribution(table_,specific_errors)
		ey_lo_sys, ey_hi_sys, ey_lo_stat, ey_hi_stat = [], [], [], []
		str_tot_legend = kwargs.get("label","distribution") + " ( " + " + ".join(keys) + " )"
		str_tot_legend = "\n".join([str_tot_legend[legend_char_width*i:min(len(str_tot_legend),legend_char_width*(i+1))] for i in range(int(len(str_tot_legend)/legend_char_width)+1)])
	else :
		x, y, [ex_lo,ex_hi], [ey_lo,ey_hi], labels, keys = get_1D_distribution(table_)
		x, y, [ex_lo,ex_hi], [ey_lo_sys,ey_hi_sys], labels, sys_keys = get_1D_distribution(table_,"sys")
		x, y, [ex_lo,ex_hi], [ey_lo_stat,ey_hi_stat], labels, stat_keys = get_1D_distribution(table_,"stat")
		str_tot_legend = kwargs.get("label","distribution") + " ( " + " + ".join(keys) + " )"
		str_tot_legend = "\n".join([str_tot_legend[legend_char_width*i:min(len(str_tot_legend),legend_char_width*(i+1))] for i in range(int(len(str_tot_legend)/legend_char_width)+1)])
		str_sys_legend = kwargs.get("label","distribution") + " ( " + " + ".join(sys_keys) + " )"
		str_sys_legend = "\n".join([str_sys_legend[legend_char_width*i:min(len(str_sys_legend),legend_char_width*(i+1))] for i in range(int(len(str_sys_legend)/legend_char_width)+1)])
	if sum([np.fabs(el) for el in ey_hi_sys+ey_lo_sys]) > 0 :
		ax.errorbar(x, y, yerr=[ey_lo_sys,ey_hi_sys], c='royalblue', linewidth=18, linestyle='None', marker='None', alpha=0.4, label=str_sys_legend)
	if sum([np.fabs(el) for el in ey_hi_stat+ey_lo_stat]) > 0 :
		ax.errorbar(x, y, yerr=[ey_lo_stat,ey_hi_stat], c='indianred', linewidth=6, linestyle='None', marker='None', alpha=0.6, label=kwargs.get("label","distribution") + " ( stat )")
	ax.errorbar(x, y, yerr=[ey_lo,ey_hi], xerr=[ex_lo,ex_hi], c='k', linewidth=2, linestyle='None', marker='+', alpha=1, label=str_tot_legend)
	if labels :
		ax.set_xticks(x)
		ax.set_xticklabels(table_._indep_vars[0]._bin_labels,rotation=90)
	plt.subplots_adjust(left=0.1, right=0.5, top=0.95, bottom=0.4)
	if "legend_loc" in kwargs : ax.legend(loc=kwargs.get("legend_loc","best"))
	else : ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	try : plt.xlabel(kwargs.get("xlabel",table_._indep_vars[0].name().replace("\\\\","\\").replace(r"\text{",r"{\rm ")))
	except : plt.xlabel("<error reading xlabel")
	try : plt.ylabel(kwargs.get("ylabel",table_._dep_var.name().replace("\\\\","\\").replace(r"\text{",r"{\rm ")))
	except : plt.ylabel("<error reading ylabel")
	plt.title(kwargs.get("title",""))
	xlim = kwargs.get("xlim",[x[0]-np.fabs(ex_lo[0]),x[-1]+np.fabs(ex_hi[-1])])
	ylim = kwargs.get("ylim",None)
	ax.axis(xmin=xlim[0],xmax=xlim[1])
	if ylim : ax.axis(ymin=ylim[0],ymax=ylim[1])
	if kwargs.get("logy",False) is True : plt.yscale("log")
	if kwargs.get("logx",False) is True : plt.xscale("log")
	plt.grid()
	if kwargs.get("show",False) :
		plt.show()
	if kwargs.get("save",False) :
		fig.savefig ( document , format='pdf' )
	plt.close(fig)


#  Brief: plot ratio of 1D distributions from HEPDataTable table_
def plot_ratio ( table_num_ , table_den_ , **kwargs ) :
	specific_errors_num = [ err for err in kwargs.get("errors",[]) if err in { **table_num_._dep_var._symerrors, **table_num_._dep_var._asymerrors_up } ]
	specific_errors_den = [ err for err in kwargs.get("errors",[]) if err in { **table_den_._dep_var._symerrors, **table_den_._dep_var._asymerrors_up } ]
	x_n, y_n, [ex_lo_n,ex_hi_n], [ey_lo_n,ey_hi_n], labels, keys_num = get_1D_distribution(table_num_,specific_errors_num)
	x_d, y_d, [ex_lo_d,ex_hi_d], [ey_lo_d,ey_hi_d], labels, keys_den = get_1D_distribution(table_den_,specific_errors_den)
	ex_lo_d, ex_hi_d = np.zeros(shape=(len(x_d))), np.zeros(shape=(len(x_d)))
	for i in range(len(x_n)) :
		if x_n[i] == x_d[i] : continue
		msg.error("HEP_data_utils.plotting.plot_ratio","Arguments do not have the same binning")
		raise ValueError("Ratio of distributions with bin centres at {0} and {1}",x_n.all(),x_d.all()) 
	fig = plt.figure(figsize=(10,10))
	ax1 = fig.add_subplot(211)
	legend_char_width = 53
	str_num_legend = kwargs.get("numerator_label","numerator") + " ( " + " + ".join(keys_num) + " )"
	str_num_legend = "\n".join([str_num_legend[legend_char_width*i:min(len(str_num_legend),legend_char_width*(i+1))] for i in range(int(len(str_num_legend)/legend_char_width)+1)])
	str_den_legend = kwargs.get("denominator_label","denominator") + " ( " + " + ".join(keys_den) + " )"
	str_den_legend = "\n".join([str_den_legend[legend_char_width*i:min(len(str_den_legend),legend_char_width*(i+1))] for i in range(int(len(str_den_legend)/legend_char_width)+1)])
	ax1.errorbar(x_d, y_d, yerr=[ey_lo_d,ey_hi_d], xerr=[ex_lo_d,ex_hi_d], c='r', linewidth=7, linestyle='None', marker='+', alpha=0.5, label=str_den_legend)
	ax1.errorbar(x_n, y_n, yerr=[ey_lo_n,ey_hi_n], xerr=[ex_lo_n,ex_hi_n], c='k', linestyle='None', alpha=0.5, label=str_num_legend)
	if "legend_loc" in kwargs : ax1.legend(loc=kwargs.get("legend_loc","best"))
	else : ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.subplots_adjust(left=0.1, right=0.5, top=0.95, bottom=0.4)
	try : plt.ylabel(kwargs.get("ylabel",table_num_._dep_var.name().replace("\\\\","\\").replace(r"\text{",r"{\rm ")))
	except : plt.ylabel("<error reading ylabel")
	plt.title(kwargs.get("title",""))
	xlim = kwargs.get("xlim",[x_d[0]-np.fabs(ex_lo_n[0]),x_d[-1]+np.fabs(ex_hi_n[-1])])
	ylim = kwargs.get("ylim",None)
	ax1.axis(xmin=xlim[0],xmax=xlim[1])
	if ylim : ax1.axis(ymin=ylim[0],ymax=ylim[1])
	if kwargs.get("logy",False) is True : plt.yscale("log")
	if kwargs.get("logx",False) is True : plt.xscale("log")
	plt.grid()
	ax2 = fig.add_subplot(212)
	ax2.errorbar(x_d, y_d/y_d, yerr=[ey_lo_d/y_d,ey_hi_d/y_d], xerr=[ex_lo_d,ex_hi_d], c='r', linewidth=7, linestyle='None', marker='+', alpha=0.5)
	ax2.errorbar(x_n, y_n/y_d, yerr=[ey_lo_n/y_d,ey_hi_n/y_d], xerr=[ex_lo_n,ex_hi_n], c='k', linestyle='None', alpha=0.5)
	box = ax2.get_position()
	ax2.set_position([box.x0, box.y0, box.width * 0.4, box.height])
	ax2.axis(xmin=xlim[0],xmax=xlim[1])
	plt.ylabel("Ratio")
	try : plt.xlabel(kwargs.get("xlabel",table_den_._indep_vars[0].name().replace("\\\\","\\").replace(r"\text{",r"{\rm ")))
	except : plt.xlabel("<error reading xlabel")
	plt.subplots_adjust(left=0.1, right=0.5, top=0.95, bottom=0.4)
	plt.grid()
	if kwargs.get("show",False) :
		plt.show()
	if kwargs.get("save",False) :
		fig.savefig ( document , format='pdf' )
	plt.close(fig)


#  Brief: return bins_x,bins_y,value for 2D distribution
def get_2D_plottable_bins ( table_ ) :
	if str(type(table_)) != "<class 'HEP_data_utils.data_structures.HEPDataTable'>" :
		msg.fatal("HEP_data_utils.plotting.get_2D_plottable_bins","argument must be of type HEPDataTable")
	dep_var, indep_vars = table_._dep_var, table_._indep_vars
	if len(indep_vars) != 2 :
		msg.fatal("HEP_data_utils.plotting.get_2D_plottable_bins","HEPDataTable {0} has {1} independent_variable where 2 are required".format(dep_var.name(),len(indep_vars)))
	values, n_vals = dep_var._values, len(dep_var)
	old_bin_labels_x, old_bin_labels_y = indep_vars[0]._bin_labels, indep_vars[1]._bin_labels
	use_labels_x, use_labels_y = True, True
	for label in old_bin_labels_x :
		if len(label) == 0 : use_labels_x = False
	for label in old_bin_labels_y :
		if len(label) == 0 : use_labels_y = False
	if not use_labels_x :
		for i in range(len(old_bin_labels_x)) :
			old_bin_labels_x[i] = "{0:.2f}[{1:.2f},{2:.2f}]".format(indep_vars[0]._bin_centers[i],indep_vars[0]._bin_centers[i]-indep_vars[0]._bin_widths_lo[i],indep_vars[0]._bin_centers[i]+indep_vars[0]._bin_widths_hi[i])
	if not use_labels_y :
		for i in range(len(old_bin_labels_y)) :
			old_bin_labels_y[i] = "{0:.2f}[{1:.2f},{2:.2f}]".format(indep_vars[1]._bin_centers[i],indep_vars[1]._bin_centers[i]-indep_vars[1]._bin_widths_lo[i],indep_vars[1]._bin_centers[i]+indep_vars[1]._bin_widths_hi[i])
	old_n_bins_x = len(old_bin_labels_x)
	old_n_bins_y = len(old_bin_labels_y)
	if values.shape == (old_n_bins_x,old_n_bins_y) : return old_bin_labels_x, old_bin_labels_y, values
	if values.shape == (old_n_bins_y,old_n_bins_x) : return old_bin_labels_x, old_bin_labels_y, values.T()
	if n_vals == old_n_bins_x == old_n_bins_y :
		bin_labels_x = [y for y in {x for x in old_bin_labels_x}]
		bin_labels_x = natsorted(bin_labels_x, alg=ns.IGNORECASE)
		bin_labels_y = [y for y in {x for x in old_bin_labels_y}]
		bin_labels_y = natsorted(bin_labels_y, alg=ns.IGNORECASE)
		new_n_bins_x = len(bin_labels_x)
		new_n_bins_y = len(bin_labels_y)
		new_values = np.array(np.zeros(shape=(new_n_bins_x,new_n_bins_y)))
		for x,y,v in zip(old_bin_labels_x,old_bin_labels_y,values) :
			new_values[bin_labels_x.index(x),bin_labels_y.index(y)] = v
		return bin_labels_x, bin_labels_y, new_values
	if n_vals == old_n_bins_x*old_n_bins_y :
		new_values = np.array(np.zeros(shape=(old_n_bins_x,old_n_bins_y)))
		for x_idx in enumerate(old_bin_labels_x) :
			for y_idx in enumerate(old_bin_labels_y) :
				v = values[ x_idx + old_n_bins_x*y_idx ]
				new_values[x_idx,y_idx] = v
		return old_bin_labels_x, old_bin_labels_y, new_values
	msg.fatal("HEP_data_utils.plotting.get_2D_plottable_bins","HEPDataTable {0} is not a valid matrix".format(dep_var.name()))


#  Brief: return bins_x,bins_y,value for 2D distribution
def get_2D_distribution ( table_ ) :
	return get_2D_plottable_bins(table_)


#  Brief: plot 2D distribution from DistributionContainer dataset_ on the current axes
def plot_2D_distribution_on_current_axes ( table_ , **kwargs ) :
	if table_.n_indep_vars() != 2 :
		msg.error("HEP_data_utils.plotting.plot_2D_distribution_on_current_canvas","Table has {0} independent variables where 1 was expected".format(table_.name(),table_.n_indep_vars()))
		return
	dep_var, indep_var  = table_._dep_var, table_._indep_vars
	values = dep_var._values
	ax = plt.gca()
	try : plt.xlabel(kwargs.get("x_label",indep_var[0].name().replace("\\\\","\\").replace(r"\text{",r"{\rm ")))
	except Exception : pass 
	try : plt.ylabel(kwargs.get("y_label",indep_var[1].name().replace("\\\\","\\").replace(r"\text{",r"{\rm ")))
	except Exception : pass 
	max_val = max([np.fabs(val) for val in values.flatten()])
	vmin, vmax = -1*max_val, max_val
	if "vlim" in kwargs : 
		vmin = kwargs["vlim"][0]
		vmax = kwargs["vlim"][1]
	labels_x, labels_y, values = get_2D_distribution(table_)
	ax.imshow(values.transpose(),cmap="bwr",vmin=vmin,vmax=vmax)
	precision = kwargs.get("flt_precision",2)
	for i in range(len(labels_x)) :
		for j in range(len(labels_y)) :
			ax.text(i, j, "{0:.{1}f}".format(values[i, j],precision), ha="center", va="center", color="k", fontsize=kwargs.get("fontsize","xx-small"))
	ax.set_xticks(np.arange(len(labels_x)))
	ax.set_xticklabels(labels_x,rotation=90)
	ax.set_yticks(np.arange(len(labels_y)))
	ax.set_yticklabels(labels_y)


#  Brief: plot 2D distribution from DistributionContainer dataset_
def plot_2D_distribution ( table_ , **kwargs ) :
	if table_.n_indep_vars() != 2 :
		msg.error("HEP_data_utils.plotting.plot_2D_distribution","Table has {0} independent variables where 1 was expected".format(table_.name(),table_.n_indep_vars()))
		return
	dep_var, indep_var  = table_._dep_var, table_._indep_vars
	values = dep_var._values
	fig = plt.figure(figsize=(7,7))
	ax = fig.add_subplot(111)
	plot_2D_distribution_on_current_axes ( table_ , **kwargs )
	try : plt.title(table_._dep_var.name().replace("\\\\","\\").replace(r"\text{",r"{\rm "))
	except Exception : pass
	if kwargs.get("show",False) :
		plt.show()
	if kwargs.get("save",False) :
		fig.savefig ( document , format='pdf' )
	plt.close(fig)
