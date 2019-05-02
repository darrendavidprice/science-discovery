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


#  Brief: open a 1D distribution and turn it into plottable data
def get_1D_distribution ( table_ , err_ = "total" ) :
	dep_var = table_._dep_var
	indep_var = table_._indep_vars[0]
	bins = indep_var._bin_edges
	x = [ 0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1) ]
	ex = [ 0.5*(bins[i+1]-bins[i]) for i in range(len(bins)-1) ]
	use_labels = True
	for label in indep_var._bin_labels :
		label = str(label)
		if len(label) > 0 : continue 
		use_labels = False
	keys = []
	y = dep_var._values
	ey_lo = np.zeros(shape=(len(y)))
	ey_hi = np.zeros(shape=(len(y)))
	for key in dep_var._symerrors :
		if err_ != "total" and key[:len(err_)] != err_ : continue
		errs = dep_var._symerrors[key]
		keys.append(key)
		for i in range(0,len(errs)) :
			ey_lo[i] = ey_lo[i] + errs[i]*errs[i]
			ey_hi[i] = ey_hi[i] + errs[i]*errs[i]
	for key in dep_var._asymerrors_up :
		if err_ != "total" and key[:len(err_)] != err_ : continue
		errs1 = dep_var._asymerrors_up[key]
		errs2 = dep_var._asymerrors_dn[key]
		keys.append(key)
		for i in range(0,len(errs1)) :
			err1 = errs1[i]
			err2 = errs2[i]
			if err1 > 0 : ey_hi[i] = ey_hi[i] + err1*err1
			else : ey_lo[i] = ey_lo[i] + err1*err1
			if err2 > 0 : ey_hi[i] = ey_hi[i] + err2*err2
			else : ey_lo[i] = ey_lo[i] + err2*err2
	ey_lo = np.sqrt(ey_lo)
	ey_hi = np.sqrt(ey_hi)
	return x, y, [ey_lo,ey_hi], ex, use_labels, keys


#  Brief: plot 1D distribution from HEPDataTable table_
def plot_1D_distribution ( table_ , **kwargs ) :
	if table_.n_indep_vars() != 1 :
		msg.error("plotting.plot_1D_distribution","Table has {0} independent variables where 1 was expected".format(table_.n_indep_vars()))
		return
	x, y, [ey_lo,ey_hi], ex, labels, keys = get_1D_distribution(table_)
	x, y, [ey_lo_sys,ey_hi_sys], ex, labels, sys_keys = get_1D_distribution(table_,"sys")
	x, y, [ey_lo_stat,ey_hi_stat], ex, labels, stat_keys = get_1D_distribution(table_,"stat")
	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot(111)
	legend_char_width = 53
	str_tot_legend = kwargs.get("label","distribution") + " ( " + " + ".join(keys) + " )"
	str_tot_legend = "\n".join([str_tot_legend[legend_char_width*i:min(len(str_tot_legend),legend_char_width*(i+1))] for i in range(int(len(str_tot_legend)/legend_char_width)+1)])
	str_sys_legend = kwargs.get("label","distribution") + " ( " + " + ".join(sys_keys) + " )"
	str_sys_legend = "\n".join([str_sys_legend[legend_char_width*i:min(len(str_sys_legend),legend_char_width*(i+1))] for i in range(int(len(str_sys_legend)/legend_char_width)+1)])
	if sum([np.fabs(x) for x in ey_hi_sys+ey_lo_sys]) > 0 :
		ax.errorbar(x, y, yerr=[ey_lo_sys,ey_hi_sys], c='royalblue', linewidth=18, linestyle='None', marker='None', alpha=0.4, label=str_sys_legend)
	if sum([np.fabs(x) for x in ey_hi_stat+ey_lo_stat]) > 0 :
		ax.errorbar(x, y, yerr=[ey_lo_stat,ey_hi_stat], c='indianred', linewidth=6, linestyle='None', marker='None', alpha=0.6, label=kwargs.get("label","distribution") + " ( stat )")
	ax.errorbar(x, y, yerr=[ey_lo,ey_hi], xerr=ex, c='k', linewidth=2, linestyle='None', marker='+', alpha=1, label=str_tot_legend)
	if labels :
		ax.set_xticks(x)
		ax.set_xticklabels(table_._indep_vars[0]._bin_labels,rotation=90)
	plt.subplots_adjust(left=0.1, right=0.5, top=0.95, bottom=0.4)
	if "legend_loc" in kwargs : ax.legend(loc=kwargs.get("legend_loc","best"))
	else : ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlabel(kwargs.get("xlabel",table_._indep_vars[0].name().replace(r"\text{",r"{\rm ")))
	plt.ylabel(kwargs.get("ylabel",table_._dep_var.name().replace(r"\text{",r"{\rm ")))
	plt.title(kwargs.get("title",""))
	xlim = kwargs.get("xlim",[x[0]-np.fabs(ex[0]),x[-1]+np.fabs(ex[-1])])
	ylim = kwargs.get("ylim",None)
	ax.axis(xmin=xlim[0],xmax=xlim[1])
	if ylim : ax.axis(ymin=ylim[0],ymax=ylim[1])
	if kwargs.get("logy",False) is True : plt.yscale("log")
	if kwargs.get("logx",False) is True : plt.xscale("log")
	plt.grid()
	plt.show()
	if kwargs.get("save",False) :
		fig.savefig ( document , format='pdf' )
		plt.close(fig)


#  Brief: plot ratio of 1D distributions from HEPDataTable table_
def plot_ratio ( table_num_ , table_den_ , **kwargs ) :
	x_n, y_n, [ey_lo_n,ey_hi_n], ex_n, labels, keys_num = get_1D_distribution(table_num_)
	x_d, y_d, [ey_lo_d,ey_hi_d], ex_d, labels, keys_den = get_1D_distribution(table_den_)
	ex_d = np.zeros(shape=(len(x_d)))
	if x_n != x_d :
		msg.error("HEP_data_helpers.plot_ratio","Arguments do not have the same binning")
		raise ValueError("Ratio of distributions with bin centres at {0} and {1}",x_n,x_d) 
	fig = plt.figure(figsize=(10,10))
	ax1 = fig.add_subplot(211)
	legend_char_width = 53
	str_num_legend = kwargs.get("numerator_label","numerator") + " ( " + " + ".join(keys_num) + " )"
	str_num_legend = "\n".join([str_num_legend[legend_char_width*i:min(len(str_num_legend),legend_char_width*(i+1))] for i in range(int(len(str_num_legend)/legend_char_width)+1)])
	str_den_legend = kwargs.get("denominator_label","denominator") + " ( " + " + ".join(keys_den) + " )"
	str_den_legend = "\n".join([str_den_legend[legend_char_width*i:min(len(str_den_legend),legend_char_width*(i+1))] for i in range(int(len(str_den_legend)/legend_char_width)+1)])
	ax1.errorbar(x_d, y_d, yerr=[ey_lo_d,ey_hi_d], xerr=ex_d, c='r', linewidth=7, linestyle='None', marker='+', alpha=0.5, label=str_den_legend)
	ax1.errorbar(x_n, y_n, yerr=[ey_lo_n,ey_hi_n], xerr=ex_n, c='k', linestyle='None', alpha=0.5, label=str_num_legend)
	if "legend_loc" in kwargs : ax1.legend(loc=kwargs.get("legend_loc","best"))
	else : ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.subplots_adjust(left=0.1, right=0.5, top=0.95, bottom=0.4)
	plt.ylabel(kwargs.get("ylabel",table_num_._dep_var.name().replace(r"\text{",r"{\rm ")))
	plt.title(kwargs.get("title",""))
	xlim = kwargs.get("xlim",[x_d[0]-np.fabs(ex_n[0]),x_d[-1]+np.fabs(ex_n[-1])])
	ylim = kwargs.get("ylim",None)
	ax1.axis(xmin=xlim[0],xmax=xlim[1])
	if ylim : ax1.axis(ymin=ylim[0],ymax=ylim[1])
	if kwargs.get("logy",False) is True : plt.yscale("log")
	if kwargs.get("logx",False) is True : plt.xscale("log")
	plt.grid()
	ax2 = fig.add_subplot(212)
	ax2.errorbar(x_d, y_d/y_d, yerr=[ey_lo_d/y_d,ey_hi_d/y_d], xerr=ex_d, c='r', linewidth=7, linestyle='None', marker='+', alpha=0.5)
	ax2.errorbar(x_n, y_n/y_d, yerr=[ey_lo_n/y_d,ey_hi_n/y_d], xerr=ex_n, c='k', linestyle='None', alpha=0.5)
	box = ax2.get_position()
	ax2.set_position([box.x0, box.y0, box.width * 0.4, box.height])
	ax2.axis(xmin=xlim[0],xmax=xlim[1])
	plt.ylabel("Ratio")
	if "xlabel" in kwargs : plt.xlabel(kwargs["xlabel"])
	plt.subplots_adjust(left=0.1, right=0.5, top=0.95, bottom=0.4)
	plt.grid()
	plt.show()


#  Brief: return bins_x,bins_y,value for 2D distribution
def get_2D_plottable_bins ( table_ ) :
	if str(type(table_)) != "<class 'HEP_data_utils.data_structures.HEPDataTable'>" :
		msg.fatal("HEP_data_helpers.get_2D_plottable_bins","argument must be of type HEPDataTable")
	dep_var, indep_vars = table_._dep_var, table_._indep_vars
	if len(indep_vars) != 2 :
		msg.fatal("HEP_data_helpers.get_2D_plottable_bins","HEPDataTable {0} has {1} independent_variable where 2 are required".format(dep_var.name(),len(indep_vars)))
	values, n_vals = dep_var._values, len(dep_var)
	old_bin_labels_x, old_bin_labels_y = indep_vars[0]._bin_labels, indep_vars[1]._bin_labels
	use_labels_x, use_labels_y = True, True
	for label in old_bin_labels_x :
		if len(label) == 0 : use_labels_x = False
	for label in old_bin_labels_y :
		if len(label) == 0 : use_labels_y = False
	if not use_labels_x :
		for i in range(n_vals) : old_bin_labels_x = "[{0},{1}]".format(dep_var._bin_edges[i],dep_var._bin_edges[i+1])
	if not use_labels_y :
		for i in range(n_vals) : old_bin_labels_y = "[{0},{1}]".format(dep_var._bin_edges[i],dep_var._bin_edges[i+1])
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
	msg.fatal("HEP_data_helpers.get_2D_plottable_bins","HEPDataTable {0} is not a valid matrix".format(dep_var.name()))


#  Brief: return bins_x,bins_y,value for 2D distribution
def get_2D_distribution ( table_ ) :
	return get_2D_plottable_bins(table_)


#  Brief: plot 2D distribution from DistributionContainer dataset_
def plot_2D_distribution ( table_ , **kwargs ) :
	if table_.n_indep_vars() != 2 :
		msg.error("plotting.plot_2D_distribution","Table has {0} independent variables where 1 was expected".format(table_.name(),table_.n_indep_vars()))
		return
	dep_var  = table_._dep_var
	indep_var  = table_._indep_vars
	values = dep_var._values
	fig = plt.figure(figsize=(7,7))
	ax = fig.add_subplot(111)
	#x_label = str([ "{0} [{1}:{2}]".format(var,table_._local_key_indices[var][0],table_._local_key_indices[var][1]) for var in table_._local_keys ])
	x_label = kwargs.get("x_label",indep_var[0]._name)
	y_label = kwargs.get("y_label",indep_var[1]._name)
	max_val = max([np.fabs(val) for val in values.flatten()])
	vmin = -1*max_val
	vmax = max_val
	if "vlim" in kwargs : 
		vmin = kwargs["vlim"][0]
		vmax = kwargs["vlim"][1]
	labels_x, labels_y, values = get_2D_distribution(table_)
	ax.imshow(values,cmap="bwr",vmin=vmin,vmax=vmax)
	plt.xlabel(kwargs.get("xlabel",x_label))
	plt.ylabel(kwargs.get("ylabel",y_label))
	precision = kwargs.get("flt_precision",2)
	for i in range(len(labels_x)) :
		for j in range(len(labels_y)) :
			ax.text(j, i, "{0:.{1}f}".format(values[i, j],precision), ha="center", va="center", color="k",fontsize="xx-small")
	plt.title(kwargs.get("title",dep_var._name))
	plt.show()
