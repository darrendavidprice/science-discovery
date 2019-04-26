# ====================================================================================================
#  Brief: functions for plotting the contents of HEP_data_utils.data_structures.SubmissionFileTable
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================

import matplotlib.pyplot as plt
import numpy as np

import HEP_data_utils.messaging as msg


#  Brief: open a 1D distribution and turn it into plottable data
def get_1D_distribution ( dataset_ , key_ , err_ = "total" ) :
	if key_ not in dataset_._1D_distributions :
		msg.fatal("plotting.get_1D_distribution","No 1D distribution with key {0} in {1}".format(key_,dataset_._name))
	dist = dataset_._1D_distributions[key_]
	dep_var = dist._dep_var
	indep_var = dist._indep_vars[0]
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
		keys.append(key+"(asymm)")
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


#  Brief: plot 1D distribution from DistributionContainer dataset_
def plot_1D_distribution ( dataset_ , key_ , **kwargs ) :
	if key_ not in dataset_._1D_distributions :
		msg.error("plotting.plot_1D_distribution","No 1D distribution with key {0} in {1}".format(key_,dataset_._name))
		return
	dist = dataset_._1D_distributions[key_]
	x, y, [ey_lo,ey_hi], ex, labels, keys  = get_1D_distribution(dataset_,key_)
	x, y, [ey_lo_sys,ey_hi_sys], ex, labels, sys_keys = get_1D_distribution(dataset_,key_,"sys")
	x, y, [ey_lo_stat,ey_hi_stat], ex, labels, stat_keys = get_1D_distribution(dataset_,key_,"stat")
	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot(111)
	legend_char_width = 53
	str_tot_legend = kwargs.get("label","distribution") + " ( " + " + ".join(keys) + " )"
	str_tot_legend = "\n".join([str_tot_legend[legend_char_width*i:min(len(str_tot_legend),legend_char_width*(i+1))] for i in range(int(len(str_tot_legend)/legend_char_width)+1)])
	str_sys_legend = kwargs.get("label","distribution") + " ( " + " + ".join(sys_keys) + " )"
	str_sys_legend = "\n".join([str_sys_legend[legend_char_width*i:min(len(str_sys_legend),legend_char_width*(i+1))] for i in range(int(len(str_sys_legend)/legend_char_width)+1)])
	if sum([np.fabs(x) for x in ey_hi_sys+ey_lo_sys]) > 0 :
		ax.errorbar(x, y, yerr=[ey_lo_sys,ey_hi_sys], c='royalblue', linewidth=18, linestyle='None', marker='None', alpha=0.4, label=str_tot_legend)
	if sum([np.fabs(x) for x in ey_hi_stat+ey_lo_stat]) > 0 :
		ax.errorbar(x, y, yerr=[ey_lo_stat,ey_hi_stat], c='indianred', linewidth=6, linestyle='None', marker='None', alpha=0.6, label=kwargs.get("label","distribution") + " ( stat )")
	ax.errorbar(x, y, yerr=[ey_lo,ey_hi], xerr=ex, c='k', linewidth=2, linestyle='None', marker='+', alpha=1, label=str_sys_legend)
	if labels :
		ax.set_xticks(x)
		ax.set_xticklabels(dist._indep_vars[0]._bin_labels,rotation=90)
	plt.subplots_adjust(left=0.1, right=0.5, top=0.95, bottom=0.4)
	if "legend_loc" in kwargs : ax.legend(loc=kwargs.get("legend_loc","best"))
	else : ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlabel(kwargs.get("xlabel",dist._indep_vars[0].name().replace(r"\text{",r"{\rm ")))
	plt.ylabel(kwargs.get("ylabel",dist._dep_var.name().replace(r"\text{",r"{\rm ")))
	plt.title(kwargs.get("title",""))
	xlim = kwargs.get("xlim",[x[0]-np.fabs(ex[0]),x[-1]+np.fabs(ex[-1])])
	ylim = kwargs.get("ylim",None)
	ax.axis(xmin=xlim[0],xmax=xlim[1])
	if ylim : ax.axis(ymin=ylim[0],ymax=ylim[1])
	if kwargs.get("logy",False) is True : plt.yscale("log")
	if kwargs.get("logx",False) is True : plt.xscale("log")
	plt.grid()
	plt.show()


#  Brief: return bins_x,bins_y,value for 2D distribution
def get_2D_distribution ( dist_2D_ ) :
	if str(type(dist_2D_)) != "<class 'HEP_data_utils.data_structures.HEPDataTable'>" :
		msg.fatal("HEP_data_helpers.regularise_bins","argument must be of type HEPDataTable")
	if len(dist_2D_._indep_vars) != 2 :
		msg.fatal("HEP_data_helpers.regularise_bins","HEPDataTable {0} has {1} independent_variable where 2 are required".format(dist_2D_._dep_var._name,len(dist_2D_._indep_vars)))
	n_vals = len(dist_2D_._dep_var)
	values = dist_2D_._dep_var._values
	old_bin_labels_x = dist_2D_._indep_vars[0]._bin_labels
	old_bin_labels_y = dist_2D_._indep_vars[1]._bin_labels
	old_n_bins_x = len(old_bin_labels_x)
	old_n_bins_y = len(old_bin_labels_y)
	if n_vals == old_n_bins_x == old_n_bins_y :
		bin_labels_x = [y for y in {x for x in old_bin_labels_x}]
		bin_labels_x.sort()
		bin_labels_y = [y for y in {x for x in old_bin_labels_y}]
		bin_labels_y.sort()
		new_n_bins_x = len(bin_labels_x)
		new_n_bins_y = len(bin_labels_y)
		new_values = np.array(np.zeros(shape=(new_n_bins_x,new_n_bins_y)))
		for x,y,v in zip(old_bin_labels_x,old_bin_labels_y,values) :
			new_values[bin_labels_x.index(x),bin_labels_y.index(y)] = v
		return bin_labels_x, bin_labels_y, new_values
	if n_vals == n_bins_x*n_bins_y :
		new_values = np.array(np.zeros(shape=(n_bins_x,n_bins_y)))
		for x_idx in enumerate(old_bin_labels_x) :
			for y_idx in enumerate(old_bin_labels_y) :
				v = values[ x_idx + n_bins_x*y_idx ]
				new_values[x_idx,y_idx] = v
		return old_bin_labels_x, old_bin_labels_y, new_values
	msg.fatal("HEP_data_helpers.regularise_bins","HEPDataTable {0} is not a valid matrix".format(dist_2D_._dep_var._name))


#  Brief: plot 2D distribution from DistributionContainer dataset_
def plot_2D_distribution ( dataset_ , key_ , **kwargs ) :
	if key_ not in dataset_._2D_distributions :
		msg.error("plotting.plot_2D_distribution","No 2D distribution with key {0} in {1}".format(key_,dataset_._name))
		return
	dist = dataset_._2D_distributions[key_]
	dep_var  = dist._dep_var
	indep_var  = dist._indep_vars
	values = dep_var._values
	fig = plt.figure(figsize=(7,7))
	ax = fig.add_subplot(111)
	#x_label = str([ "{0} [{1}:{2}]".format(var,dist._local_key_indices[var][0],dist._local_key_indices[var][1]) for var in dist._local_keys ])
	x_label = kwargs.get("x_label",indep_var[0]._name)
	y_label = kwargs.get("y_label",indep_var[1]._name)
	max_val = max([np.fabs(val) for val in values.flatten()])
	vmin = -1*max_val
	vmax = max_val
	if "vlim" in kwargs : 
		vmin = kwargs["vlim"][0]
		vmax = kwargs["vlim"][1]
	labels_x, labels_y, values = get_2D_distribution(dist)
	ax.imshow(values,cmap="bwr",vmin=vmin,vmax=vmax)
	plt.xlabel(kwargs.get("xlabel",x_label))
	plt.ylabel(kwargs.get("ylabel",y_label))
	precision = kwargs.get("flt_precision",2)
	for i in range(len(labels_x)) :
		for j in range(len(labels_y)) :
			ax.text(j, i, "{0:.{1}f}".format(values[i, j],precision), ha="center", va="center", color="k",fontsize="xx-small")
	plt.title(kwargs.get("title",dep_var._name))
	plt.show()
