import numpy as np
import sys, configparser, yaml
import matplotlib.pyplot as plt

import general_utils.messaging as msg
import HEP_data_utils.plotting as HEPData_plt


class Distribution (object) :
	def __init__ ( self ) :
		self._name = "unknown"
		self._values = np.array([])
		self._symm_errors = {}
		self._asymm_errors_up = {}
		self._asymm_errors_down = {}
		self._description = "unknown"
		self._units = "unknown"
		self._meta = {}
		self._has_errors = False
	def __type__ ( self ) :
		return "Distribution"
	def __len__ ( self ) :
		return len(self._values)
	def plot ( self ) :
		msg.fatal("Distribution.plot","not implemented for base class - you should have created a Distribution_1D or Distribution_2D object")
	def print_meta ( self ) :
		msg.info("Distribution.print_meta","printing all metadata for object "+self._name)
		for key in self._meta : print("{0}   :   {1}".format(key,self._meta[key]))

class Distribution_1D (Distribution) :
	def __init__ ( self ) :
		super(Distribution_1D,self).__init__()
		self._bin_values = np.array([])
		self._bin_labels = []
		self._dep_var = "observable"
		self._indep_var = "observable"
	def __type__ ( self ) :
		return "Distribution_1D"
	def __len__ ( self ) :
		return len(self._values)
	def __str__ ( self ) :
		ret = "1D Distribution\n   - name: " + self._name
		ret = ret + "\n   - description: " + self._description
		ret = ret + "\n   - units: " + self._units
		ret = ret + "\n   - values ({0}): ".format(len(self._values)) + str(self._values)
		for err in self._symm_errors :
			ret = ret + "\n   - symmetric error [{0}]: ".format(err) + str(self._symm_errors[err])
		for err in self._asymm_errors_up :
			ret = ret + "\n   - asymmetric error [{0}]_up  : ".format(err) + str(self._asymm_errors_up[err])
			ret = ret + "\n   - asymmetric error [{0}]_down: ".format(err) + str(self._asymm_errors_down[err])
		ret = ret + "\n   - bin labels: " + str(self._bin_labels)
		ret = ret + "\n   - bin values: " + str(self._bin_values)
		return ret

class Distribution_2D (Distribution) :
	def __init__ ( self ) :
		super(Distribution_2D,self).__init__()
		self._local_keys = []
		self._local_key_indices = {}
		self._bin_labels_x = []
		self._bin_labels_y = []
	def __type__ ( self ) :
		return "Distribution_2D"
	def __len__ ( self ) :
		return len(self._values)
	def __str__ ( self ) :
		ret = "2D Distribution\n   - name: " + self._name
		ret = "   - variable keys are: " + str(["{0}@[{1},{2}]".format(key,self._local_key_indices[k][0],self._local_key_indices[k][1]) for key in self._local_keys])
		ret = ret + "\n   - description: " + self._description
		ret = ret + "\n   - units: " + self._units
		ret = ret + "\n   - values ({0}): ".format(len(self._values)) + str(self._values)
		for err in self._symm_errors :
			ret = ret + "\n   - symmetric error [{0}]: ".format(err) + str(self._symm_errors[err])
		for err in self._asymm_errors_up :
			ret = ret + "\n   - asymmetric error [{0}]_up  : ".format(err) + str(self._asymm_errors_up[err])
			ret = ret + "\n   - asymmetric error [{0}]_down: ".format(err) + str(self._asymm_errors_down[err])
		ret = ret + "\n   - bin labels (x,{0}): ".format(len(self._bin_labels_x)) + str(self._bin_labels_x)
		ret = ret + "\n   - bin labels (y,{0}): ".format(len(self._bin_labels_y)) + str(self._bin_labels_y)
		return ret
	def set_local_key ( self , key_ , key_idx_lower_ , key_idx_upper_ ) :
		if key_ not in self._local_keys : self._local_keys.append(key_)
		if key_idx_lower_ > key_idx_upper_ :
			msg.error("HEP_data_utils.data_structures.Distribution_2D.set_local_key","upper index {0} cannot be greater than lower index {1}... returning with nothing done".format(key_idx_lower_,key_idx_upper_),_verbose_level=0)
			return
		self._local_key_indices[key_] = [key_idx_lower_,key_idx_upper_]
	def remove_local_key ( self , key_ ) :
		if key_ not in self._local_keys :
			msg.error("HEP_data_utils.data_structures.Distribution_2D.remove_local_key","key {0} does not exist... returning with nothing done".format(key_),_verbose_level=0)
			return
		self._local_keys.remove(key_)
		del self._local_key_indices[key_]
	def change_local_key ( self , old_key_ , new_key_ ) :
		if old_key_ not in self._local_keys :
			msg.error("HEP_data_utils.data_structures.Distribution_2D.change_local_key","key {0} does not exist... returning with nothing done".format(old_key_),_verbose_level=0)
			return
		if new_key_ in self._local_keys :
			msg.error("HEP_data_utils.data_structures.Distribution_2D.change_local_key","key {0} already exists... returning with nothing done".format(new_key_),_verbose_level=0)
			return
		self._local_keys.remove(old_key_)
		self._local_keys.append(new_key_)
		self._local_key_indices[new_key_] = self._local_key_indices[old_key_]
		del self._local_key_indices[old_key_]

class Distribution_store (object) :
	def __init__ ( self , name_ = "unnamed" ) :
		self._distributions_1D = {}
		self._distributions_2D = {}
		self._name = name_
		self._description = ""
		self._location = ""
		self._comment = ""
		self._hepdata_doi = ""
	def __type__ ( self ) :
		return "Distribution_store"
	def __len__ ( self ) :
		return len(self._distributions) + len(self._correlations)
	def clear ( self ) :
		self._distributions_1D.clear()
		self._distributions_2D.clear()
		self._description = ""
		self._location = ""
		self._comment = ""
		self._hepdata_doi = ""
	def get_1D_keys ( self ) :
		return [ key for key in self._distributions_1D ]
	def get_2D_keys ( self ) :
		return [ key for key in self._distributions_2D ]
	def __str__ ( self ) :
		ret = "Distribution_store  [" + self._name + "]"
		if len(self._hepdata_doi) > 0 : ret = ret + "\n   - " + self._hepdata_doi
		if len(self._location) > 0 : ret = ret + "\n   - " + self._location
		if len(self._description) > 0 : ret = ret + "\n   - " + self._description
		if len(self._comment) > 0 : ret = ret + "\n   - " + self._comment
		for key in self._distributions_1D :
			d = self._distributions_1D[key]
			ret = ret + "\n   \033[1m- 1D distribution with key \"\033[95m" + key + "\033[0m\033[1m\", name  \"" + str(d._name) + "\" and {0} bins\033[0m".format(len(d._values))
		for key in self._distributions_2D :
			d = self._distributions_2D[key]
			ret = ret + "\n   \033[1m- 2D distribution with key \"\033[95m" + key + "\033[0m\033[1m\", name  \"" + str(d._name) + "\" and {0} bins\033[0m".format(len(d._values))
		return ret
	def rename ( self , old_key_ , new_key_ ) :
		something_done = False
		old_key_ = old_key_.replace("\text","\\text")
		for key in self._distributions_1D :
			if old_key_ != key : continue
			self._distributions_1D[new_key_] = self._distributions_1D.pop(old_key_)
			msg.info("Distribution_store.rename","Store \"{0}\" renaming 1D distribution key {1} to {2}".format(self._name,old_key_,new_key_),_verbose_level=0)
			something_done = True
		for key in self._distributions_2D :
			dist_key = key
			if old_key_ == key :
				self._distributions_2D[new_key_] = self._distributions_2D.pop(old_key_)
				msg.info("Distribution_store.rename","Store \"{0}\" renaming 2D distribution key {1} to {2}".format(self._name,old_key_,new_key_),_verbose_level=0)
				dist_key = new_key_
				something_done = True
			indices = [idx for idx, k2 in enumerate(self._distributions_2D[dist_key]._local_keys) if k2 == old_key_]
			if old_key_ not in indices : continue
			for idx in indices :
				self._distributions_2D[dist_key]._local_keys[idx] = new_key_
				msg.info("Distribution_store.rename","Store \"{0}\" using 2D distribution key {1}... renaming subkey {2} to {3}".format(self._name,dist_key,old_key_,new_key_),_verbose_level=0)
				something_done = True
		if not something_done :
			msg.warning("Distribution_store.rename","Store \"{0}\" with nothing done for old_key_={1}, new_key_={2}".format(self._name,old_key_,new_key_),_verbose_level=0)
	def load_keys ( self , filename_ ) :
		config = configparser.ConfigParser()
		config.optionxform = str
		try : config.read(filename_)
		except :
			msg.check_verbosity_and_print ( str(sys.exc_info()[0]) , _verbose_level=-1 )
			msg.error("HEP_data_utils.data_structures.Distribution_store","an exception occured when parsing the config file... Continuing with nothing done")
			return
		if "KEYS" not in config.sections() :
			msg.error("HEP_data_utils.data_structures.Distribution_store","no section titled \"KEYS\" in file {0}".format(filename_))
			return
		keys = config["KEYS"]
		for old_key in keys :
			self.rename(old_key,keys[old_key])
		self.print_keys()
	def print_keys ( self ) :
		msg.info ( "Distribution_store.print_keys" , "keys for _distributions_1D are:" )
		for key in self._distributions_1D : print(key)
		msg.info ( "Distribution_store.print_keys" , "keys for _distributions_2D are:" )
		for key in self._distributions_2D : print(key)
		for key in self._distributions_2D : msg.info("Distribution_store.print_keys","2D distribution [key={0}] with local-keys: {1}".format(key,["{0}@{1}".format(little_key,self._distributions_2D[key]._local_key_indices[little_key]) for little_key in self._distributions_2D[key]._local_keys]))
		msg.info ( "Distribution_store.print_keys" , "N.B. you can rename these keys using obj.rename(<old-key>,<new-key>)" , _verbose_level=0 )
	def plot_1D_distribution ( self , key_ , **kwargs ) :
		if self._distributions_1D[key_]._has_errors :
			msg.error("Distribution_store.plot_1D_distribution","Key {0} had some errors when loading. Please clear them before plotting.".format(key_),-1)
			return
		x, y, [ey_lo,ey_hi], ex, labels, keys  = HEPData_plt.get_1D_distribution(self,key_)
		x, y, [ey_lo_sys,ey_hi_sys], ex, labels, sys_keys = HEPData_plt.get_1D_distribution(self,key_,"sys")
		x, y, [ey_lo_stat,ey_hi_stat], ex, labels, stat_keys = HEPData_plt.get_1D_distribution(self,key_,"stat")
		fig = plt.figure(figsize=(15,5))
		ax = fig.add_subplot(111)
		str_tot_legend = kwargs.get("label","distribution") + " ( " + " + ".join(keys) + " )"
		str_tot_legend = "\n".join([str_tot_legend[120*i:min(len(str_tot_legend),120*(i+1))] for i in range(int(len(str_tot_legend)/120)+1)])
		str_sys_legend = kwargs.get("label","distribution") + " ( " + " + ".join(sys_keys) + " )"
		str_sys_legend = "\n".join([str_sys_legend[120*i:min(len(str_sys_legend),120*(i+1))] for i in range(int(len(str_sys_legend)/120)+1)])
		if sum([np.fabs(x) for x in ey_hi_sys+ey_lo_sys]) > 0 : ax.errorbar(x, y, yerr=[ey_lo_sys,ey_hi_sys], c='royalblue', linewidth=18, linestyle='None', marker='None', alpha=0.4, label=str_tot_legend)
		if sum([np.fabs(x) for x in ey_hi_stat+ey_lo_stat]) > 0 : ax.errorbar(x, y, yerr=[ey_lo_stat,ey_hi_stat], c='indianred', linewidth=6, linestyle='None', marker='None', alpha=0.6, label=kwargs.get("label","distribution") + " ( stat )")
		ax.errorbar(x, y, yerr=[ey_lo,ey_hi], xerr=ex, c='k', linewidth=2, linestyle='None', marker='+', alpha=1, label=str_sys_legend)
		if labels :
			ax.set_xticks(x)
			ax.set_xticklabels(self._distributions_1D[key_]._bin_labels,rotation=45)
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.4, box.height])
		if "legend_loc" in kwargs : ax.legend(loc=kwargs.get("legend_loc","best"))
		else : ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.xlabel(kwargs.get("xlabel",self._distributions_1D[key_]._indep_var))
		plt.ylabel(kwargs.get("ylabel",self._distributions_1D[key_]._dep_var))
		plt.title(kwargs.get("title",""))
		ax.axis(xlim=kwargs.get("xlim",[x[0],x[len(x)-1]]))
		ax.axis(ylim=kwargs.get("ylim",[x[0],x[len(x)-1]]))
		if kwargs.get("logy",False) is True : plt.yscale("log")
		if kwargs.get("logx",False) is True : plt.xscale("log")
		plt.grid()
		plt.show()
	def plot_data_vs_prediction ( self , key_meas_ , key_pred_ , **kwargs ) :
		x_m, y_m, [ey_lo_m,ey_hi_m], ex_m, labels, keys_meas = HEPData_plt.get_1D_distribution(self,key_meas_)
		x_p, y_p, [ey_lo_p,ey_hi_p], ex_p, labels, keys_pred = HEPData_plt.get_1D_distribution(self,key_pred_)
		fig = plt.figure(figsize=(15,7))
		ax1 = fig.add_subplot(211)
		ax1.errorbar(x_p, y_p, yerr=[ey_lo_p,ey_hi_p], xerr=ex_p, c='r', linestyle='None', marker='+', alpha=0.8, label="Prediction ( "+" $\oplus$ ".join(keys_meas)+" )")
		ax1.errorbar(x_m, y_m, yerr=[ey_lo_m,ey_hi_m], xerr=ex_m, c='k', linestyle='None', alpha=1, label="Data ( "+" $\oplus$ ".join(keys_pred)+" )")
		box = ax1.get_position()
		ax1.set_position([box.x0, box.y0, box.width * 0.4, box.height])
		if "legend_loc" in kwargs : ax1.legend(loc=kwargs.get("legend_loc","best"))
		else : ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.ylabel(kwargs.get("ylabel","observable"))
		plt.title(kwargs.get("title",""))
		ax.axis(xlim=kwargs.get("xlim",[x_m[0],x_m[len(x_m)-1]]))
		ax.axis(ylim=kwargs.get("ylim",[x_m[0],x_m[len(x_m)-1]]))
		if kwargs.get("logy",False) is True : plt.yscale("log")
		if kwargs.get("logx",False) is True : plt.xscale("log")
		plt.grid()
		ax2 = fig.add_subplot(212)
		ax2.errorbar(x_p, y_p/y_p, yerr=[ey_lo_p/y_p,ey_hi_p/y_p], xerr=ex_p, c='r', linestyle='None', marker='+', alpha=0.8)
		ax2.errorbar(x_m, y_m/y_p, yerr=[ey_lo_m/y_p,ey_hi_m/y_p], xerr=ex_m, c='k', linestyle='None', alpha=1)
		box = ax2.get_position()
		ax2.set_position([box.x0, box.y0, box.width * 0.4, box.height])
		ax.axis(xlim=kwargs.get("xlim",[x_m[0],x_m[len(x_m)-1]]))
		plt.ylabel("Measured / prediction")
		if "xlabel" in kwargs : plt.xlabel(kwargs["xlabel"])
		if labels :
			ax2.set_xticks(x)
			ax2.set_xticklabels(self._distributions_1D[key_]._bin_labels,rotation=45)
		plt.grid()
		plt.show()
	def plot_matrix ( self , key_ , **kwargs ) :
		dist = self._distributions_2D[key_]
		fig = plt.figure(figsize=(7,7))
		ax = fig.add_subplot(111)
		x_label = str([ "{0} [{1}:{2}]".format(var,dist._local_key_indices[var][0],dist._local_key_indices[var][1]) for var in dist._local_keys ])
		max_val = max([np.fabs(val) for val in dist._values.flatten()])
		vmin = -1*max_val
		vmax = max_val
		if "vlim" in kwargs : 
			vmin = kwargs["vlim"][0]
			vmax = kwargs["vlim"][1]
		ax.imshow(dist._values,cmap="bwr",vmin=vmin,vmax=vmax)
		plt.xlabel(kwargs.get("xlabel",x_label))
		plt.ylabel(kwargs.get("ylabel",x_label))
		precision = kwargs.get("flt_precision",2)
		for i in range(len(dist._bin_labels_x)) :
			for j in range(len(dist._bin_labels_y)) :
				ax.text(j, i, "{0:.{1}f}".format(dist._values[i, j],precision), ha="center", va="center", color="k",fontsize="xx-small")
		if "title" in kwargs : plt.title(kwargs["title"])
		plt.show()
	def plot_all ( self ) :
		for key in self._distributions_1D :
			try : self.plot_1D_distribution(key,label=key)
			except Exception as e :
				print(e)
				msg.error("Distribution_store.plot_all","Error when plotting 1D distribution with key {0}... skipping".format(key))
		for key in self._distributions_2D :
			try : self.plot_matrix(key,label=key)
			except Exception as e :
				print(e)
				msg.error("Distribution_store.plot_all","Error when plotting 2D distribution with key {0}... skipping".format(key))
	def print_meta ( self , target_key_ ) :
		something_done = False
		for key in self._distributions_1D :
			if key != target_key_ : continue
			self._distributions_1D[key].print_meta()
			something_done = True
		for key in self._distributions_2D :
			if key != target_key_ : continue
			self._distributions_2D[key].print_meta()
			something_done = True
		if something_done == False :
			msg.warning ( "Distribution_store.print_meta" , "distribution {0} has no key called {1}".format(self._name,target_key_) )
	def print_all ( self ) :
		print(self)
	def copy_2D_local_keys ( self , from_key_ , *args ) :
		if from_key_ not in self._distributions_2D :
			msg.error("HEP_data_utils.data_structures.Distribution_store.copy_2D_local_keys","key {0} does not exist... returning with nothing done".format(from_key_),_verbose_level=0)
			return
		from_dist = self._distributions_2D[from_key_]
		for to_key in args :
			to_dist = self._distributions_2D[to_key]
			for local_key in to_dist._local_keys :
				to_dist.remove_local_key(local_key)
			for local_key in from_dist._local_keys :
				to_dist.set_local_key ( local_key , from_dist._local_key_indices[local_key][0] , from_dist._local_key_indices[local_key][1] )
