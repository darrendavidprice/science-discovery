# ====================================================================================================
#  Brief: data structures which hold the contents and metadata associated with distributions
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================

import numpy as np
import sys, configparser, yaml
from copy import copy

import HEP_data_utils.messaging as msg
import HEP_data_utils.plotting as plotter


#  Brief: store the data contained within a submission.yaml file for a whole dataset
class SubmissionFileMeta (object) :
	def clear (self) :
		self._additional_resources = []
		self._comment = ""
		self._hepdata_doi = ""
	def __init__ (self) :
		self.clear()
	def __type__ ( self ) :
		return "HEP_data_utils.data_structures.SubmissionFileMeta"
	def __str__ ( self ) :
		ret = "\033[1msubmission-file dataset doi:\033[0m " + self._hepdata_doi 
		ret = ret + "\n\033[1msubmission-file dataset comment:\033[0m  " + self._comment
		for v in self._additional_resources : ret = ret + "\n\033[1msubmission-file dataset additional info:\033[0m  {0}".format(v)
		return ret
	def additional_resources (self) : return copy(self._additional_resources)
	def hepdata_doi (self) : return copy(self._hepdata_doi)
	def comment (self) : return copy(self._comment)


#  Brief: store the data contained within a submission.yaml file for a specific distribution
class SubmissionFileTable (object) :
	def clear (self) :
		self._name = ""
		self._table_doi = ""
		self._location = ""
		self._description = ""
		self._data_file = ""
		self._keywords = []
		self._table_metadata = {}
	def __init__ (self) :
		self.clear()
	def __type__ ( self ) :
		return "HEP_data_utils.data_structures.SubmissionFileTable"
	def __str__ ( self ) :
		ret = "\033[1msubmission-file table name:\033[0m " + self._name 
		ret = ret + "\n\033[1msubmission-file table description:\033[0m  " + self._description
		ret = ret + "\n\033[1msubmission-file table doi:\033[0m  " + self._table_doi
		ret = ret + "\n\033[1msubmission-file table location:\033[0m  " + self._table_doi
		ret = ret + "\n\033[1msubmission-file table data-file:\033[0m  " + self._data_file
		ret = ret + "\n\033[1msubmission-file table keywords:\033[0m  "
		for name, value in self._keywords : ret = ret + "\n\t{0}:  {1}".format(name,value)
		ret = ret + "\n\033[1msubmission-file table metadata:\033[0m  "
		for name, value in self._table_metadata.items() : ret = ret + "\n\t{0}:  {1}".format(name,value)
		return ret
	def name (self) : return copy(self._name)
	def table_doi (self) : return copy(self._table_doi)
	def location (self) : return copy(self._location)
	def description (self) : return copy(self._description)
	def data_file (self) : return copy(self._data_file)
	def keywords (self) : return copy(self._keywords)
	def table_metadata (self) : return copy(self._table_metadata)


#  Brief: store the data of an independent_variable as defined in a HEPData table
class IndependentVariable (object) :
	def clear (self) :
		self._name = ""
		self._units = ""
		self._bin_edges = np.empty(shape=(0))
		self._bin_labels = []
	def __init__ (self) :
		self.clear()
	def __type__ (self) :
		return "HEP_data_utils.data_structures.IndependentVariable"
	def __len__ (self) :
		return len(self._bin_edges) - 1
	def __str__ (self) :
		ret = "Independent variable < {0} > ".format(self._name)
		if len(self._units) > 0 : ret = ret + "[ units = {0} ] ".format(self._units)
		ret = ret + "with {0} values".format(len(self._bin_edges)-1)
		ret = ret + "\n-  Bin labels are {0}".format(self._bin_labels)
		ret = ret + "\n-  Bin edges are {0}".format(self._bin_edges)
		return ret
	def name (self) : return copy(self._name)
	def units (self) : return copy(self._units)
	def bin_edges (self) : return copy(self._bin_edges)
	def bin_labels (self) : return copy(self._bin_labels)
	def n_bins (self) : return len(self)
	def set_bin_labels ( self , labels_ ) :
		new_length = len(labels_)
		self._bin_labels = labels_
		self._bin_edges = np.zeros(shape=(new_length+1))
		for i in range(0,new_length+1) : self._bin_edges[i] = float(i) - 0.5


#  Brief: store the data of an dependent_variable as defined in a HEPData table
class DependentVariable (object) :
	def clear (self) :
		self._name = ""
		self._units = ""
		self._qualifiers = []
		self._values = np.empty(shape=(0))
		self._symerrors = {}
		self._asymerrors_up = {}
		self._asymerrors_dn = {}
	def __init__ (self) :
		self.clear()
	def __type__ (self) :
		return "HEP_data_utils.data_structures.DependentVariable"
	def __str__ (self) :
		ret = "Dependent variable < {0} > ".format(self._name)
		if len(self._units) > 0 : ret = ret + "[ units = {0} ] ".format(self._units)
		ret = ret + "with {0} values".format(len(self._values))
		for (name, value) in self._qualifiers :
			if len(value) > 0 : ret = ret + "\n-  Qualifier: {0} = {1}".format(name,value)
			else : ret = ret + "\n-  Qualifier: {0}".format(name)
		ret = ret + "\n-  Values are {0}".format(self._values)
		for name, values in self._symerrors.items() : ret = ret + "\n-  Symmetric error {0} with values  {1}".format(name,values)
		for key in self._asymerrors_up :
			ret = ret + "\n-  Asymmetric error {0} with values UP: {1}".format(key,self._asymerrors_up[key]) + "\n-"
			for i in range(len(key)) : ret = ret + " "
			ret = ret + "                              DOWN: {0}".format(self._asymerrors_dn[key])
		return ret
	def __len__ (self) : return len(self._values)
	def is_valid (self) :
		n_values = len(self._values)
		if n_values == 0 : return False, "No values provided"
		for key, errors in self._symerrors.items() :
			if len(errors) == n_values : continue
			return False, "Symmetric error {0} has length {1} where {2} was expected".format(key,len(errors),n_values)
		for key, errors in self._asymerrors_up.items() :
			if len(errors) == n_values : continue
			return False, "Upwards asymmetric error {0} has length {1} where {2} was expected".format(key,len(errors),n_values)
		for key, errors in self._asymerrors_dn.items() :
			if len(errors) == n_values : continue
			return False, "Downwards asymmetric error {0} has length {1} where {2} was expected".format(key,len(errors),n_values)
		return True, ""
	def name (self) : return copy(self._name)
	def units (self) : return copy(self._units)
	def qualifiers (self) : return copy(self._qualifiers)
	def values (self) : return copy(self._values)
	def n_bins (self) : return len(self)
	def symerrors (self, key_=None) :
		if key_ is None : return self._symerrors
		if key_ not in self._symerrors :
			msg.error("DependentVariable.symerrors","Key {0} not found in self._symerrors... returning None.".format(key_),verbose_level=0)
			return None
		return self._symerrors[key_]
	def asymerrors_up (self, key_=None) :
		if key_ is None : return self._asymerrors_up
		if key_ not in self._asymerrors_up :
			msg.error("DependentVariable.asymerrors_up","Key {0} not found in self._asymerrors_up... returning None.".format(key_),verbose_level=0)
			return None
		return self._asymerrors_up[key_]
	def asymerrors_dn (self, key_=None) :
		if key_ is None : return self._asymerrors_dn
		if key_ not in self._asymerrors_dn :
			msg.error("DependentVariable.asymerrors_dn","Key {0} not found in self._asymerrors_dn... returning None.".format(key_),verbose_level=0)
			return None
		return self._asymerrors_dn[key_]


#  Brief: store a HEPData table
class HEPDataTable (object) :
	def clear (self) :
		self._submission_file_meta = None
		self._submission_file_table = None
		self._dep_var = DependentVariable()
		self._indep_vars = []
	def __init__ (self) :
		self.clear()
	def __type__ (self) :
		return "HEP_data_utils.data_structures.HEPDataTable"
	def __str__ (self) :
		ret = ""
		if self._submission_file_meta : ret = ret + str(self._submission_file_meta) + "\n"
		if self._submission_file_table : ret = ret + str(self._submission_file_table) + "\n"
		ret = ret + "\033[1m\033[95mDEPENDENT VARIABLE:\033[0m\n" + str(self._dep_var)
		ret = ret + "\n\033[1m\033[95mINDEPENDENT VARIABLES:\033[0m"
		for v in self._indep_vars : ret = ret + "\n" + str(v)
		return ret
	def submission_file_meta (self) : return copy(self._submission_file_meta)
	def submission_file_table (self) : return copy(self._submission_file_table)
	def dep_var (self) : return copy(self._dep_var)
	def n_bins (self) : return self._dep_var.n_bins()
	def indep_vars (self) : return copy(self._indep_vars)
	def n_indep_vars (self) : return len(self._indep_vars)
	def is_valid (self) :
		n_values = len(self._dep_var)
		if n_values == 0 : return False, "self._dep_var has zero length"
		product_of_bins = 1
		for indep_var in self._indep_vars : product_of_bins = product_of_bins * len(indep_var)
		if product_of_bins != n_values :
			for indep_var in self._indep_vars :
				if indep_var.n_bins() == n_values : continue
				return False, "Independent variable {0} has length {1} where {2} was expected".format(indep_var.name(),indep_var.n_bins(),n_values)
		return self._dep_var.is_valid()
	def values (self) : return self._dep_var.values()


#  Brief: store a number of HEPData tables, and be able to manipulate them
class DistributionContainer (object) :
	def clear_entries (self) :
		self._inclusive_distributions = {}
		self._1D_distributions = {}
		self._2D_distributions = {}
		self._ND_distributions = {}
	def clear (self) :
		self.clear_entries()
		self._name = ""
		self._make_matrix_if_possible = True
	def __init__ ( self , name_ = "" ) :
		self.clear()
		self._name = name_
	def __type__ (self) :
		return "HEP_data_utils.data_structures.DistributionContainer"
	def __len__ (self) :
		return len(self._inclusive_distributions) + len(self._1D_distributions) + len(self._2D_distributions) + len(self._ND_distributions)
	def __str__ (self) :
		ret = "DistributionContainer \"{0}\" with the following entries".format(self._name)
		ret = ret + "\n\033[1mINCLUSIVE DISTRIBUTIONS:\033[0m"
		for key, dist in self._inclusive_distributions.items() : ret = ret + "\n   key: \033[95m{0}\033[0m\n      --> name \"{1}\" with {2} bins".format(key,dist._dep_var.name(),dist.n_bins())
		ret = ret + "\n\033[1m1D DISTRIBUTIONS:\033[0m"
		for key, dist in self._1D_distributions.items() : ret = ret + "\n   key: \033[95m{0}\033[0m\n      --> name \"{1}\" with {2} bins".format(key,dist._dep_var.name(),dist.n_bins())
		ret = ret + "\n\033[1m2D DISTRIBUTIONS:\033[0m"
		for key, dist in self._2D_distributions.items() : ret = ret + "\n   key: \033[95m{0}\033[0m\n      --> name \"{1}\" with {2} bins".format(key,dist._dep_var.name(),dist.n_bins())
		ret = ret + "\n\033[1m>=3D DISTRIBUTIONS:\033[0m"
		for key, dist in self._ND_distributions.items() : ret = ret + "\n  key: \033[95m{0}\033[0m\n      --> name \"{1}\" with {2} bins".format(key,dist._dep_var.name(),dist.n_bins())
		return ret
	def __contains__( self , key_ ) :
		return key_ in {**self._inclusive_distributions, **self._1D_distributions, **self._2D_distributions, **self._ND_distributions}
	def get_table ( self , key_ ) :
		for key, table in {**self._inclusive_distributions,**self._1D_distributions,**self._2D_distributions,**self._ND_distributions}.items() :
			if key != key_ : continue
			return table
		return None
	def __getitem__ ( self , key_ ) :
		if key_ not in self :
			raise KeyError("No distribution with key {0} in DistributionContainer {1}".format(key_,self._name))
		return self.get_table(key_)
	def print_keys (self) :
		print(self)
	def print_all (self) :
		print("DistributionContainer \"{0}\" with the following entries".format(self._name))
		for key, dist in self._inclusive_distributions.items() :
			print("\n\033[1m\033[95mINCLUSIVE DISTRIBUTION with key {0}, name {1} and {2} bins\033[0m\n".format(key,dist._dep_var.name(),dist.n_bins()))
			print(dist)
		for key, dist in self._1D_distributions.items() :
			print("\n\033[1m\033[95m1D DISTRIBUTION with key {0}, name {1} and {2} bins\033[0m\n".format(key,dist._dep_var.name(),dist.n_bins()))
			print(dist)
		for key, dist in self._2D_distributions.items() :
			print("\n\033[1m\033[95m2D DISTRIBUTION with key {0}, name {1} and {2} bins\033[0m\n".format(key,dist._dep_var.name(),dist.n_bins()))
			print(dist)
		for key, dist in self._ND_distributions.items() :
			print("\n\033[1m\033[95mND DISTRIBUTION with key {0}, name {1} and {2} bins\033[0m\n".format(key,dist._dep_var.name(),dist.n_bins()))
			print(dist)
	def get_inclusive_keys (self) : return [ key for key in self._inclusive_distributions ]
	def get_1D_keys (self) : return [ key for key in self._1D_distributions ]
	def get_2D_keys (self) : return [ key for key in self._2D_distributions ]
	def get_ND_keys (self) : return [ key for key in self._ND_distributions ]
	def rename_key ( self , old_key_ , new_key_ ) :
		something_done = False
		old_key = r"{0}".format(old_key_)
		new_key = r"{0}".format(new_key_)
		for key in self._inclusive_distributions :
			if old_key != key : continue
			self._inclusive_distributions[new_key_] = self._inclusive_distributions.pop(old_key_)
			msg.info("DistributionContainer.rename_key","Store \"{0}\" renaming inclusive distribution key {1} to {2}".format(self._name,old_key_,new_key_),verbose_level=1)
			something_done = True
		for key in self._1D_distributions :
			if old_key != key : continue
			self._1D_distributions[new_key_] = self._1D_distributions.pop(old_key_)
			msg.info("DistributionContainer.rename_key","Store \"{0}\" renaming 1D distribution key {1} to {2}".format(self._name,old_key_,new_key_),verbose_level=1)
			something_done = True
		for key in self._2D_distributions :
			if old_key != key : continue
			self._2D_distributions[new_key_] = self._2D_distributions.pop(old_key_)
			msg.info("DistributionContainer.rename_key","Store \"{0}\" renaming 2D distribution key {1} to {2}".format(self._name,old_key_,new_key_),verbose_level=1)
			something_done = True
		for key in self._ND_distributions :
			if old_key != key : continue
			self._ND_distributions[new_key_] = self._ND_distributions.pop(old_key_)
			msg.info("DistributionContainer.rename_key","Store \"{0}\" renaming ND distribution key {1} to {2}".format(self._name,old_key_,new_key_),verbose_level=1)
			something_done = True
		if not something_done :
			msg.error("DistributionContainer.rename_key","Store \"{0}\" with nothing done for old_key_={1}, new_key_={2}".format(self._name,old_key_,new_key_),verbose_level=1)
	def load_keys ( self , filename_ ) :
		config = configparser.ConfigParser()
		config.optionxform = str
		try : config.read(filename_)
		except :
			msg.check_verbosity_and_print ( str(sys.exc_info()[0]) , verbose_level=-1 )
			msg.error("HEP_data_utils.data_structures.DistributionContainer","An exception occured when parsing the config file... Continuing with nothing done")
			return
		if "KEYS" not in config.sections() :
			msg.error("HEP_data_utils.data_structures.DistributionContainer","No section titled \"KEYS\" in file {0}".format(filename_))
			return
		keys = config["KEYS"]
		for old_key in keys :
			self.rename_key(old_key,keys[old_key])
		self.print_keys()
	def generate_key ( self , table_ ) :
		table_doi = ""
		if table_._submission_file_table : table_doi = table_._submission_file_table.table_doi()
		hepdata_doi = ""
		if table_._submission_file_meta : table_._submission_file_meta.hepdata_doi()
		name = ""
		if table_._submission_file_table : table_.submission_file_table().name()
		if len(name) == 0 and type(table_._dep_var) is DependentVariable : name = name = table_._dep_var.name()
		key = ""
		if len(table_doi) > 0 : key = table_doi + "||" + name
		if len(key) == 0 and len(hepdata_doi) > 0 : key = hepdata_doi + "||" + name
		if len(key) == 0 : key = name
		if key in self :
			key = key + "-;1"
			while key in self : key = key[:-1] + str(1+int(key[-1:]))
		return r"{0}".format(key)
	def plot ( self , key_ , **kwargs ) :
		for key, dist in self._inclusive_distributions.items() :
			if key != key_ : continue
			try : plotter.plot_inclusive_distribution(dist,**kwargs)
			except Exception as e :
				print(e)
				msg.error("HEP_data_utils.data_structures.DistributionContainer.plot","Error when plotting inclusive distribution with key {0}... skipping".format(key))
		for key, dist in self._1D_distributions.items() :
			if key != key_ : continue
			try : plotter.plot_1D_distribution(dist,**kwargs)
			except Exception as e :
				print(e)
				msg.error("HEP_data_utils.data_structures.DistributionContainer.plot","Error when plotting 1D distribution with key {0}... skipping".format(key))
		for key, dist in self._2D_distributions.items() :
			if key != key_ : continue
			try : plotter.plot_2D_distribution(dist,**kwargs)
			except Exception as e :
				print(e)
				msg.error("HEP_data_utils.data_structures.DistributionContainer.plot","Error when plotting 1D distribution with key {0}... skipping".format(key))
	def plot_all (self) :
		for key in { **self._inclusive_distributions , **self._1D_distributions , **self._2D_distributions } :
			self.plot(key)
	def plot_ratio ( self , key_num_ , key_den_ , **kwargs ) :
		table_num = self._1D_distributions.get(key_num_,None)
		if not table_num :
			msg.error("HEP_data_utils.data_structures.DistributionContainer.plot_ratio","Error when plotting 1D distribution with key {0}... skipping".format(key_num_))
			raise KeyError("key {0} not in {1}".format(key_num_,self._name))
		table_den = self._1D_distributions.get(key_den_,None)
		if not table_den :
			msg.error("HEP_data_utils.data_structures.DistributionContainer.plot_ratio","Error when plotting 1D distribution with key {0}... skipping".format(key_den_))
			raise KeyError("key {0} not in {1}".format(key_den_,self._name))
		try : plotter.plot_ratio(table_num,table_den,**kwargs)
		except Exception as e :
			print(e)
			msg.error("HEP_data_utils.data_structures.DistributionContainer.plot_ratio","Error when plotting {0} / {1} ratio... skipping".format(key_num_,key_den_))



'''
	def rename ( self , old_key_ , new_key_ ) :
		something_done = False
		old_key = r"{0}".format(old_key_)
		new_key = r"{0}".format(new_key_)
		for key in self._distributions_1D :
			if old_key != key : continue
			self._distributions_1D[new_key_] = self._distributions_1D.pop(old_key_)
			msg.info("DistributionContainer.rename","Store \"{0}\" renaming 1D distribution key {1} to {2}".format(self._name,old_key_,new_key_),_verbose_level=0)
			something_done = True
		for key in self._distributions_2D :
			dist_key = key
			if old_key_ == key :
				self._distributions_2D[new_key_] = self._distributions_2D.pop(old_key_)
				msg.info("DistributionContainer.rename","Store \"{0}\" renaming 2D distribution key {1} to {2}".format(self._name,old_key_,new_key_),_verbose_level=0)
				dist_key = new_key_
				something_done = True
			indices = [idx for idx, k2 in enumerate(self._distributions_2D[dist_key]._local_keys) if k2 == old_key_]
			if old_key_ not in indices : continue
			for idx in indices :
				self._distributions_2D[dist_key]._local_keys[idx] = new_key_
				msg.info("DistributionContainer.rename","Store \"{0}\" using 2D distribution key {1}... renaming subkey {2} to {3}".format(self._name,dist_key,old_key_,new_key_),_verbose_level=0)
				something_done = True
		if not something_done :
			msg.error("DistributionContainer.rename","Store \"{0}\" with nothing done for old_key_={1}, new_key_={2}".format(self._name,old_key_,new_key_),_verbose_level=0)
'''


'''
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
'''

'''
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
'''
