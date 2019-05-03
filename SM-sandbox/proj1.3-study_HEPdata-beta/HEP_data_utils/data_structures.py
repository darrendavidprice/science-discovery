# ====================================================================================================
#  Brief: data structures which hold the contents and metadata associated with distributions
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================

import numpy as np
from copy import copy

import HEP_data_utils.messaging as msg


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
		self._bin_labels = []
		self._bin_centers = np.empty(shape=(0))
		self._bin_widths_lo = np.empty(shape=(0))
		self._bin_widths_hi = np.empty(shape=(0))
	def __init__ (self) :
		self.clear()
	def __type__ (self) :
		return "HEP_data_utils.data_structures.IndependentVariable"
	def __len__ (self) :
		return len(self._bin_centers)
	def __str__ (self) :
		ret = "Independent variable < {0} > ".format(self._name)
		if len(self._units) > 0 : ret = ret + "[ units = {0} ] ".format(self._units)
		ret = ret + "with {0} values".format(len(self))
		ret = ret + "\n-  Bin labels are {0}".format(self._bin_labels)
		ret = ret + "\n-  Bin centres are {0}".format(self._bin_centers)
		ret = ret + "\n-  Bin widths (low)  are {0}".format(self._bin_widths_lo)
		ret = ret + "\n-  Bin widths (high) are {0}".format(self._bin_widths_hi)
		return ret
	def name (self) : return copy(self._name)
	def units (self) : return copy(self._units)
	def bin_centers (self) : return copy(self._bin_centers)
	def bin_edges (self) : return copy(self._bin_edges)
	def bin_labels (self) : return copy(self._bin_labels)
	def n_bins (self) : return len(self)
	def set_bin_labels ( self , labels_ ) :
		new_length = len(labels_)
		self._bin_labels = labels_
		self._bin_centers = np.zeros(shape=(new_length))
		self._bin_widths_lo = np.zeros(shape=(new_length))
		self._bin_widths_hi = np.zeros(shape=(new_length))
		for i in range(0,new_length) :
			self._bin_centers[i] = i
			self._bin_widths_lo[i] = 0.5
			self._bin_widths_hi[i] = 0.5


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
	def __len__ (self) : return len(self._values.flatten())
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
		n_axes = self.n_indep_vars()
		if n_values == 0 : return False, "self._dep_var has zero length"
		shape = self._dep_var._values.shape
		if len(shape) == 1 and n_axes != 1 :
			for n_values_axis in [ x for x in [ len(indep_var) for indep_var in self._indep_vars ] if x != n_values ] :
				return False, "Independent variable with length {0} where {1} was expected".format(n_values_axis,shape)
		else :
			for i in range(self.n_indep_vars()) :
				indep_var = self._indep_vars[i]
				if indep_var.n_bins() == shape[i] : continue
				return False, "Independent variable {0} (index {1}) has length {2} where {3} was expected".format(indep_var.name(),i,indep_var.n_bins(),shape[i])
		return self._dep_var.is_valid()
	def values (self) : return self._dep_var.values()
