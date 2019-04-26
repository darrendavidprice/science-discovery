# ===================================================================================================================================
#  Brief: functions which allow us to load and manipulate HEPdata files, as well as the containers which store their data
#  Author: Stephen Menary (stmenary@cern.ch)
# ===================================================================================================================================


import os, yaml
import HEP_data_utils.messaging as msg
import HEP_data_utils.general_helpers as hlp
from HEP_data_utils.data_structures import *


#  Brief: open a yaml file at the specified path and return a list of it's contents in yaml format
def open_yaml_file ( path_ ) :
	yaml_file = open(path_, 'r')
	data = []
	try :
		for datum in yaml.safe_load_all(yaml_file) :
			msg.info("HEP_data_helpers.open_yaml_file","yaml file opened with entries:",verbose_level=2)
			msg.check_verbosity_and_print(yaml.safe_dump(datum),verbose_level=2)
			data.append(datum)
	except yaml.YAMLError as exc :
		print ( exc )
		msg.fatal("HEP_data_helpers.open_yaml_file","Exception thrown when opening the yaml file (see previous messages)")
	return data


#  Brief: from a yaml map called error_, get the uncertainty source (default labelled err{err_idx_}) and add the pt_idx_'th entry to dep_var_
def get_error_from_yaml_map ( dep_var_ , error_ , pt_idx_ , err_idx_=0 ) :
	key = error_.get("label","err{0}".format(err_idx_))
	if "symerror" in error_ :
		if key not in dep_var_._symerrors :
			msg.info("HEP_data_helpers.get_error_from_yaml_map","Creating symmetric error {0} with length {1}".format(key,len(dep_var_)),verbose_level=2)
			dep_var_._symerrors[key] = np.zeros(shape=(len(dep_var_)))
		dep_var_._symerrors[key][pt_idx_] = float(error_["symerror"])
	elif "asymerror" in error_ :
		err_asymm = error_["asymerror"]
		if key not in dep_var_._asymerrors_up :
			msg.info("HEP_data_helpers.get_error_from_yaml_map","Creating asymmetric error {0} with length {1}".format(key,len(dep_var_)),verbose_level=2)
			dep_var_._asymerrors_up[key] = np.zeros(shape=(len(dep_var_)))
			dep_var_._asymerrors_dn[key] = np.zeros(shape=(len(dep_var_)))
		if "plus" not in err_asymm : msg.error("HEP_data_helpers.get_error_from_yaml_map","No entry named \"plus\" for error \"asymerror\"")
		else : dep_var_._asymerrors_up[key][pt_idx_] = float(err_asymm["plus"])
		if "minus" not in err_asymm : msg.error("HEP_data_helpers.get_error_from_yaml_map","No entry named \"minus\" for error \"asymerror\"")
		else : dep_var_._asymerrors_dn[key][pt_idx_] = float(err_asymm["minus"])
	else :
		print(yaml.safe_dump(error_))
		msg.error("HEP_data_helpers.get_error_from_yaml_map","map does not have an entry called symerror or asymerror")
	return key


#  Brief: set the bins of an independent variable based on its HEPData yaml dict
def get_bins_from_dict ( indep_var_ , n_bins_ ) :
	bin_edges = np.zeros(shape=(1+n_bins_))
	bin_labels = [ "" for i in range(0,n_bins_) ]
	values = indep_var_.get("values",[])
	for i in range(0,len(values)) :
		bin = values[i]
		if bin.get("value",None) != None :
			bin_labels[i] = str(bin["value"])
			bin_lo, bin_hi = float(i)-0.5 , float(i)+0.5
		if "high" in bin and "low" in bin :
			bin_lo, bin_hi = bin["low"], bin["high"]
		if i == 0 :
			bin_edges[0], bin_edges[1] = float(bin_lo), float(bin_hi)
			continue
		bin_edges[i+1] = bin_hi
	return bin_edges, bin_labels


#  Brief: from a yaml map called error_, get the uncertainty source (default labelled err{err_idx_}) and add the pt_idx_'th entry to dep_var_
def get_dependent_variable_from_yaml_map ( dep_var_ , path_="unknown" ) :
	dep_var = DependentVariable()
	dep_header = dep_var_.get("header",None)
	if dep_header :
		dep_var._name = dep_header.get("name","")
		dep_var._units = dep_header.get("units","")
	dep_qualifiers = dep_var_.get("qualifiers",None)
	if dep_qualifiers :
		if type(dep_qualifiers) is not list :
			msg.error("HEP_data_helpers.get_dependent_variable_from_yaml_map","Dependent variable {0} in file {1} has qualifiers stored in type {2} where I was expecting a list".format(dep_var._name,path_,type(dep_qualifiers)),verbose_level=0)
		else :
			for entry in dep_qualifiers :
				if type(entry) is dict :
					name, value = entry.get("name",None), str(entry.get("value",""))
					if name : dep_var._qualifiers.append((name,value))
	for entry in dep_var_.get("values",[]) :
		try : val = float(entry["value"])
		except ValueError as exc :
			msg.error("HEP_data_helpers.get_dependent_variable_from_yaml_map","ValueError ({0}) when loading dependent variable {1} from file {2}".format(exc,dep_var._name,path_),verbose_level=1)
			val = 0
		except KeyError as exc :
			msg.error("HEP_data_helpers.get_dependent_variable_from_yaml_map","KeyError ({0}) when loading dependent variable {1} from file {2}".format(exc,dep_var._name,path_),verbose_level=1)
			val = 0
		dep_var._values = np.append(dep_var._values, val)
	pt_idx = 0
	for entry in dep_var_.get("values",[]) :
		errors = entry.get("errors",[])
		err_idx = 0
		for error in errors :
			get_error_from_yaml_map(dep_var,error,pt_idx,err_idx)
			err_idx = err_idx + 1
		pt_idx = pt_idx + 1
	return dep_var


#  Brief: load yaml contents into a HEPDataTable object within dataset_ (of type DistributionContainer)
def load_distribution_from_yaml ( dataset_ , dep_var_ , indep_vars_ , path_ , **argv ) :
	# Create the table object
	table = HEPDataTable()
	table._submission_file_meta = argv.get("submission_file_metadata",None)
	table._submission_file_table = argv.get("submission_file_table",None)
	# Set the dependent variable
	dep_var = get_dependent_variable_from_yaml_map(dep_var_,path_)
	table._dep_var = dep_var
	# Set the independent variables
	for indep_var_map in indep_vars_ :
		indep_var = IndependentVariable()
		indep_header = indep_var_map.get("header",None)
		if indep_header :
			indep_var._name = indep_header.get("name","")
			indep_var._units = indep_header.get("units","")
		indep_var._bin_edges, indep_var._bin_labels = get_bins_from_dict(indep_var_map,len(dep_var))
		table._indep_vars.append(indep_var)
	# Validate our table
	is_valid, validity_message = table.is_valid()
	if not is_valid :
		msg.error("HEP_data_helpers.load_distribution_from_yaml","Error occured when loading table {0} from file {1}... returning with nothing done.")
		msg.error("HEP_data_helpers.load_distribution_from_yaml",">>>>>>>>>>>>>>")
		msg.error("HEP_data_helpers.load_distribution_from_yaml",validity_message)
		msg.error("HEP_data_helpers.load_distribution_from_yaml","<<<<<<<<<<<<<<")
		return
	# Figure out what key to give it
	key = dataset_.generate_key(table)
	# Add to dataset
	n_dim = len(table._indep_vars)
	if n_dim == 0 : dataset_._inclusive_distributions[key] = table
	elif n_dim == 1 : dataset_._1D_distributions[key] = table
	elif n_dim == 2 : dataset_._2D_distributions[key] = table
	else : dataset_._ND_distributions[key] = table


#  Brief: load yaml contents into HEPDataTable objects within dataset_ (of type DistributionContainer) for each of dep_vars_
def load_distributions_from_yaml ( dataset_ , dep_vars_ , indep_vars_ , path_ , **argv ) :
	for dep_var in dep_vars_ :
		load_distribution_from_yaml ( dataset_ , dep_var , indep_vars_ , path_ , **argv )


#  Brief: load a single yaml file based on the file path
def load_yaml_file ( dataset_ , path_ , **kwargs ) :
	msg.info("HEP_data_helpers.load_yaml_file","Opening yaml file {0}".format(path_),verbose_level=0)
	data = open_yaml_file(path_)
	if len(data) != 1 :
		msg.error("HEP_data_helpers.load_yaml_file","{0} contains {1} tables where I was expecting 1... is the file format correct?".format(path_,len(data)))
		return
	dep_vars = data[0].get("dependent_variables",None)
	if dep_vars is None :
		msg.fatal("HEP_data_helpers.load_yaml_file","{0} contains no dependent_variables as required".format(path_))
		return
	indep_vars = data[0].get("independent_variables",None)
	if indep_vars is None :
		msg.fatal("HEP_data_helpers.load_yaml_file","{0} contains no independent_variables as required".format(path_))
		return
	load_distributions_from_yaml ( dataset_ , dep_vars , indep_vars , path_ , **kwargs )


#  Brief: load all tables specified in a submission.yaml file
def load_submission_file ( dataset_ , path_ ,  **kwargs ) :
	data = open_yaml_file(path_)
	# First get the dataset metadata
	sub_file_meta = SubmissionFileMeta()
	indices_of_metadata = []
	for idx in range(len(data)) :
		dataset_properties = data[idx]
		if "data_file" in dataset_properties : continue
		indices_of_metadata.append(idx)
		additional_resources = dataset_properties.get("additional_resources",[])
		for idx in range(len(additional_resources)) : sub_file_meta._additional_resources.append(additional_resources[idx])
		if "comment" in dataset_properties : sub_file_meta._comment = dataset_properties["comment"]
		if "hepdata_doi" in dataset_properties : sub_file_meta._hepdata_doi = dataset_properties["hepdata_doi"]
	# Now load tables from all remaining entries
	for idx in range(len(data)) :
		if idx in indices_of_metadata : continue
		table = data[idx]
		data_file = hlp.get_directory(path_) + "/" + table["data_file"]
		if not os.path.isfile(data_file) :
			msg.error("HEP_data_helpers.load_submission_file","Submission file asks for a yaml file called {0} but none exists".format(data_file))
			return
		sub_file_table = SubmissionFileTable()
		sub_file_table._data_file = data_file
		sub_file_table._name = table.get("name","")
		sub_file_table._table_doi = table.get("table_doi","")
		sub_file_table._location = table.get("location","")
		sub_file_table._description = table.get("description","")
		sub_file_table._keywords = [(d.get("name",""),d.get("values","")) for d in table.get("keywords",[])]
		load_yaml_file(dataset_,data_file,submission_file_metadata=sub_file_meta,submission_file_table=sub_file_table,**kwargs)


#  Brief: load yaml files based on the file path
def load_dataset ( dataset_ , path_ , **kwargs ) :
	do_recurse = kwargs.get("recurse",True)
	path_ = hlp.remove_subleading(path_,"/")
	if hlp.is_directory(path_) :
		yaml_files = hlp.keep_only_yaml_files(path_,recurse=do_recurse)
		if len(yaml_files) == 0 :
			msg.error("HEP_data_helpers.load_dataset","Directory {0} has no yaml files... returning with nothing done.".format(path_),verbose_level=-1)
			return
		for f in yaml_files : load_dataset(dataset_,path_,**kwargs)
		return
	if not hlp.is_yaml_file(path_) :
		msg.error("HEP_data_helpers.load_dataset","Path {0} is not a directory or yaml file... returning with nothing done.".format(path_),verbose_level=-1)
		return
	if hlp.is_submission_file(path_) :
		load_submission_file(dataset_,path_,**kwargs)
		return
	load_yaml_file(dataset_,path_,**kwargs)


#  Brief: load yaml files based on a directory path or file list
def load_all ( dataset_ , dir_ , **kwargs ) :
	if hlp.is_directory(dir_) :
		for filename in [ dir_+"/"+f for f in os.listdir(dir_) if is_yaml_file(f) ] :
			msg.info("HEP_data_helpers.load_all_yaml_files","Opening yaml file {0}".format(filename),verbose_level=0)
			load_dataset(dataset_,filename,**kwargs)
	elif type(dir_) == list :
		for filename in dir_ :
			if type(filename) != str : continue
			msg.info("HEP_data_helpers.load_all_yaml_files","Opening yaml file {0}".format(filename),verbose_level=0)
			load_dataset(dataset_,filename,**kwargs)
	else :
		msg.error("HEP_data_helpers.load_all_yaml_files","Input {0} is neither a directory nor a list... returning with nothing done".format(dir_),verbose_level=-1)





'''
def set_1D_bins ( distribution_ , indep_vars_ ) :
	if len(indep_vars_) != 1 :
		msg.fatal("HEP_data_helpers.set_1D_bins","distribution {0} has {1} independent_variables but I am only configured to deal with 1".format(distribution_._description,len(indep_vars_)))
	distribution_._bin_values = np.zeros(shape=(1+len(distribution_)))
	distribution_._bin_labels = [ "unlabeled" for i in range(0,len(distribution_)) ]
	for i in range(0,len(indep_vars_[0]["values"])) :
		bin = indep_vars_[0]["values"][i]
		if bin.get("value",None) != None :
			distribution_._bin_labels[i] = bin["value"]
		elif bin.get("high",None) != None and bin.get("low",None) != None :
			bin_lo, bin_hi = bin["low"], bin["high"]
			if i == 0 :
				distribution_._bin_values[0], distribution_._bin_values[1] = bin_lo, bin_hi
				continue
			if bin_hi == distribution_._bin_values[i] : distribution_._bin_values[i+1] = bin_lo
			elif bin_lo == distribution_._bin_values[i] : distribution_._bin_values[i+1] = bin_hi
			else :
				distribution_._has_errors = True
				msg.error("HEP_data_helpers.set_1D_bins","Bin entry {0} for distribution {1} is not continuous from the previous bin which ended at {2}".format(bin,distribution_._description,distribution_._bin_values[i]),verbose_level=-1)
		else :
			distribution_._has_errors = True
			msg.error("HEP_data_helpers.set_1D_bins","Could not interpret bin entry {0} for distribution {1}".format(bin,distribution_._description),verbose_level=-1)


def set_2D_bins ( distribution_ , indep_vars_ ) :
	if len(indep_vars_) != 2 :
		msg.fatal("HEP_data_helpers.set_2D_bins","distribution {0} has {1} independent_variables but I am only configured to deal with 2".format(distribution_._description,len(indep_vars_)))
	distribution_._bin_labels_x = [ "unlabeled" for i in range(0,len(distribution_)) ]
	distribution_._bin_labels_y = [ "unlabeled" for i in range(0,len(distribution_)) ]
	for i in range(0,len(indep_vars_[0]["values"])) :
		bin = indep_vars_[0]["values"][i]
		if bin.get("value",None) != None :
			distribution_._bin_labels_x[i] = bin["value"]
		else :
			msg.fatal("HEP_data_helpers.set_2D_bins","Could not interpret bin entry {0} for distribution {1}".format(bin,distribution_._description))
	for i in range(0,len(indep_vars_[1]["values"])) :
		bin = indep_vars_[1]["values"][i]
		if bin.get("value",None) != None :
			distribution_._bin_labels_y[i] = bin["value"]
		else :
			msg.fatal("HEP_data_helpers.set_2D_bins","Could not interpret bin entry {0} for distribution {1}".format(bin,distribution_._description))

'''
