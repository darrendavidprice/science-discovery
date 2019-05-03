# ===================================================================================================================================
#  Brief: functions which allow us to load HEPdata files
#  Author: Stephen Menary (stmenary@cern.ch)
# ===================================================================================================================================


import os, yaml
from natsort import natsorted, ns
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
def get_bins_from_dict ( indep_var_ , n_bins_ , **kwargs ) :
	bin_centers = np.zeros(shape=(n_bins_))
	bin_labels = [ "" for i in range(0,n_bins_) ]
	bin_widths_lo = np.zeros(shape=(n_bins_))
	bin_widths_hi = np.zeros(shape=(n_bins_))
	values = indep_var_.get("values",[])
	if kwargs.get("force_labels",False) :
		for i in range(0,len(values)) :
			bin = values[i]
			bin_center, bin_lo, bin_hi = float(i), float(i)-0.5 , float(i)+0.5
			if bin.get("value",None) != None :
				bin_labels[i] = str(bin["value"])
			if "high" in bin and "low" in bin :
				bin_labels[i] = "{0}[{1},{2}]".format(bin_labels[i],bin["low"],bin["high"])
			bin_centers[i] = bin_center
			bin_widths_lo[i] = bin_center - bin_lo
			bin_widths_hi[i] = bin_hi - bin_center
	else :
		for i in range(0,len(values)) :
			bin = values[i]
			if bin.get("value",None) != None :
				bin_labels[i] = str(bin["value"])
				bin_lo, bin_hi = float(i)-0.5 , float(i)+0.5
			if "high" in bin and "low" in bin :
				bin_lo, bin_hi = bin["low"], bin["high"]
			try : bin_center = float(bin["value"],0.5*(bin_lo+bin_hi))
			except : bin_center = 0.5*(bin_lo+bin_hi)
			bin_centers[i] = bin_center
			bin_widths_lo[i] = bin_center - bin_lo
			bin_widths_hi[i] = bin_hi - bin_center
	return bin_centers, bin_widths_lo, bin_widths_hi, bin_labels


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
			val = None
		except KeyError as exc :
			msg.error("HEP_data_helpers.get_dependent_variable_from_yaml_map","KeyError ({0}) when loading dependent variable {1} from file {2}".format(exc,dep_var._name,path_),verbose_level=1)
			val = None
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


#  Brief: reorder bins into a matrix for 2D distribution
def reformat_2D_bins_as_matrix_using_labels ( dist_2D_ ) :
	if type(dist_2D_) != HEPDataTable :
		msg.fatal("HEP_data_helpers.reformat_2D_bins_as_matrix_using_labels","argument must be of type HEPDataTable")
	dep_var = dist_2D_._dep_var
	if len(dist_2D_._indep_vars) != 2 :
		msg.fatal("HEP_data_helpers.reformat_2D_bins_as_matrix_using_labels","HEPDataTable {0} has {1} independent_variable where 2 are required".format(dep_var._name,len(dist_2D_._indep_vars)))
	n_vals = len(dep_var)
	old_bin_labels_x = dist_2D_._indep_vars[0]._bin_labels
	old_bin_labels_y = dist_2D_._indep_vars[1]._bin_labels
	old_n_bins_x = len(old_bin_labels_x)
	old_n_bins_y = len(old_bin_labels_y)
	if n_vals == old_n_bins_x == old_n_bins_y :
		bin_labels_x = [y for y in {x for x in old_bin_labels_x}]
		bin_labels_x = natsorted(bin_labels_x, alg=ns.IGNORECASE)
		bin_labels_y = [y for y in {x for x in old_bin_labels_y}]
		bin_labels_y = natsorted(bin_labels_y, alg=ns.IGNORECASE)
		new_n_bins_x = len(bin_labels_x)
		new_n_bins_y = len(bin_labels_y)
		new_values = np.array(np.zeros(shape=(new_n_bins_x,new_n_bins_y)))
		for x,y,v in zip(old_bin_labels_x,old_bin_labels_y,dep_var._values) :
			new_values[bin_labels_x.index(x),bin_labels_y.index(y)] = v
		dep_var._values = new_values
		for key, values in dep_var._symerrors.items() :
			new_error = np.array(np.zeros(shape=(new_n_bins_x,new_n_bins_y)))
			for x,y,v in zip(old_bin_labels_x,old_bin_labels_y,values) :
				new_error[bin_labels_x.index(x),bin_labels_y.index(y)] = v
			dep_var._symerrors[key] = new_error
		for key, values in dep_var._asymerrors_up.items() :
			new_error = np.array(np.zeros(shape=(new_n_bins_x,new_n_bins_y)))
			for x,y,v in zip(old_bin_labels_x,old_bin_labels_y,values) :
				new_error[bin_labels_x.index(x),bin_labels_y.index(y)] = v
			dep_var._asymerrors_up[key] = new_error
		for key, values in dep_var._asymerrors_dn.items() :
			new_error = np.array(np.zeros(shape=(new_n_bins_x,new_n_bins_y)))
			for x,y,v in zip(old_bin_labels_x,old_bin_labels_y,values) :
				new_error[bin_labels_x.index(x),bin_labels_y.index(y)] = v
			dep_var._asymerrors_dn[key] = new_error
		dist_2D_._indep_vars[0].set_bin_labels(bin_labels_x)
		dist_2D_._indep_vars[1].set_bin_labels(bin_labels_y)
		return
	if n_vals == old_n_bins_x*old_n_bins_y :
		new_values = np.array(np.zeros(shape=(old_n_bins_x,old_n_bins_y)))
		for x_idx in enumerate(old_bin_labels_x) :
			for y_idx in enumerate(old_bin_labels_y) :
				new_values[x_idx,y_idx] = values[ x_idx + old_n_bins_x*y_idx ]
		dep_var._values = new_values
		for key, values in dist_2D_._symerrors.items() :
			new_error = np.array(np.zeros(shape=(new_n_bins_x,new_n_bins_y)))
			for x_idx in enumerate(old_bin_labels_x) :
				for y_idx in enumerate(old_bin_labels_y) :
					new_error[x_idx,y_idx] = values[ x_idx + old_n_bins_x*y_idx ]
			dep_var._symerrors[key] = new_error
		for key, values in dist_2D_._asymerrors_up.items() :
			new_error = np.array(np.zeros(shape=(new_n_bins_x,new_n_bins_y)))
			for x_idx in enumerate(old_bin_labels_x) :
				for y_idx in enumerate(old_bin_labels_y) :
					new_error[x_idx,y_idx] = values[ x_idx + old_n_bins_x*y_idx ]
			dep_var._asymerrors_up[key] = new_error
		for key, values in dist_2D_._asymerrors_dn.items() :
			new_error = np.array(np.zeros(shape=(new_n_bins_x,new_n_bins_y)))
			for x_idx in enumerate(old_bin_labels_x) :
				for y_idx in enumerate(old_bin_labels_y) :
					new_error[x_idx,y_idx] = values[ x_idx + old_n_bins_x*y_idx ]
			dep_var._asymerrors_dn[key] = new_error
		return
	msg.warning("HEP_data_helpers.reformat_2D_bins_as_matrix_using_labels","HEPDataTable {0} is not a matrix... assuming it is a non-factorisable distribution and returning with nothing done".format(dep_var._name),verbose_level=0)
	return


#  Brief: reorder bins into a matrix for 2D distribution
def reformat_2D_bins_as_matrix ( dist_2D_ ) :
	reformat_2D_bins_as_matrix_using_labels(dist_2D_)
'''
	use_labels = True
	for indep_var in dist_2D_._indep_vars :
		for label in indep_var._bin_labels :
			label = str(label)
			if len(label) > 0 : continue 
			use_labels = False
	if use_labels :
'''


#  Brief: remove all bins from a 1D table which include a "None" entry
def remove_None_entries_from_1D_table ( table_ ) :
		# check dimensions and number of Nones
	if table_.n_indep_vars() != 1 : return
	dep_var = table_._dep_var
	indep_var = table_._indep_vars[0]
	old_values, old_labels, old_centers, old_bw_lo, old_bw_hi = dep_var._values, indep_var._bin_labels, indep_var._bin_centers, indep_var._bin_widths_lo, indep_var._bin_widths_hi
	num_Nones = len([x for x in old_values if x is None])
	if num_Nones == 0 : return
		# make sure bin labels are sensibly configured for a discontinuous distribution
	if len(old_values) != len(old_labels) : 
		msg.error("HEP_data_utils.HEP_data_helpers.remove_None_entries_from_1D_table","Cannot interpret binning when len(values) = {0} but len(labels) = {1}".format(len(old_values),len(old_labels)),verbose_level=1)
		return
	for i in range(len(old_labels)) :
		label = old_labels[i]
		if len(label) > 0 : continue
		old_labels[i] = "[{0},{1}]".format(old_centers[i]-old_bw_lo[i],old_centers[i]+old_bw_hi[i])
		# set new values
	None_indices = []
	new_labels = []
	new_bw_lo , new_bw_hi = np.full(shape=(len(old_bw_lo)-num_Nones),fill_value=0.5), np.full(shape=(len(old_bw_hi)-num_Nones),fill_value=0.5)
	new_centers = np.zeros(shape=(len(old_centers)-num_Nones))
	for i in range(len(new_centers)) :
		new_centers[i] = i
		new_bw_lo[i], new_bw_hi[i] = 0.5, 0.5
	indep_var._bin_centers, indep_var._bin_widths_lo, indep_var._bin_widths_hi = new_centers, new_bw_lo, new_bw_hi
	for i in range(len(old_values)) :
		if old_values[i] == None :
			None_indices.append(i)
			continue
		new_labels.append(old_labels[i])
	dep_var._values = np.delete(old_values,None_indices)
	indep_var._bin_labels = new_labels
		# set new errors
	for key in dep_var._symerrors :
		dep_var._symerrors[key] = np.delete(dep_var._symerrors[key],None_indices)
	for key in dep_var._asymerrors_up :
		dep_var._asymerrors_up[key] = np.delete(dep_var._asymerrors_up[key],None_indices)
	for key in dep_var._asymerrors_dn :
		dep_var._asymerrors_dn[key] = np.delete(dep_var._asymerrors_dn[key],None_indices)


#  Brief: load yaml contents into a HEPDataTable object within dataset_ (of type DistributionContainer)
def load_distribution_from_yaml ( dataset_ , dep_var_ , indep_vars_ , path_ , **kwargs ) :
	# Create the table object
	table = HEPDataTable()
	table._submission_file_meta = kwargs.get("submission_file_metadata",None)
	table._submission_file_table = kwargs.get("submission_file_table",None)
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
		indep_var._bin_centers, indep_var._bin_widths_lo, indep_var._bin_widths_hi, indep_var._bin_labels = get_bins_from_dict(indep_var_map,len(dep_var))
		table._indep_vars.append(indep_var)
	# If our table has 'None' entries, we want to remove them
	remove_None_entries_from_1D_table(table)
	# Figure out what key to give it
	key = dataset_.generate_key(table)
	# Validate our table
	is_valid, validity_message = table.is_valid()
	if not is_valid :
		msg.error("HEP_data_helpers.load_distribution_from_yaml","Error occured when loading table {0} from file {1}... returning with nothing done.".format(key,path_))
		msg.error("HEP_data_helpers.load_distribution_from_yaml",">>>>>>>>>>>>>>")
		msg.error("HEP_data_helpers.load_distribution_from_yaml",validity_message)
		msg.error("HEP_data_helpers.load_distribution_from_yaml","<<<<<<<<<<<<<<")
		return
	# Reformat 2D distribution bins into a matrix
	n_dim = len(table._indep_vars)
	if n_dim == 2 and dataset_._make_matrix_if_possible : reformat_2D_bins_as_matrix(table)
	# Add to dataset
	if n_dim == 0 : dataset_._inclusive_distributions[key] = table
	elif n_dim == 1 : dataset_._1D_distributions[key] = table
	elif n_dim == 2 : dataset_._2D_distributions[key] = table
	else : dataset_._ND_distributions[key] = table


#  Brief: load yaml contents into HEPDataTable objects within dataset_ (of type DistributionContainer) for each of dep_vars_
def load_distributions_from_yaml ( dataset_ , dep_vars_ , indep_vars_ , path_ , **kwargs ) :
	for dep_var in dep_vars_ :
		load_distribution_from_yaml ( dataset_ , dep_var , indep_vars_ , path_ , **kwargs )


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
def load_all_yaml_files ( dataset_ , path_ , **kwargs ) :
	do_recurse = kwargs.get("recurse",True)
	path_ = hlp.remove_subleading(path_,"/")
	if hlp.is_directory(path_) :
		yaml_files = hlp.keep_only_yaml_files(path_,recurse=do_recurse)
		if len(yaml_files) == 0 :
			msg.error("HEP_data_helpers.load_all_yaml_files","Directory {0} has no yaml files... returning with nothing done.".format(path_),verbose_level=-1)
			return
		for f in yaml_files : load_all_yaml_files(dataset_,f,**kwargs)
		return
	if not hlp.is_yaml_file(path_) :
		msg.error("HEP_data_helpers.load_all_yaml_files","Path {0} is not a directory or yaml file... returning with nothing done.".format(path_),verbose_level=-1)
		return
	if hlp.is_submission_file(path_) :
		load_submission_file(dataset_,path_,**kwargs)
		return
	load_yaml_file(dataset_,path_,**kwargs)


#  Brief: load yaml files based on a directory path or file list
def load_yaml_files_from_list ( dataset_ , dir_ , **kwargs ) :
	if hlp.is_directory(dir_) :
		for filename in [ dir_+"/"+f for f in os.listdir(dir_) if is_yaml_file(f) ] :
			msg.info("HEP_data_helpers.load_yaml_files_from_list","Opening yaml file {0}".format(filename),verbose_level=0)
			load_all_yaml_files(dataset_,filename,**kwargs)
	elif type(dir_) == list :
		for filename in dir_ :
			if type(filename) != str : continue
			msg.info("HEP_data_helpers.load_yaml_files_from_list","Opening yaml file {0}".format(filename),verbose_level=0)
			load_all_yaml_files(dataset_,filename,**kwargs)
	else :
		msg.error("HEP_data_helpers.load_yaml_files_from_list","Input {0} is neither a directory nor a list... returning with nothing done".format(dir_),verbose_level=-1)





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
