import os, yaml
import general_utils.messaging as msg
import general_utils.helpers as hlp
from HEP_data_utils.data_structures import *


#  Brief: check that file has a yaml extension
def is_yaml_file ( path_ ) :
	return hlp.check_extension(path_,"yaml")


#  Brief: check that file is a HEPdata submission file
def is_submission_file ( path_ ) :
	return path_[-15:] == "submission.yaml"


#  Brief: open a yaml file at the specified path and return a _list_ of it's contents in yaml format
def open_yaml_file ( path_ ) :
	yaml_file = open(path_, 'r')
	data = []
	try :
		for datum in yaml.safe_load_all(yaml_file) :
			msg.info("HEP_data_utils.helpers.open_yaml_file","yaml file opened with entries:",_verbose_level=1)
			msg.check_verbosity_and_print(yaml.safe_dump(datum),_verbose_level=1)
			data.append(datum)
	except yaml.YAMLError as exc :
		print ( exc )
		msg.fatal("HEP_data_utils.helpers.open_yaml_file","Exception thrown when opening the yaml file (see previous messages)")
	return data


#  Brief: from a yaml map called error_, get the err_ix_'th uncertainty source and add the pt_idx_'th entry to distribution_
def get_error_from_yaml_map ( distribution_ , error_ , pt_idx_ , err_idx_=0 ) :
	key = error_.get("label","err{0}".format(err_idx_))
	if "symerror" in error_ :
		if key not in distribution_._symm_errors :
			msg.info("HEP_data_utils.helpers.get_error_from_yaml_map","Creating symmetric error {0} with length {1}".format(key,len(distribution_)),_verbose_level=1)
			distribution_._symm_errors[key] = np.zeros(shape=(len(distribution_)))
		distribution_._symm_errors[key][pt_idx_] = error_["symerror"]
	elif "asymerror" in error_ :
		err_asymm = error_["asymerror"]
		if key not in distribution_._asymm_errors_up :
			msg.info("HEP_data_utils.helpers.get_error_from_yaml_map","Creating asymmetric error {0} with length {1}".format(key,len(distribution_)),_verbose_level=1)
			distribution_._asymm_errors_up[key] = np.zeros(shape=(len(distribution_)))
			distribution_._asymm_errors_down[key] = np.zeros(shape=(len(distribution_)))
		if "plus" not in err_asymm :
			msg.fatal("HEP_data_utils.helpers.get_error_from_yaml_map","No entry named \"plus\" for error \"asymerror\"")
		else :
			distribution_._asymm_errors_up[key][pt_idx_] = err_asymm["plus"]
		if err_asymm.get("minus",None) == None :
			msg.fatal("HEP_data_utils.helpers.get_error_from_yaml_map","No entry named \"minus\" for error \"asymerror\"")
		else :
			distribution_._asymm_errors_down[key][pt_idx_] = err_asymm["minus"]
	else :
		print(error_)
		msg.fatal("HEP_data_utils.helpers.get_error_from_yaml_map","map does not have an entry called symerror or asymerror")
	return key


def set_1D_bins ( distribution_ , indep_vars_ ) :
	if len(indep_vars_) != 1 :
		msg.fatal("HEP_data_utils.helpers.set_1D_bins","distribution {0} has {1} independent_variables but I am only configured to deal with 1".format(distribution_._description,len(indep_vars_)))
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
				msg.error("HEP_data_utils.helpers.set_1D_bins","Bin entry {0} for distribution {1} is not continuous from the previous bin which ended at {2}".format(bin,distribution_._description,distribution_._bin_values[i]),verbose_level=-1)
		else :
			distribution_._has_errors = True
			msg.error("HEP_data_utils.helpers.set_1D_bins","Could not interpret bin entry {0} for distribution {1}".format(bin,distribution_._description),verbose_level=-1)


def set_2D_bins ( distribution_ , indep_vars_ ) :
	if len(indep_vars_) != 2 :
		msg.fatal("HEP_data_utils.helpers.set_2D_bins","distribution {0} has {1} independent_variables but I am only configured to deal with 2".format(distribution_._description,len(indep_vars_)))
	distribution_._bin_labels_x = [ "unlabeled" for i in range(0,len(distribution_)) ]
	distribution_._bin_labels_y = [ "unlabeled" for i in range(0,len(distribution_)) ]
	for i in range(0,len(indep_vars_[0]["values"])) :
		bin = indep_vars_[0]["values"][i]
		if bin.get("value",None) != None :
			distribution_._bin_labels_x[i] = bin["value"]
		else :
			msg.fatal("HEP_data_utils.helpers.set_2D_bins","Could not interpret bin entry {0} for distribution {1}".format(bin,distribution_._description))
	for i in range(0,len(indep_vars_[1]["values"])) :
		bin = indep_vars_[1]["values"][i]
		if bin.get("value",None) != None :
			distribution_._bin_labels_y[i] = bin["value"]
		else :
			msg.fatal("HEP_data_utils.helpers.set_2D_bins","Could not interpret bin entry {0} for distribution {1}".format(bin,distribution_._description))


def regularise_bins ( dist_2D_ ) :
	if not isinstance(dist_2D_,Distribution_2D) : msg.fatal("HEP_data_utils.helpers.regularise_bins","argument must be of type Distribution_2D")
	n_vals = len(dist_2D_._values)
	n_bins_x = len(dist_2D_._bin_labels_x)
	n_bins_y = len(dist_2D_._bin_labels_y)
	if n_vals == n_bins_x == n_bins_y :
		bin_labels_x = [y for y in { x for x in dist_2D_._bin_labels_x }]
		bin_labels_x.sort()
		bin_labels_y = [y for y in { x for x in dist_2D_._bin_labels_y }]
		bin_labels_y.sort()
		if bin_labels_x != bin_labels_y : return
		new_n_bins_x = len(bin_labels_x)
		new_n_bins_y = len(bin_labels_y)
		new_values = np.array(np.zeros(shape=(new_n_bins_x,new_n_bins_y)))
		for x,y,v in zip(dist_2D_._bin_labels_x,dist_2D_._bin_labels_y,dist_2D_._values) :
			new_values[bin_labels_x.index(x),bin_labels_y.index(y)] = v
		dist_2D_._values = new_values
		dist_2D_._bin_labels_x = bin_labels_x
		dist_2D_._bin_labels_y = bin_labels_y
	elif n_vals == n_bins_x*n_bins_y :
		new_values = np.array(np.zeros(shape=(n_bins_x,n_bins_y)))
		for x_idx in enumerate(dist_2D_._bin_labels_x) :
			for y_idx in enumerate(dist_2D_._bin_labels_y) :
				v = dist_2D_._values[ x_idx + n_bins_x*y_idx ]
				new_values[x_idx,y_idx] = v
		dist_2D_._values = new_values
	else :
		msg.error("HEP_data_utils.helpers.regularise_bins","function not implemented for this type of matrix",_verbose_level=0)


def load_distributions_from_yaml ( dataset_ , dep_vars_ , indep_vars_ , path_ , **argv ) :
	n_dim_ = argv.get("n_dim_",1)
	extra_info_global = argv.get("metadata_global_",{})
	extra_info_local = argv.get("metadata_global_",{})
	for var_idx in range(0,len(dep_vars_)) :
		dep_var = dep_vars_[var_idx]
		distribution = Distribution()
		dist_key = "|"
		if extra_info_global.get("table_doi",None) != None : dist_key = dist_key + str(extra_info_global["table_doi"]) + "|"
		if extra_info_global.get("data_file",None) != None : dist_key = dist_key + str(extra_info_global["data_file"]) + "|"
		else : dist_key = dist_key + path_ + "|"
		if n_dim_ == 1 : distribution = Distribution_1D()
		if n_dim_ == 2 : distribution = Distribution_2D()
		distribution._description = dep_var["header"].get("name","unknown")
		distribution._name = distribution._description
		distribution._dep_var = dep_var["header"].get("name","unknown")
		distribution._indep_var = indep_vars_[0]["header"].get("name","unknown")
		distribution._units = dep_var["header"].get("units","unknown")
		for key in extra_info_local :
			if key == "dependent_variables" or key == "independent_variables" : continue
			distribution._meta["LOCAL::"+key] = extra_info_local[key]
		for key in extra_info_global : distribution._meta["GLOBAL::"+key] = extra_info_global[key]
		for key in dep_var :
			if key == "values" : continue
			if key == "errors" : continue
			distribution._meta["LOCAL::DEP_VARS::"+key] = dep_var[key]
		pt_idx = 0
		for entry in dep_var["values"] :
			try :
				distribution._values = np.append(distribution._values, entry["value"])
			except KeyError as exc :
				msg.error("HEP_data_utils.helpers.load_distributions_from_yaml","KeyError: {0}".format(exc),_verbose_level=-1)
				msg.fatal("HEP_data_utils.helpers.load_distributions_from_yaml","Entry with no \"value\" when trying to create distribution {0} in file {1}".format(distribution._description,path_))
		for entry in dep_var["values"] :
			try :
				errors = entry["errors"]
			except KeyError as exc :
				msg.error("HEP_data_utils.helpers.load_distributions_from_yaml","KeyError: {0}".format(exc),_verbose_level=1)
				msg.warning("HEP_data_utils.helpers.load_distributions_from_yaml","Entry with no \"errors\" when trying to create distribution {0} in file {1}... Assuming there are none".format(distribution._description,path_),_verbose_level=1)
				errors = []
			err_idx = 0
			for error in errors :
				get_error_from_yaml_map(distribution,error,pt_idx,err_idx)
				err_idx = err_idx + 1
			pt_idx = pt_idx + 1
		for var_idx in range(0,len(indep_vars_)) :
			indep_var = indep_vars_[var_idx]
			for key in indep_var :
				if key == "values" : continue
				distribution._meta["LOCAL::INDEP_VARS::"+key] = indep_var[key]
		expected_size = len(distribution._values)
		if n_dim_ == 1 : set_1D_bins(distribution,indep_vars_)
		elif n_dim_ == 2 :
			set_2D_bins(distribution,indep_vars_)
			regularise_bins(distribution)
		else : msg.fatal("HEP_data_utils.helpers.load_distributions_from_yaml","number of bin dimensions is {0} but I can only handle 1 or 2".format(n_dim_))
		for error in [distribution._symm_errors,distribution._asymm_errors_up,distribution._asymm_errors_down] :
			for key in error :
				this_err_size = len(error[key])
				if this_err_size == expected_size : continue
				msg.fatal("HEP_data_utils.helpers.load_distributions_from_yaml","error source {0} has length {1} for distribution [{2}] where {3} was expected".format(key,this_err_size,distribution._description,expected_size))
		msg.info("HEP_data_utils.helpers.load_distributions_from_yaml","yaml file loaded with the following entries",_verbose_level=0)
		dist_key = dist_key + str(distribution._name) + "|"
		if msg.VERBOSE_LEVEL >= 0 : print(distribution)
		if n_dim_ == 1 :
			if dataset_._distributions_1D.get(dist_key,None) != None :
				dist_key = dist_key + "-duplicated-auto-key;1"
				while dataset_._distributions_1D.get(dist_key,None) != None : dist_key = dist_key[:-1] + str(1+int(dist_key[-1:]))
			dataset_._distributions_1D[dist_key] = distribution
		if n_dim_ == 2 :
			if dataset_._distributions_2D.get(dist_key,None) != None :
				dist_key = dist_key + "-duplicated-auto-key;1"
				while dataset_._distributions_2D.get(dist_key,None) != None : dist_key = dist_key[:-1] + str(1+int(dist_key[-1:]))
			dataset_._distributions_2D[dist_key] = distribution


def load_yaml_file ( dataset_ , path_ , **kwargs ) :
	data = open_yaml_file(path_)
	if len(data) != 1 :
		msg.fatal("HEP_data_utils.helpers.load_yaml_file","{0} contains {1} entries, but I am only configured for 1... is this really a single distribution/matrix?".format(path_,len(data)) )
	dep_vars = data[0]["dependent_variables"]
	indep_vars = data[0]["independent_variables"]
	if len(indep_vars) < 3 :
		load_distributions_from_yaml ( dataset_ , dep_vars , indep_vars , path_ , n_dim_=len(indep_vars) , metadata_global_=kwargs.get("metadata_global_",{}) , metadata_local_=data[0] )
	else :
		msg.error("HEP_data_utils.helpers.load_yaml_file","file {0} has {1} independent_variables... I don't know what to do. I'm such a failure, I knew I wasn't cut out for this :(. The problematic entries are as follows:".format(path_,len(indep_vars)))
		print(indep_vars)
		msg.fatal("HEP_data_utils.helpers.load_yaml_file","could not interpret number of independent_variables")


def load_submission_file ( dataset_ , path_ , fname_ = "" ) :
	if len(fname_) > 0 :
		path_ = path_ + "/" + fname_
	data = open_yaml_file(path_)
	dataset_properties = data[0]
	msg.info("HEP_data_utils.helpers.load_submission_file","submission file with the following metadata:",_verbose_level=1)
	msg.check_verbosity_and_print(yaml.safe_dump(dataset_properties),_verbose_level=1)
	dataset_._description = dataset_properties["additional_resources"][0]["description"]
	dataset_._location = dataset_properties["additional_resources"][0]["location"]
	dataset_._comment = dataset_properties["comment"]
	dataset_._hepdata_doi = dataset_properties["hepdata_doi"]
	for idx in range(1,len(data)) :
		datum = data[idx]
		msg.info("HEP_data_utils.helpers.load_submission_file","submission file entry with the following definitions:",_verbose_level=1)
		msg.check_verbosity_and_print(yaml.safe_dump(datum),_verbose_level=1)
		filename = hlp.get_directory(path_) + "/" + datum["data_file"]
		if not os.path.isfile(filename) :
			msg.fatal("HEP_data_utils.helpers.load_submission_file","submission file asks for a yaml file called {0} but none exists".format(filename))
		msg.info("HEP_data_utils.helpers.load_submission_file","opening yaml file {0}".format(filename),_verbose_level=0)
		load_yaml_file(dataset_,filename,metadata_global_=datum)


def load_all_yaml_files ( dataset_ , dir_ ) :
	for filename in [ dir_+"/"+f for f in os.listdir(dir_) if is_yaml_file(f) ] :
		msg.info("HEP_data_utils.helpers.load_all_yaml_files","opening yaml file {0}".format(filename),_verbose_level=0)
		load_yaml_file(dataset_,filename)


def load_dataset ( dataset_ , path_ ) :
	path_ = hlp.remove_subleading(path_,"/")
	if os.path.isdir(path_) :
		msg.info("HEP_data_utils.helpers.load_dataset","{0} is a directory... I am expanding the entries (but will only go one directory deep!)".format(path_))
		if os.path.isfile(path_+"/submission.yaml") :
			msg.info("HEP_data_utils.helpers.load_dataset","submission.yaml file found in directory {0}... I will use this to steer the directory".format(path_))
			load_submission_file ( dataset_ , path_ , "submission.yaml" )
		else :
			msg.info("HEP_data_utils.helpers.load_dataset","no submission.yaml file found in directory {0}... I will open all available yaml files".format(path_))
			load_all_yaml_files ( dataset_ , path_ )
	else :
		if is_yaml_file(path_) == False :
			msg.fatal("HEP_data_utils.helpers.load_dataset","{0} doesn't seem to be a yaml file or a directory... I don't know what to do with it".format(path_))
		if is_submission_file(path_) :
			msg.info("HEP_data_utils.helpers.load_dataset","{0} is a submission.yaml file... I will use this to steer the directory".format(path_))
			load_submission_file ( dataset_ , path_ )
		else :
			msg.info("HEP_data_utils.helpers.load_dataset","Interpreting {0} as a yaml file... I will use it as my only input".format(path_))
			load_yaml_file ( dataset_ , path_ )