import os, yaml
import general_utils.messaging as msg
import general_utils.helpers as hlp
from HEP_data_utils.data_structures import *


def is_yaml_file ( path_ ) :
	return hlp.check_extension(path_,"yaml")


def is_submission_file ( path_ ) :
	return path_[-15:] == "submission.yaml"


def open_yaml_file ( path_ ) :
	yaml_file = open(path_, 'r')
	data = []
	try :
		for datum in yaml.safe_load_all(yaml_file) :
			msg.info("open_yaml_file","yaml file opened with entries:",_verbose_level=1)
			msg.check_verbosity_and_print(yaml.safe_dump(datum),_verbose_level=1)
			data.append(datum)
	except yaml.YAMLError as exc :
		print ( exc )
		msg.fatal("open_yaml_file","Exception thrown when opening the yaml file (see previous messages)")
	return data


def get_error_from_yaml_map ( distribution_ , error_ , pt_idx , err_idx_=0 ) :
	key = error_.get("label","err{0}".format(err_idx_))
	if error_.get("symerror",None) != None :
		if distribution_._symm_errors.get(key,None) == None :
			msg.info("get_error_from_yaml_map","Creating symmetric error {0} with length {1}".format(key,len(distribution_)),_verbose_level=1)
			distribution_._symm_errors[key] = np.zeros(shape=(len(distribution_)))
		distribution_._symm_errors[key][pt_idx] = error_["symerror"]
	elif error_.get("asymerror",None) != None :
		err_asymm = error_["asymerror"]
		if distribution_._asymm_errors_up.get(key,None) == None :
			msg.info("get_error_from_yaml_map","Creating asymmetric error {0} with length {1}".format(key,len(distribution_)),_verbose_level=1)
			distribution_._asymm_errors_up[key] = np.zeros(shape=(len(distribution_)))
			distribution_._asymm_errors_down[key] = np.zeros(shape=(len(distribution_)))
		if err_asymm.get("plus",None) == None :
			msg.fatal("get_error_from_yaml_map","No entry named \"plus\" for error \"asymerror\"")
		else :
			distribution_._asymm_errors_up[key][pt_idx] = err_asymm["plus"]
		if err_asymm.get("minus",None) == None :
			msg.fatal("get_error_from_yaml_map","No entry named \"minus\" for error \"asymerror\"")
		else :
			distribution_._asymm_errors_down[key][pt_idx] = err_asymm["minus"]
	else :
		print(error_)
		msg.fatal("get_error_from_yaml_map","map does not have an entry called symerror or asymerror")
	return key


def set_1D_bins ( distribution_ , indep_vars_ ) :
	if len(indep_vars_) != 1 :
		msg.fatal("set_1D_bins","distribution {0} has {1} independent_variables but I am only configured to deal with 1".format(distribution_._description,len(indep_vars_)))
	distribution_._bin_values = np.zeros(shape=(1+len(distribution_)))
	distribution_._bin_labels = [ "unlabeled" for i in range(0,len(distribution_)) ]
	for i in range(0,len(indep_vars_[0]["values"])) :
		bin = indep_vars_[0]["values"][i]
		if bin.get("value",None) != None :
			distribution_._bin_labels[i] = bin["value"]
		elif bin.get("high",None) != None and bin.get("low",None) != None :
			if i == 0 :
				distribution_._bin_values[0] = bin["low"]
				distribution_._bin_values[1] = bin["high"]
				continue
			if bin["low"] != distribution_._bin_values[i] :
				msg.fatal("set_1D_bins","Bin entry {0} for distribution {1} is not continuous from the previous bin which ended at {2}".format(bin,distribution_._description,distribution_._bin_values[i]))
			distribution_._bin_values[i+1] = bin["high"]
		else :
			msg.fatal("set_1D_bins","Could not interpret bin entry {0} for distribution {1}".format(bin,distribution_._description))


def set_2D_bins ( distribution_ , indep_vars_ ) :
	if len(indep_vars_) != 2 :
		msg.fatal("set_2D_bins","distribution {0} has {1} independent_variables but I am only configured to deal with 2".format(distribution_._description,len(indep_vars_)))
	distribution_._bin_labels_x = np.zeros(shape=(len(distribution_)))
	distribution_._bin_labels_y = np.zeros(shape=(len(distribution_)))
	for i in range(0,len(indep_vars_[0]["values"])) :
		bin = indep_vars_[0]["values"][i]
		if bin.get("value",None) != None :
			distribution_._bin_labels_x[i] = bin["value"]
		else :
			msg.fatal("set_2D_bins","Could not interpret bin entry {0} for distribution {1}".format(bin,distribution_._description))
	for i in range(0,len(indep_vars_[1]["values"])) :
		bin = indep_vars_[1]["values"][i]
		if bin.get("value",None) != None :
			distribution_._bin_labels_y[i] = bin["value"]
		else :
			msg.fatal("set_2D_bins","Could not interpret bin entry {0} for distribution {1}".format(bin,distribution_._description))


def load_distributions_from_yaml ( dataset_ , dep_vars_ , indep_vars_ , path_ , n_dim_ = 1 ) :
	for var_idx in range(0,len(dep_vars_)) :
		dep_var = dep_vars_[var_idx]
		distribution = Distribution()
		if n_dim_ == 1 : distribution = Distribution_1D()
		if n_dim_ == 2 : distribution = Distribution_2D()
		distribution._description = dep_var["header"].get("name","unknown")
		distribution._units = dep_var["header"].get("units","unknown")
		pt_idx = 0
		for entry in dep_var["values"] :
			try :
				distribution._values = np.append(distribution._values, entry["value"])
			except KeyError as exc :
				msg.error("load_distributions_from_yaml","KeyError: {0}".format(exc),_verbose_level=-1)
				msg.fatal("load_distributions_from_yaml","Entry with no \"value\" when trying to create distribution {0} in file {1}".format(distribution._description,path_))
		for entry in dep_var["values"] :
			try :
				errors = entry["errors"]
			except KeyError as exc :
				msg.error("load_distributions_from_yaml","KeyError: {0}".format(exc),_verbose_level=1)
				msg.warning("load_distributions_from_yaml","Entry with no \"errors\" when trying to create distribution {0} in file {1}... Assuming there are none".format(distribution._description,path_),_verbose_level=1)
				errors = []
			err_idx = 0
			for error in errors :
				get_error_from_yaml_map(distribution,error,pt_idx,err_idx)
				err_idx = err_idx + 1
			pt_idx = pt_idx + 1
		expected_size = len(distribution._values)
		if n_dim_ == 1 : set_1D_bins(distribution,indep_vars_)
		elif n_dim_ == 2 : set_2D_bins(distribution,indep_vars_)
		else : msg.fatal("load_distributions_from_yaml","number of bin dimensions is {0} but I can only handle 1 or 2".format(n_dim_))
		for error in [distribution._symm_errors,distribution._asymm_errors_up,distribution._asymm_errors_down] :
			for key in error :
				this_err_size = len(error[key])
				if this_err_size == expected_size : continue
				msg.fatal("load_distributions_from_yaml","error source {0} has length {1} for distribution [{2}] where {3} was expected".format(key,this_err_size,distribution._description,expected_size))
		msg.info("load_distributions_from_yaml","yaml file loaded with the following entries",_verbose_level=0)
		if msg.VERBOSE_LEVEL < 1 : print(distribution)


def load_yaml_file ( dataset_ , path_ ) :
	data = open_yaml_file(path_)
	if len(data) != 1 :
		msg.fatal("load_yaml_file","{0} contains {1} entries, but I am only configured for 1... is this really a single distribution/matrix?".format(path_,len(data)) )
	dep_vars = data[0]["dependent_variables"]
	indep_vars = data[0]["independent_variables"]
	if len(indep_vars) < 3 :
		load_distributions_from_yaml ( dataset_ , dep_vars , indep_vars , path_ , len(indep_vars) )
	else :
		msg.error("load_yaml_file","file {0} has {1} independent_variables... I don't know what to do. I'm such a failure, I knew I wasn't cut out for this :(. The problematic entries are as follows:".format(path_,len(indep_vars)))
		print(indep_vars)
		msg.fatal("load_yaml_file","could not interpret number of independent_variables")


def load_submission_file ( dataset_ , path_ , fname_ = "" ) :
	if len(fname_) > 0 :
		path_ = path_ + "/" + fname_
	data = open_yaml_file(path_)
	dataset_properties = data[0]
	msg.info("load_submission_file","submission file with the following metadata:",_verbose_level=1)
	msg.check_verbosity_and_print(yaml.safe_dump(dataset_properties),_verbose_level=1)
	dataset_._description = dataset_properties["additional_resources"][0]["description"]
	dataset_._location = dataset_properties["additional_resources"][0]["location"]
	dataset_._comment = dataset_properties["comment"]
	dataset_._hepdata_doi = dataset_properties["hepdata_doi"]
	for idx in range(1,len(data)) :
		datum = data[idx]
		msg.info("load_submission_file","submission file entry with the following definitions:",_verbose_level=1)
		msg.check_verbosity_and_print(yaml.safe_dump(datum),_verbose_level=1)
		filename = hlp.get_directory(path_) + "/" + datum["data_file"]
		if not os.path.isfile(filename) :
			msg.fatal("load_submission_file","submission file asks for a yaml file called {0} but none exists".format(filename))
		msg.info("load_submission_file","opening yaml file {0}".format(filename),_verbose_level=0)
		load_yaml_file(dataset_,filename)


def load_all_yaml_files ( dataset_ , dir_ ) :
	for filename in [ dir_+"/"+f for f in os.listdir(dir_) if is_yaml_file(f) ] :
		msg.info("load_all_yaml_files","opening yaml file {0}".format(filename),_verbose_level=0)
		load_yaml_file(dataset_,filename)

