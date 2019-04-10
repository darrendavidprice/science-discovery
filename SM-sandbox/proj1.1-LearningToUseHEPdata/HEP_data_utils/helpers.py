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


def load_yaml_file ( dataset_ , path_ ) :
	data = open_yaml_file(path_)
	if len(data) != 1 :
		msg.fatal("load_yaml_file","{0} contains {1} entries, but I am only configured for 1... is this really a single distribution/matrix?".format(path_,len(data)) )
	dep_vars = data[0]["dependent_variables"][0]
	indep_vars = data[0]["independent_variables"]
	distribution = Distribution()
	distribution._description = dep_vars["header"].get("name","unknown")
	distribution._units = dep_vars["header"].get("units","unknown")
	for entry in dep_vars["values"] :
		distribution._values = np.append(distribution._values, entry["value"])
		errors = entry["errors"]
		for error in errors :
			print (type(error),error)
	msg.fatal("load_yaml_file","not implemented yet")


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

