# ====================================================================================================
#  Brief: some very generic helper functions
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================

import os


#  Brief: return the file extension
def get_extension ( path_ ) :
	os.path.splitext(path_)[1][1:]
	return path_


#  Brief: return the directory of the path
def get_directory ( path_ ) :
	return os.path.dirname(path_)


#  Brief: check if the file has the correct extension
def check_extension ( path_ , ext_ ) :
	if type(path_) is not str : return False
	f_ext = get_extension(os.path.splitext(path_)[1][1:])
	return f_ext == ext_


#  Brief: remove recurrances of pattern_ from the end of string path_
def remove_subleading ( path_ , pattern_ ) :
	while path_[len(path_)-len(pattern_)] == pattern_ :
		path_ = path_[:len(path_)-len(pattern_)]
	return path_


#  Brief: check that file has a yaml extension
def is_yaml_file ( path_ ) :
	if type(path_) is not str : return False
	return check_extension(path_,"yaml")


#  Brief: check that file is a HEPdata submission file
def is_submission_file ( path_ ) :
	if type(path_) is not str : return False
	return path_[-15:] == "submission.yaml"


#  Brief: check that file has a yaml extension
def is_root_file ( path_ ) :
	if type(path_) is not str : return False
	return check_extension(path_,"root")


#  Brief: check that file has a yaml extension
def is_directory ( path_ ) :
	if type(path_) is not str : return False
	return os.path.isdir(path_)


#  Brief: return the rootfiles to be processed.
def keep_only_root_files ( inputs_ , **kwargs ) :
	if type(inputs_) is not list : inputs_ = [inputs_]
	do_recurse = kwargs.get("recurse",False)
	ret = []
	for f in inputs_ :
		if do_recurse and is_directory(f) :
			ret = ret + keep_only_root_files([f+"/"+f2 for f2 in os.listdir(f)],recurse=True)
		if not is_root_file(f) : continue
		ret.append(f)
	return ret


#  Brief: add red colour formatting to a string
def red_str ( s , **kwargs ) :
	ret = "\x1b["
	if kwargs.get("bold",False) : ret = ret + "1"
	else : ret = ret + "0"
	ret = ret + ";31"
	if "bkg" in kwargs : ret = ret + kwargs["bkg"]
	return ret + "m" + s + "\x1b[0m"


#  Brief: add green colour formatting to a string
def green_str ( s , **kwargs ) :
	ret = "\x1b["
	if kwargs.get("bold",False) : ret = ret + "1"
	else : ret = ret + "0"
	ret = ret + ";32"
	if "bkg" in kwargs : ret = ret + kwargs["bkg"]
	return ret + "m" + s + "\x1b[0m"


#  Brief: add green colour formatting to a string
def magenta_str ( s , **kwargs ) :
	ret = "\x1b["
	if kwargs.get("bold",False) : ret = ret + "1"
	else : ret = ret + "0"
	ret = ret + ";35"
	if "bkg" in kwargs : ret = ret + kwargs["bkg"]
	return ret + "m" + s + "\x1b[0m"


#  Brief: take list of inputs and return the yaml files to be processed. If a submission file is found, all other arguments are ignored.
def keep_only_yaml_files ( inputs_ , **kwargs ) :
	if type(inputs_) is not list : inputs_ = [inputs_]
	do_recurse = kwargs.get("recurse",False)
	ret = [ f for f in inputs_ if is_submission_file(f) ]
	if len(ret) > 0 :
		return ret
	for f in inputs_ :
		if do_recurse and is_directory(f) :
			ret = ret + keep_only_yaml_files([f+"/"+f2 for f2 in os.listdir(f)],recurse=True)
		if not is_yaml_file(f) : continue
		ret.append(f)
	return ret