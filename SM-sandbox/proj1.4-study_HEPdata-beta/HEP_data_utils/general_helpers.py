# ====================================================================================================
#  Brief: some very generic helper functions
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================

import os, math
import numpy as np


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


#  Brief: check whether code is executed in an interactive shell
def is_ipython () :
	try :
		this_shell = get_ipython().__class__.__name__
		return True
	except NameError :
		return False
	except Exception :
		return False


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


# Brief: return the chi^2 between two distributions, using the uncertainty amplitudes in the inside direction
#        - assumes that the x-axes are identical
#        - assumes that bin-to-bin correlation is zero
def get_chi2 ( y1 , ey1_lo , ey1_hi , y2 , ey2_lo , ey2_hi ) :
	if y1.shape != y2.shape : return None
	if ey1_lo.shape != y1.shape or ey1_hi.shape != y1.shape : return None
	if ey2_lo.shape != y2.shape or ey2_hi.shape != y2.shape : return None
	y1, ey1_lo, ey1_hi = y1.flatten(), ey1_lo.flatten(), ey1_hi.flatten()
	y2, ey2_lo, ey2_hi = y2.flatten(), ey2_lo.flatten(), ey2_hi.flatten()
	chi2 = 0.
	for i in range(len(y1)) :
		res = y2[i] - y1[i]
		if res > 0 : err2 = ey1_hi[i]*ey1_hi[i] + ey2_lo[i]*ey2_lo[i]
		else : err2 = ey1_lo[i]*ey1_lo[i] + ey2_hi[i]*ey2_hi[i]
		chi2 = chi2 + res*res/err2
	return chi2


#  Brief: return True if two arrays look very similar within some margin of error
def do_arrays_look_similar ( array1 , array2 , **kwargs ) :
	if array1.shape != array2.shape : return False
	value_margin = kwargs.get("value_margin",1e-4)
	zero_margin = kwargs.get("zero_margin",None)
	array1, array2 = array1.flatten(), array2.flatten()
	ratio = np.zeros(shape=(len(array1)))
	for i in range(len(ratio)) :
		if array2[i] == 0 :
			if array1[i] == 0 : ratio[i] = 1
			else : ratio[i] = 0
			continue
		ratio[i] = array1[i] / array2[i]
	is_similar = True
	max1, max2 = max(np.fabs(array1.astype(np.float32))), max(np.fabs(array2.astype(np.float32)))
	for idx in range(len(ratio)) :
		if ratio[idx] > 1 - value_margin and ratio[idx] < 1 + value_margin : continue
		if zero_margin is not None and math.fabs(array1[idx]) / max1 < zero_margin : continue     #  Don't count zero entries, defined as less than zero_margin of max
		if zero_margin is not None and math.fabs(array2[idx]) / max2 < zero_margin : continue     #  Don't count zero entries, defined as less than zero_margin of max
		is_similar = False
	return is_similar