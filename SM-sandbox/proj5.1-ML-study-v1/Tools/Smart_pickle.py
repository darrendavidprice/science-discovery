import pickle
import os.path


rcParams = {
    "ignore_load" : False
}

def rcParam (key, default) :
    return rcParam.get(key, default)

def resolve_rcParam (key, argument, rogue=None) :
    if argument == rogue :
        global rcParams
        return rcParams[key]
    return argument


def dir_exists (fname) :
	return os.path.exists(fname) and is_dir(fname)

def dump (fname, obj, **kwargs) :
	to_pickle = [obj, kwargs]
	pickle.dump(to_pickle, open(fname, "wb"))

def file_exists (fname) :
	return os.path.exists(fname) and is_file(fname)

def is_dir(fname) :
	return os.path.isdir(fname)

def is_file(fname) :
	return os.path.isfile(fname)

def load (fname, **kwargs) :
	if not file_exists(fname) :
		return None
	if resolve_rcParam("ignore_load", kwargs.get("ignore_load",None), None) :
		return None
	from_pickle = pickle.load(open(fname, "rb"))
	saved_args = from_pickle[1]
	for key in kwargs :
		if key     not in saved_args      : return None
		if kwargs[key] != saved_args[key] : return None
	return from_pickle[0]

