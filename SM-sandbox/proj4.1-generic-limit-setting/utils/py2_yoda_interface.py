##  =======================================================================================================================
##  Brief: Module to be run in python2.X, in which yoda is compatible. Extracts data from yoda and pickles it
##  Author: Stephen Menary
##  Email: sbmenary@gmail.com
##  =======================================================================================================================

import sys
import os
import getopt
import pickle


##  Global placeholder for imported yoda module
#
yoda = None


#  ========================================================
#  Python2 Messaging
#  ========================================================

def print_message (typ, origin, message) :
	print ("{0}\t\t{1}\t\t{2}".format(typ, origin, message))

def info (origin, message) :
	print_message("INFO", origin, message)

def error (origin, message) :
	print_message("ERROR", origin, message)

def warning (origin, message) :
	print_message("WARNING", origin, message)

def fatal (origin, message, code=0) :
	print_message("FATAL", origin, message)
	sys.exit(code)


#  ========================================================
#  Python2 Helpers
#  ========================================================

def path_exists (fname) :
	return os.path.exists(fname)

def is_file (fname) :
	return os.path.isfile(fname)

def is_dir (fname) :
	return os.path.isdir(fname)

def is_yoda_file (fname) :
	return fname.split('.')[-1] == "yoda"

def validate_yoda_file_path (fname) :
	if not path_exists(fname):
		raise RuntimeError("validate_yoda_file_path(): path {} does not exist".format(fname))
	if not is_file(fname):
		raise RuntimeError("validate_yoda_file_path(): no file named {}".format(fname))
	if not is_yoda_file(fname):
		raise RuntimeError("validate_yoda_file_path(): {} is not a .yoda file".format(fname))
	return True


#  ========================================================
#  Python2 yoda parsing methods
#  ========================================================

##  Check that we are running in python2
#
def check_python_version () :
	if sys.version_info.major is 2 : return
	raise EnvironmentError("check_python_version(): py2_yoda_interface module must be run with python2 in order to import yoda")

##  Import yoda and return module. Once run, yoda module is globally accessible using sys.modules["yoda"]
#
def import_yoda () :
	check_python_version ()
	try :
		import yoda
		info("import_yoda()", "yoda successfully imported")
		global yoda
		yoda = sys.modules["yoda"]
	except ImportError as e :
		warning("import_yoda()", "yoda import failed with error: {}".format(e))
		python_path = "/usr/local/lib/python2.7/site-packages/"
		if python_path in sys.path :
			error("import_yoda()", "{} already in sys.path".format(python_path))
			raise ImportError(e)
		warning("import_yoda()", "adding {} to sys.path and retrying".format(python_path))
		sys.path.append(python_path)
		import_yoda()

##  Make a pickle-able dictionary usign a yoda file
#
def yoda_to_dict (yoda_file) :
	ret = {}
	for key, item in yoda_file.items() :
		ret[key] = {"key": key, "type": str(type(item))}
		if type(item) is yoda.core.Counter :
			ret[key]["val"] = item.val
			ret[key]["err"] = item.err
			ret[key]["numEntries"] = item.numEntries()
			ret[key]["relErr"] = item.relErr
			ret[key]["sumW"] = item.sumW()
			ret[key]["sumW2"] = item.sumW2()
		elif type(item) is yoda.core.Histo1D :
			ret[key]["xEdges"] = item.xEdges()
			ret[key]["yVals"] = item.yVals()
			ret[key]["yErrs"] = item.yErrs()
		elif type(item) is yoda.core.Scatter1D :
			ret[key]["x"]     = [p.x for p in item.points]
			ret[key]["ex_lo"] = [p.xErrs.minus for p in item.points]
			ret[key]["ex_hi"] = [p.xErrs.plus for p in item.points]
		elif type(item) is yoda.core.Scatter2D :
			ret[key]["x"]     = [p.x for p in item.points]
			ret[key]["ex_lo"] = [p.xErrs.minus for p in item.points]
			ret[key]["ex_hi"] = [p.xErrs.plus for p in item.points]
			ret[key]["y"]     = [p.y for p in item.points]
			ret[key]["ey_lo"] = [p.yErrs.minus for p in item.points]
			ret[key]["ey_hi"] = [p.yErrs.plus for p in item.points]
		else :
			warning("yoda_to_dict)_", "Key {0} with unknown type {1}".format(key, type(item)))
	return ret


##  Open yoda file and pickle the dictionary of it's contents
#
def yoda_to_pickle (in_fname, out_fname, record=None) :
	info("py2_yoda_interface()", "using {} as input .yoda file".format(in_fname))
	info("py2_yoda_interface()", "using {} as output pickle binary".format(out_fname))
	import_yoda ()
	yoda_file = yoda.read(in_fname)
	pickle.dump(yoda_to_dict(yoda_file), open(out_fname, "wb"))
	if record is None : return
	info("py2_yoda_interface()", "storing record in file {}".format(record))
	rec_file = open(record, "w")
	rec_file.write(in_fname + "\n")
	rec_file.write(out_fname + "\n")
	rec_file.close()


##  Parse command line arguments and return the target input and output files, and record file if specified
#
def parse_command_line_arguments () :
	argv = sys.argv[1:]
	try :
		opts, rest = getopt.getopt(argv, "o:r:", ["out=","record=="])
	except getopt.GetoptError as err :
		error("parse_command_line_arguments()", "The following error was thrown whilst parsing command-line arguments")
		fatal("parse_command_line_arguments()", err)
	if len(rest) is not 1 :
		raise ValueError("parse_command_line_arguments(): expected 1 unlabelled argument where {0} provided".format(len(argv)))
	in_name = rest[0]
	validate_yoda_file_path(in_name)
	out_file = in_name.split("/")
	if out_file[-1][0] is not "." : out_file[-1] = "." + out_file[-1]
	out_file, use_default_out_file = "/".join(out_file)[:-5] + ".pickle", True
	record = ".py2_yoda_interface.record"
	for opt, arg in opts:
		if opt in ['-o', "--out"] :
			out_file = arg
			use_default_out_file = False
		if opt in ['-r', "--record"] :
			record = arg
	if path_exists(out_file) :
		if use_default_out_file :
			raise RuntimeError("parse_command_line_arguments(): path {} already exists. Specify a different one using -o or --out.".format(out_file))
		warning("parse_command_line_arguments()", "output {} already exists and will be replaced".format(out_file))
	return in_name, out_file, record


#  ========================================================
#  Scripting fallback
#  ========================================================

if __name__ == "__main__" :
	info("py2_yoda_interface()", "running as main")
	in_name, out_file, record = parse_command_line_arguments()
	yoda_to_pickle(in_name, out_file, record=record)


