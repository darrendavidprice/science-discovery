# ====================================================================================================
#  Brief: open a HEPdata file (either submission.yaml or table-file.yaml) and see whether the contents
#         can be found in another dataset of root files
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================


import sys, os, getopt, shutil, math
import numpy as np
import HEP_data_utils.messaging as msg
import HEP_data_utils.plotting as plotter
import HEP_data_utils.general_helpers as hlp
import HEP_data_utils.HEP_data_helpers as HD
import HEP_data_utils.ROOT_helpers as RT
from HEP_data_utils.DistributionContainer import DistributionContainer


#  Brief: print help
def print_help () :
	msg.info("validate_yaml_files.py:print_help","Usage is: python3 validate_yaml_files.py --yaml <yaml file(s)> --compare <root file(s)>")
	msg.info("validate_yaml_files.py:print_help","If input files/directory contains a submission.yaml file, all other inputs will be ignored")
	msg.info("validate_yaml_files.py:print_help","I assume that you follow the format instructions provided at https://hepdata-submission.readthedocs.io/en/latest/introduction.html")
	msg.info("validate_yaml_files.py:print_help","Optional arguments are:")
	msg.info("validate_yaml_files.py:print_help","       -h, --help\t\tPrint this help message and close")
	msg.info("validate_yaml_files.py:print_help","       -v, --verbosity\t\tSet HEP_data_utils.messaging.VERBOSE_LEVEL as {-1, 0, 1, 2} (-1 by default)")
	msg.info("validate_yaml_files.py:print_help","       -r, --recursive\t\tAllow recursive searching of directories. Recursion stops if submission.yaml file is found")
	msg.info("validate_yaml_files.py:print_help","N.B. you can validate your yaml files using the following package: https://github.com/HEPData/hepdata-validator")
#	msg.info("validate_yaml_files.py:print_help","       --keys\t\tSpecify a file of keys to load")


#  Brief: return the list provided after one of the option arguments
def get_argument_list ( argv , options ) :
	if type(options) is not list : options = [options]
	options = [ str(opt) for opt in options ]
	options = [ a for a in argv if a in options ]
	if len(options) > 1 :
		msg.fatal("validate_yaml_files.py","Program arguments {0} mean the same thing - please only provide one".format(options))
	arguments = []
	save_arg = False
	for el in argv :
		if el in options :
			save_arg = True
			continue
		if el[:1] is "-" :
			save_arg = False
			continue
		if el is "validate_yaml_files.py" :
			save_arg = False
			continue
		if not save_arg : continue
		arguments.append(el)
	return arguments


#  Brief: parse command line arguments and check for errors
def parse_inputs ( argv_ ) :
	#  Get arguments
	try :
		opts, rest = getopt.getopt(argv_,"hrv:",["help","recursive","verbosity=","yaml=","compare=",""])
	except getopt.GetoptError as err :
		msg.error("validate_yaml_files.py","The following error was thrown whilst parsing command-line arguments")
		print(">>>>>>>>\n",err,"\n<<<<<<<<")
		msg.error("validate_yaml_files.py","Falling back to to --help...")
		print_help()
		msg.fatal("validate_yaml_files.py","Command-line arguments not recognised.")
	#  Parse arguments
	do_recurse = False
	for opt, arg in opts:
		if opt in ['-h',"--help"] :
			print_help()
			sys.exit(0)
		if opt in ['-r',"--recursive",] :
			msg.info("validate_yaml_files.py","Config: using recursion if needed",verbose_level=0)
			do_recurse = True
		if opt in ['-v',"--verbosity"] :
			msg.info("validate_yaml_files.py","Config: setting verbosity to {0}".format(arg),verbose_level=0)
			try : msg.VERBOSE_LEVEL = int(arg)
			except : msg.fatal("validate_yaml_files.py","Could not cast verbosity level {0} to integer".format(arg))
	yaml_files = hlp.keep_only_yaml_files(get_argument_list(argv_,"--yaml"),recurse=do_recurse)
	root_files = hlp.keep_only_root_files(get_argument_list(argv_,"--compare"),recurse=do_recurse)
	if len(yaml_files) == 0 :
		msg.fatal("validate_yaml_files.py","Please provide at least one yaml file using the --yaml option")
	if len(root_files) == 0 :
		msg.fatal("validate_yaml_files.py","Please provide at least one root file using the --compare option")
	#  Return
	return yaml_files, root_files


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


#  Brief: return True if two tables have the same central values
def do_central_values_look_similar ( table1 , table2 , **kwargs ) :
	return do_arrays_look_similar ( table1._dep_var._values , table2._dep_var._values , **kwargs )


#  Brief: get combined errors of a dependent variable
def get_total_errors ( dep_var ) :
	shape = dep_var._values.shape
	length = len(dep_var._values.flatten())
	ey_lo, ey_hi = np.zeros( shape=(length) ), np.zeros( shape=(length) )
	for key in dep_var._symerrors :
		errs = dep_var._symerrors[key].flatten()
		ey_lo = ey_lo + np.multiply(errs,errs)
	for key in dep_var._asymerrors_up :
		errs1 = dep_var._asymerrors_up[key].flatten()
		errs2 = dep_var._asymerrors_dn[key].flatten()
		for i in range(0,len(errs1)) :
			err1 = errs1[i]
			err2 = errs2[i]
			if err1 > 0 : ey_hi[i] = ey_hi[i] + err1*err1
			else : ey_lo[i] = ey_lo[i] + err1*err1
			if err2 > 0 : ey_hi[i] = ey_hi[i] + err2*err2
			else : ey_lo[i] = ey_lo[i] + err2*err2
	return np.sqrt(ey_lo.reshape(shape)), np.sqrt(ey_hi.reshape(shape))


#  Brief: return e.g. [True,True,False] if the binnings are similar
def do_bins_look_similar ( table1 , table2 , **kwargs ) :
	bins_match = []
	for var_idx1 in range(table1.n_indep_vars()) :
		match = False
		indep_var_1 = table2._indep_vars[var_idx1]
		for var_idx2 in range(table2.n_indep_vars()) :
			indep_var_2 = table1._indep_vars[var_idx2]
			if not do_arrays_look_similar(indep_var_1._bin_centers,indep_var_2._bin_centers) : continue
			if not do_arrays_look_similar(indep_var_1._bin_widths_lo,indep_var_2._bin_widths_lo,value_margin=1e-3,zero_margin=1e-5) : continue
			if not do_arrays_look_similar(indep_var_1._bin_widths_hi,indep_var_2._bin_widths_hi,value_margin=1e-3,zero_margin=1e-5) : continue
			match = True
		bins_match.append(match)
	return bins_match


#  Brief: return a list of errors which match in table2, and a list of the rest. Include "total_up" and "total_down" options
def do_errors_look_similar ( table1 , table2 , **kwargs ) :
	keys_match, keys_no_match = [], []
	dep_var1, dep_var2 = table1._dep_var, table2._dep_var
	dep_var1_total_dn, dep_var1_total_up = get_total_errors(dep_var1)
	dep_var2_total_dn, dep_var2_total_up = get_total_errors(dep_var2)
	totalerrs1 = {"total (up)":dep_var1_total_up, "total (down)":dep_var1_total_dn}
	totalerrs2 = {"total (up)":dep_var2_total_up, "total (down)":dep_var2_total_dn}
	dicts = [ dep_var1._symerrors , dep_var1._asymerrors_up , dep_var1._asymerrors_dn , totalerrs1 ]
	labels = [ "" , " (up)" , " (down)" , "" ]
	for i in range(4) :
		d = dicts[i]
		label = labels[i]
		for key1, errs1 in d.items() :
			match_found = False
			for key2, errs2 in dep_var2._symerrors.items() :
				keys_match.append(key1+label)
				match_found = True
			for key2, errs2 in dep_var2._asymerrors_up.items() :
				keys_match.append(key1+label)
				match_found = True
			for key2, errs2 in dep_var2._asymerrors_dn.items() :
				keys_match.append(key1+label)
				match_found = True
			for key2, errs2 in totalerrs2.items() :
				keys_match.append(key1+label)
				match_found = True
			if match_found : continue
			keys_no_match.append(key1)
	keys_match = set(keys_match)
	keys_no_match = { x for x in keys_no_match if x not in keys_match }
	return keys_match, keys_no_match


#  Brief: see if root_table_ matches any error source in hep_table_, and return this source if so
def get_match ( hep_table_ , root_table_ , **kwargs ) :
	keys_match, keys_no_match = do_errors_look_similar(hep_table_,root_table_)
	if hep_table_.n_bins() != root_table_.n_bins() : return False, [False], keys_match, keys_no_match
	if hep_table_.n_indep_vars() != root_table_.n_indep_vars() : return False, [False], keys_match, keys_no_match
	values_match = do_central_values_look_similar(hep_table_,root_table_,**kwargs)
	bins_match = do_bins_look_similar(hep_table_,root_table_)
	print(values_match, bins_match, keys_match, keys_no_match)
	return values_match, bins_match, keys_match, keys_no_match


#  Brief: print the results of the table matching process
def print_match_result ( key_ , val_match_ , errs_match_ , errs_not_match_ ) :
	if val_match_ : ret = hlp.green_str("values")
	else : ret = hlp.red_str("values")
	errs_match, errs_not_match = list(errs_match_), list(errs_not_match_)
	errs_match.sort(), errs_not_match.sort()
	for err in errs_match : ret = ret + "     " + hlp.green_str(err)
	for err in errs_not_match : ret = ret + "     " + hlp.red_str(err)
	if len(key_) > len(ret) : total_width = 17 + len(key_)
	else : total_width = 17 + len(ret)
	terminal_width = shutil.get_terminal_size((-1,-1))[0]
	if terminal_width > 0 and total_width > terminal_width : total_width = terminal_width
	print("{message:{fill}{align}{width}}".format(message='',fill='-',align='<',width=total_width))
	print("DISTRIBUTION:   ",hlp.magenta_str(key_))
	print("     matches:   ",ret)
	print("{message:{fill}{align}{width}}".format(message='',fill='-',align='<',width=total_width))


#  Brief: find matches for table_ in other_store_
def print_matches ( key_ , table_ , other_store_ ) :
	n_dim = table_.n_indep_vars()
	central_values_match = False
	matches, not_matches = [], []
	if n_dim == 0 : other_distributions = other_store_._inclusive_distributions.items()
	elif n_dim == 1 : other_distributions = other_store_._1D_distributions.items()
	elif n_dim == 2 : other_distributions = other_store_._2D_distributions.items()
	else : other_distributions = other_store_._ND_distributions.items()
	for other_key, other_table in other_distributions :
		these_values_match, these_bins_match, errs_match, errs_not_match = get_match ( table_ , other_table )
		all_errs = {}
		if these_values_match : central_values_match = True
		if errs_match is not None and errs_not_match is not None : all_errs = errs_match.union(errs_not_match)
		else : all_errs = get_all_err_keys(table_)
		if False in these_bins_match or not these_values_match :
			for err in all_errs :
				not_matches.append(err)
			continue
		if errs_match is not None :
			for err in errs_match :
				matches.append(err)
		if errs_not_match is not None :
			for err in errs_not_match :
				not_matches.append(err)
	matches = list(set(matches))
	not_matches = list({ x for x in not_matches if x not in matches })
	print_match_result ( key_ , central_values_match , matches , not_matches )



#  =================================== #
#  ====    Brief: main program    ==== #
#  =================================== #
if __name__ == "__main__" :
				#
				#  Welcome
				#
	msg.info("validate_yaml_files.py","Running program")
				#
				#  Get input files and settings
				#
	yamls_to_load, roots_to_load = parse_inputs(sys.argv[1:])
				#
				#  Load input files
				#
	yaml_tables = DistributionContainer("yaml_files")
	HD.load_yaml_files_from_list(yaml_tables,yamls_to_load)
				#
	root_tables = DistributionContainer("root_files")
	RT.load_root_files_from_list(root_tables,roots_to_load)
				#
				#  Look for matches
				#
	keys = yaml_tables.get_keys()
	for l in keys :
		for key in l :
			print_matches ( key , yaml_tables.get_table(key) , root_tables )
				#
				#  Goodbye
				#
	msg.info("validate_yaml_files.py","Program reached the end without crashing and will close :) Have a nice day...")
				#
				#
