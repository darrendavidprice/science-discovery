# ====================================================================================================
#  Brief: open a HEPdata file (either submission.yaml or table-file.yaml) and plot the contents
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================


import sys, os, getopt
import HEP_data_utils.messaging as msg
import HEP_data_utils.general_helpers as hlp
import HEP_data_utils.HEP_data_helpers as HD
from HEP_data_utils.data_structures import *


#  Brief: print help
def print_help () :
	msg.info("plot_contents_of_yaml.py:print_help","Usage is: python3 plot_contents_of_yaml.py <yaml file(s) OR directory with yaml-files OR submission.yaml file>")
	msg.info("plot_contents_of_yaml.py:print_help","If input files/directory contains a submission.yaml file, all other inputs will be ignored")
	msg.info("plot_contents_of_yaml.py:print_help","I assume that you follow the format instructions provided at https://hepdata-submission.readthedocs.io/en/latest/introduction.html")
	msg.info("plot_contents_of_yaml.py:print_help","Optional arguments are:")
	msg.info("plot_contents_of_yaml.py:print_help","       -h, --help\t\tPrint this help message")
	msg.info("plot_contents_of_yaml.py:print_help","       -v, --verbosity\t\tSet HEP_data_utils.messaging.VERBOSE_LEVEL as {-1, 0, 1, 2} (-1 by default)")
	msg.info("plot_contents_of_yaml.py:print_help","       -r, --recursive\t\tAllow recursive searching of directories. Recursion stops if submission.yaml file is found")
	msg.info("plot_contents_of_yaml.py:print_help","       --print\t\t\tPrint all information on the distributions found")
	msg.info("plot_contents_of_yaml.py:print_help","       --default-2D-bins\tPrevent interpretation of 2D bins as a matrix (will be stored as a 1D vector instead)")
	msg.info("plot_contents_of_yaml.py:print_help","N.B. you can validate your yaml files using the following package: https://github.com/HEPData/hepdata-validator")


#  Brief: parse command line arguments and check for errors
def parse_inputs ( argv_ ) :
	#  Get arguments
	try :
		opts, rest = getopt.getopt(argv_,"hrpv:",["help","recursive","print","verbosity="])
	except getopt.GetoptError as err :
		msg.error("plot_contents_of_yaml.py","The following error was thrown whilst parsing command-line arguments")
		print(">>>>>>>>\n",err,"\n<<<<<<<<")
		msg.error("plot_contents_of_yaml.py","Falling back to to --help...")
		print_help()
		msg.fatal("plot_contents_of_yaml.py","Command-line arguments not recognised.")
	#  Parse arguments
	do_recurse = False
	do_print_all = False
	do_not_make_matrix = False
	for opt, arg in opts:
		if opt in ['-h',"--help"] :
			print_help()
			sys.exit(0)
		if opt in ['-r',"--recursive",] :
			msg.info("plot_contents_of_yaml.py","Config: using recursion if needed",verbose_level=0)
			do_recurse = True
		if opt in ["--default-2D-bins"] :
			msg.info("plot_contents_of_yaml.py","Config: I will *not* try to convert 2D binning into matrix format",verbose_level=0)
			do_not_make_matrix = True
		if opt in ['-p',"--print"] :
			msg.info("plot_contents_of_yaml.py","Config: printing all distributions found",verbose_level=0)
			do_print_all = True
		if opt in ['-v',"--verbosity"] :
			msg.info("plot_contents_of_yaml.py","Config: setting verbosity to {0}".format(arg),verbose_level=0)
			try : msg.VERBOSE_LEVEL = int(arg)
			except : msg.fatal("plot_contents_of_yaml.py","Could not cast verbosity level {0} to integer".format(arg))
	#  Check that the remaining argument is valid
	if len(rest) == 0 :
		msg.error("plot_contents_of_yaml.py","No argument provided")
		print_help()
		msg.fatal("plot_contents_of_yaml.py","No input yaml file or directory provided")
	if len(rest) == 1 and hlp.is_directory(rest[0]) :
		msg.info("plot_contents_of_yaml.py","Opening input directory {0}...".format(rest[0]),verbose_level=0)
		rest = [rest[0]+"/"+f for f in os.listdir(rest[0])]
	rest = hlp.keep_only_yaml_files(rest,recurse=do_recurse)
	if len(rest) == 0 :
		msg.fatal("plot_contents_of_yaml.py","No input yaml files found from the inputs provided")
	for f in rest : msg.info("plot_contents_of_yaml.py","Registered input file {0}".format(f),verbose_level=0)
	return rest, do_print_all, do_not_make_matrix


#  =================================== #
#  ====    Brief: main program    ==== #
#  =================================== #
if __name__ == "__main__" :
				#
				#  Welcome
				#
	msg.info("plot_contents_of_yaml.py","Running program")
				#
				#  Get input files and settings
				#
	files_to_load, do_print_all, do_not_make_matrix = parse_inputs(sys.argv[1:])
				#
				#  Load input files
				#
	my_tables = DistributionContainer("my_tables")
	HD._make_matrix_if_possible = not do_not_make_matrix
	HD.load_all(my_tables,files_to_load)
				#
				#  Print contents (if --print was called then go into excruciating detail)
				#
	my_tables.print_keys()
	if do_print_all : my_tables.print_all()
				#
				#  Plot everything
				#
	my_tables.plot_all()
				#
				#  Goodbye
				#
	msg.info("plot_contents_of_yaml.py","Program reached the end without crashing and will close :) Have a nice day...")

