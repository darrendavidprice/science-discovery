# ====================================================================================================
#  Brief: test uproot package
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================


import sys, os, getopt, uproot
import HEP_data_utils.messaging as msg
import HEP_data_utils.general_helpers as hlp
import HEP_data_utils.HEP_data_helpers as HD
from HEP_data_utils.data_structures import *


#  Brief: print help
'''
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
'''


#  Brief: parse command line arguments and check for errors
'''
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
'''


def get_histograms ( in_ , pre_key = "" ) :
	if len(pre_key) > 0 :
		if pre_key[-1:] != "/" : pre_key = pre_key + "/"
	ret = {}
	if type(in_) is str :
		f = uproot.open(in_)
		ret = { **ret , **get_histograms(f,"{0}{1}".format(pre_key,in_)) }
	elif type(in_) is uproot.rootio.ROOTDirectory :
		for key, value in in_.items() :
			for key2, value2 in get_histograms(value,"{0}{1}".format(pre_key,key)).items() :
				if key2 in ret : key2 = key2 + ";1"
				while key2 in ret :
					num = int(key2.split(";").pop())
					key2 = key2 + str(num+1)
				ret[key2] = value2
	else : ret [ "{0}{1}".format(pre_key,in_.name) ] = in_
	return ret


def read_TH1 ( histo_ ) :
	print("\nTH1:\n")
	print(histo_.name)
	print(histo_.edges)
	print(histo_.values)
	print(histo_.xlabels)
	return 
	print([x for x in dir(histo_) if x[:1] != "_"])


def read_TH2 ( histo_ ) :
	print([x for x in dir(histo_) if x[:1] != "_"])
	print("\nTH2:\n")
	print(histo_.name)
	print(histo_.edges[0])
	print(histo_.edges[1])
	print(histo_.values)
	print(histo_.xlabels)
	print(histo_.ylabels)
	return 
	print([x for x in dir(histo_) if x[:1] != "_"])


def read_TGraphAsymmErrors ( histo_ ) :
	return
	print("\nTGRAPHASYMMERRORS:\n")
	print(histo_.name)
	print(histo_.xlabel)
	print(histo_.xvalues)
	print(histo_.xerrorslow)
	print(histo_.xerrorshigh)
	print(histo_.ylabel)
	print(histo_.yvalues)
	print(histo_.yerrorslow)
	print(histo_.yerrorshigh)
	return 
	print([x for x in dir(histo_) if x[:1] != "_"])


def read_TGraphErrors ( histo_ ) :
	return
	print("\nTGRAPHERRORS:\n")
	print(histo_.name)
	print(histo_.xlabel)
	print(histo_.xvalues)
	print(histo_.xerrors)
	print(histo_.ylabel)
	print(histo_.yvalues)
	print(histo_.yerrors)
	return 
	print([x for x in dir(histo_) if x[:1] != "_"])


def read_TGraph ( histo_ ) :
	return
	print("\nTGRAPH:\n")
	print(histo_.name)
	print(histo_.xlabel)
	print(histo_.xvalues)
	print(histo_.ylabel)
	print(histo_.yvalues)
	return 
	print([x for x in dir(histo_) if x[:1] != "_"])


#  =================================== #
#  ====    Brief: main program    ==== #
#  =================================== #
if __name__ == "__main__" :
				#
				#  Welcome
				#
	msg.info("test_uproot.py","Running program")
				#
				#  Get input files and settings
				#
	filename = "make_rootfile_output.root"
	histograms = get_histograms(filename)
	for key, histo in histograms.items() :
		print(key)
		if str(type(histo)) == "<class 'uproot.rootio.TH1'>" : read_TH1(histo)
		elif str(type(histo)) == "<class 'uproot.rootio.TH1F'>" : read_TH1(histo)
		elif str(type(histo)) == "<class 'uproot.rootio.TH1D'>" : read_TH1(histo)
		elif str(type(histo)) == "<class 'uproot.rootio.TH2'>" : read_TH2(histo)
		elif str(type(histo)) == "<class 'uproot.rootio.TH2F'>" : read_TH2(histo)
		elif str(type(histo)) == "<class 'uproot.rootio.TH2D'>" : read_TH2(histo)
		elif str(type(histo)) == "<class 'uproot.rootio.TGraphAsymmErrors'>" : read_TGraphAsymmErrors(histo)
		elif str(type(histo)) == "<class 'uproot.rootio.TGraphErrors'>" : read_TGraphErrors(histo)
		elif str(type(histo)) == "<class 'uproot.rootio.TGraph'>" : read_TGraph(histo)
		else : print("TYPE NOT RECOGNISED  {0}".format(type(histo)))
				#
				#  Goodbye
				#
	msg.info("test_uproot.py","Program reached the end without crashing and will close :) Have a nice day...")

