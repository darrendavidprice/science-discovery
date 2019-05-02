# ===================================================================================================================================
#  Brief: functions which allow us to load ROOT files into HEPDataTables
#  Author: Stephen Menary (stmenary@cern.ch)
# ===================================================================================================================================


import os, yaml, uproot
import HEP_data_utils.messaging as msg
import HEP_data_utils.general_helpers as hlp
from HEP_data_utils.data_structures import *


#  Brief: store a ROOT indep_var axis
class ROOT_axis :
	def __init__ (self) :
		self.name = ""
		self.values = []
		self.labels = []
		self.edges = np.zeros(shape=(0))


#  Brief: store a ROOT dep_var
class ROOT_observable :
	def __init__ (self) :
		self.name = ""
		self.values = np.zeros(shape=(0))
		self.errors_up = np.zeros(shape=(0))
		self.errors_dn = np.zeros(shape=(0))


#  Brief: create a TH1F
class ROOT_Table :
	def __init__ (self) :
		self.name = ""
		self.dep_var = ROOT_observable()
		self.indep_vars = []


#  Brief: open a root file at the specified path and return a list of it's contents in uproot format
def open_root_file ( in_ , pre_key = "" ) :
	if len(pre_key) > 0 :
		if pre_key[-1:] != "/" : pre_key = pre_key + "/"
	ret = {}
	if type(in_) is str :
		f = uproot.open(in_)
		ret = { **ret , **open_root_file(f,"{0}{1}".format(pre_key,in_)) }
	elif type(in_) is uproot.rootio.ROOTDirectory :
		for key, value in in_.items() :
			for key2, value2 in open_root_file(value,"{0}{1}".format(pre_key,key)).items() :
				if key2 in ret : key2 = key2 + ";1"
				while key2 in ret :
					num = int(key2.split(";").pop())
					key2 = key2 + str(num+1)
				ret[key2] = value2
	else : ret [ "{0}{1}".format(pre_key,in_.name) ] = in_
	return ret


#  Brief: open a root file at the specified path and return a list of it's contents in uproot format
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


#  Brief: turn raw uproot data into data structures
def get_uproot_histograms ( data_ ) :
	exit()


#  Brief: use uproot to load a single root file based on the file path
def load_root_file ( dataset_ , path_ , **kwargs ) :
	msg.info("ROOT_helpers.load_root_file","Opening root file {0}".format(path_),verbose_level=0)
	raw_uproot_data = open_root_file(path_,path_)
	uproot_histos = get_uproot_histograms(raw_uproot_data)
	print(data)
	exit()


#  Brief: load root files based on the file path
def load_all_root_files ( dataset_ , path_ , **kwargs ) :
	do_recurse = kwargs.get("recurse",True)
	path_ = hlp.remove_subleading(path_,"/")
	if hlp.is_directory(path_) :
		root_files = hlp.keep_only_root_files(path_,recurse=do_recurse)
		if len(root_files) == 0 :
			msg.error("ROOT_helpers.load_all_root_files","Directory {0} has no root files... returning with nothing done.".format(path_),verbose_level=-1)
			return
		for f in root_files : load_all_root_files(dataset_,f,**kwargs)
		return
	if not hlp.is_root_file(path_) :
		msg.error("ROOT_helpers.load_all_root_files","Path {0} is not a directory or root file... returning with nothing done.".format(path_),verbose_level=-1)
		return
	load_root_file(dataset_,path_,**kwargs)


#  Brief: load root files based on a directory path or file list
def load_root_files_from_list ( dataset_ , dir_ , **kwargs ) :
	if hlp.is_directory(dir_) :
		for filename in [ dir_+"/"+f for f in os.listdir(dir_) if is_root_file(f) ] :
			msg.info("ROOT_helpers.load_root_files_from_list","Opening root file {0}".format(filename),verbose_level=0)
			load_all_root_files(dataset_,filename,**kwargs)
	elif type(dir_) == list :
		for filename in dir_ :
			if type(filename) != str : continue
			msg.info("ROOT_helpers.load_root_files_from_list","Opening yaml file {0}".format(filename),verbose_level=0)
			load_all_root_files(dataset_,filename,**kwargs)
	else :
		msg.error("ROOT_helpers.load_root_files_from_list","Input {0} is neither a directory nor a list... returning with nothing done".format(dir_),verbose_level=-1)



