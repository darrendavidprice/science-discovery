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
		self.bin_labels = []
		self.bin_centers = np.zeros(shape=(0))
		self.bin_edges = np.zeros(shape=(0))
	def __str__ (self) :
		ret = "ROOT_axis: {0}".format(self.name)
		ret = ret + "\n  -  bin labels  = " + str(self.bin_labels)
		ret = ret + "\n  -  bin centres = " + str(self.bin_centers)
		ret = ret + "\n  -  bin edges   = " + str(self.bin_edges)
		return ret


#  Brief: store a ROOT dep_var
class ROOT_observable :
	def __init__ (self) :
		self.name = ""
		self.values = np.zeros(shape=(0))
		self.errors_up = np.zeros(shape=(0))
		self.errors_dn = np.zeros(shape=(0))
	def __str__ (self) :
		ret = "ROOT_observable: {0}".format(self.name)
		ret = ret + "\n  -  values = " + str(self.values)
		ret = ret + "\n  -  errors up   = " + str(self.errors_up)
		ret = ret + "\n  -  errors down = " + str(self.errors_dn)
		return ret


#  Brief: create a TH1F
class ROOT_Table :
	def __init__ (self) :
		self.name = ""
		self.dep_var = ROOT_observable()
		self.indep_vars = []
	def __str__ (self) :
		ret = "ROOT_Table: {0}".format(self.name)
		ret = ret + "\n" + str(self.dep_var)
		for indep_var in self.indep_vars : ret = ret + "\n" + str(indep_var)
		return ret


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


#  Brief: turn an uproot.rootio.TH1F or TH1D object into an instance of ROOT_Table
#     useful members are:   histo_.name   OR   histo_._fName  =  name
#                           histo_.xlabels  =  list of xlabels (or None)
#                           histo_.numbins  =  number of bins (excluding under/overflow)
#                           histo_.allbins  =  list of [lower bin edge, upper bin edge] including under/overflow
#                           histo_.bins  =  list of [lower bin edge, upper bin edge] excluding under/overflow
#                           histo_.alledges  =  list of bin edges including under/overflow
#                           histo_.edges  =  list of bin edges excluding under/overflow
#                           histo_.allvalues  =  list of bin contents including under/overflow
#                           histo_.values  =  list of bin contents excluding under/overflow
#                           histo_.allvariances   OR   hist_._fSumw2  =  list of bin errors squared including under/overflow
#                           histo_.variances  =  list of bin errors squared excluding under/overflow
#                           histo_.underflows  =  content of underflow bin
#                           histo_.overflows  =  content of overflow bin
#                           histo_.low  =  lowest value of x-axis
#                           histo_.high  =  highest value of x-axis
#                           histo_._fEntries  =  number of entries
#                           histo_._fXaxis  =  x-axis as uproot.rootio.TAxis object
#                           histo_._fYaxis  =  y-axis as uproot.rootio.TAxis object
#                           histo_._fZaxis  =  z-axis as uproot.rootio.TAxis object
def get_ROOT_Table_from_uproot_TH1 ( histo_ ) :
		# create table
	ret = ROOT_Table()
	ROOT_Table._name = histo_.name
		# create dep_var
	dep_var = ROOT_observable()
	dep_var.name = str(histo_._fYaxis._fTitle)
	dep_var.values = np.array(histo_.values)
	dep_var.errors_up = np.sqrt( np.array(histo_.variances) )
	dep_var.errors_dn = dep_var.errors_up
	ret.dep_var = dep_var
		# create indep_var
	indep_var = ROOT_axis()
	indep_var.name = histo_._fXaxis._fTitle
	indep_var.bin_labels = []
	if histo_.xlabels is not None : indep_var.bin_labels = [ str(x) for x in histo_.xlabels ]
	if len(indep_var.bin_labels) == 0 : indep_var.bin_labels = [ "" for i in range(len(dep_var.values)) ]
	indep_var.bin_edges = np.array(histo_.edges)
	indep_var.bin_centers = np.zeros(shape=(len(indep_var.bin_edges)-1))
	for i in range(len(indep_var.bin_centers)) :
		indep_var.bin_centers[i] = 0.5 * ( indep_var.bin_edges[i] + indep_var.bin_edges[i+1] )
	ret.indep_vars.append(indep_var)
	return ret


#  Brief: turn an uproot.rootio.TH2F or TH2D object into an instance of ROOT_Table
def get_ROOT_Table_from_uproot_TH2 ( histo_ ) :
		# create table
	ret = ROOT_Table()
	ROOT_Table._name = histo_.name
		# add dep_var
	dep_var = ROOT_observable()
	dep_var.name = str(histo_._fYaxis._fTitle)
	dep_var.values = np.array(histo_.values)
	dep_var.errors_up = np.sqrt( np.array(histo_.variances) )
	dep_var.errors_dn = dep_var.errors_up
	ret.dep_var = dep_var
		# add indep_var for x-axis
	indep_var_x = ROOT_axis()
	indep_var_x.name = histo_._fXaxis._fTitle
	indep_var_x.bin_labels = []
	if histo_.xlabels is not None : indep_var_x.bin_labels = [ str(x) for x in histo_.xlabels ]
	if len(indep_var_x.bin_labels) == 0 : indep_var_x.bin_labels = [ "" for i in range(histo_.xnumbins) ]
	indep_var_x.bin_edges = np.array(histo_.edges[0])
	indep_var_x.bin_centers = np.zeros(shape=(len(indep_var_x.bin_edges)-1))
	for i in range(len(indep_var_x.bin_centers)) :
		indep_var_x.bin_centers[i] = 0.5 * ( indep_var_x.bin_edges[i] + indep_var_x.bin_edges[i+1] )
	ret.indep_vars.append(indep_var_x)
		# add indep_var for y-axis
	indep_var_y = ROOT_axis()
	indep_var_y.name = histo_._fYaxis._fTitle
	indep_var_y.bin_labels = []
	if histo_.ylabels is not None : indep_var_y.bin_labels = [ str(x) for x in histo_.ylabels ]
	if len(indep_var_y.bin_labels) == 0 : indep_var_y.bin_labels = [ "" for i in range(histo_.ynumbins) ]
	indep_var_y.bin_edges = np.array(histo_.edges[1])
	indep_var_y.bin_centers = np.zeros(shape=(len(indep_var_y.bin_edges)-1))
	for i in range(len(indep_var_y.bin_centers)) :
		indep_var_y.bin_centers[i] = 0.5 * ( indep_var_y.bin_edges[i] + indep_var_y.bin_edges[i+1] )
	ret.indep_vars.append(indep_var_y)
		# return table
	return ret


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


#  Brief: turn an uproot.rootio.TH2F or TH2D object into an instance of ROOT_Table
def get_ROOT_Table_from_uproot_TGraph ( graph_ ) :
		# create table
	ret = ROOT_Table()
	ROOT_Table._name = graph_.name
		# create dep_var
	dep_var = ROOT_observable()
	dep_var.name = str(graph_.ylabel)
	dep_var.values = np.array(graph_.yvalues)
	dep_var.errors_up = np.zeros(shape=len(dep_var.values))
	dep_var.errors_dn = np.zeros(shape=len(dep_var.values))
	ret.dep_var = dep_var
		# create indep_var
	indep_var = ROOT_axis()
	indep_var.name = str(graph_.xlabel)
	indep_var.bin_labels = [ "" for i in range(len(dep_var.values)) ]
	indep_var.bin_centers = np.array(graph_.xvalues)

	indep_var.bin_edges = np.array(histo_.edges)
	indep_var.bin_centers = np.zeros(shape=(len(indep_var.bin_edges)-1))
	for i in range(len(indep_var.bin_centers)) :
		indep_var.bin_centers[i] = 0.5 * ( indep_var.bin_edges[i] + indep_var.bin_edges[i+1] )
	ret.indep_vars.append(indep_var)
	return ret


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
	ret = {}
	for key, entry in data_.items() :
		if str(type(entry)) == "<class 'uproot.rootio.TH1F'>" : ret[key] = get_ROOT_Table_from_uproot_TH1(entry)
		if str(type(entry)) == "<class 'uproot.rootio.TH1D'>" : ret[key] = get_ROOT_Table_from_uproot_TH1(entry)
		if str(type(entry)) == "<class 'uproot.rootio.TH2F'>" : ret[key] = get_ROOT_Table_from_uproot_TH2(entry)
		if str(type(entry)) == "<class 'uproot.rootio.TH2D'>" : ret[key] = get_ROOT_Table_from_uproot_TH2(entry)
		if str(type(entry)) == "<class 'uproot.rootio.TGraph'>" : ret[key] = get_ROOT_Table_from_uproot_TGraph(entry)
		print(key,type(entry))


#  Brief: use uproot to load a single root file based on the file path
def load_root_file ( dataset_ , path_ , **kwargs ) :
	msg.info("ROOT_helpers.load_root_file","Opening root file {0}".format(path_),verbose_level=0)
	raw_uproot_data = open_root_file(path_,path_)
	uproot_histos = get_uproot_histograms(raw_uproot_data)
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



