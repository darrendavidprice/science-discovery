# ===================================================================================================================================
#  Brief: functions which allow us to load ROOT files into HEPDataTables
#  Author: Stephen Menary (stmenary@cern.ch)
# ===================================================================================================================================


import os, uproot
import HEP_data_utils.messaging as msg
import HEP_data_utils.general_helpers as hlp
from HEP_data_utils.data_structures import *


#  Brief: store a ROOT indep_var axis
class ROOT_axis :
	def __init__ (self) :
		self.name = ""
		self.bin_labels = []
		self.bin_centers = np.zeros(shape=(0))
		self.bin_widths_lo = np.zeros(shape=(0))
		self.bin_widths_hi = np.zeros(shape=(0))
	def __str__ (self) :
		ret = "ROOT_axis: {0}".format(self.name)
		ret = ret + "\n  -  bin labels  = " + str(self.bin_labels)
		ret = ret + "\n  -  bin centres = " + str(self.bin_centers)
		ret = ret + "\n  -  bin widths (low)  = " + str(self.bin_widths_lo)
		ret = ret + "\n  -  bin widths (high) = " + str(self.bin_widths_hi)
		return ret
	def __len__ (self) : return len(self.bin_centers)


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
	def __len__ (self) : return len(self.values)


#  Brief: store a ROOT table
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
	else :
		try : key = "{0}{1}".format(pre_key,in_.name)
		except :
			try : key = "{0}{1}".format(pre_key,in_.title)
			except : key = "{0}/unknown".format(pre_key)
		ret [ key ] = in_
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
	if histo_.title is not None : ret._name = str(histo_.title)
		# create dep_var
	dep_var = ROOT_observable()
	dep_var.name = str(histo_._fYaxis._fTitle)
	dep_var.values = histo_.values
	dep_var.errors_up = np.sqrt( histo_.variances )
	dep_var.errors_dn = dep_var.errors_up
	ret.dep_var = dep_var
		# create indep_var
	indep_var = ROOT_axis()
	indep_var.name = str(histo_._fXaxis._fTitle)
	indep_var.bin_labels = []
	if histo_.xlabels is not None : indep_var.bin_labels = [ str(x) for x in histo_.xlabels ]
	if len(indep_var.bin_labels) == 0 : indep_var.bin_labels = [ "" for i in range(len(dep_var.values)) ]
	bin_edges = histo_.edges
	indep_var.bin_centers = np.zeros(shape=(len(bin_edges)-1))
	for i in range(len(indep_var.bin_centers)) :
		indep_var.bin_centers[i] = 0.5 * ( bin_edges[i] + bin_edges[i+1] )
	indep_var.bin_widths_lo = np.array( [ indep_var.bin_centers[i] - bin_edges[i] for i in range(len(indep_var.bin_centers)) ] )
	indep_var.bin_widths_hi = np.array( [ bin_edges[i+1] - indep_var.bin_centers[i] for i in range(len(indep_var.bin_centers)) ] )
	ret.indep_vars.append(indep_var)
	return ret


#  Brief: turn an uproot.rootio.TH2F or TH2D object into an instance of ROOT_Table
def get_ROOT_Table_from_uproot_TH2 ( histo_ ) :
		# create table
	ret = ROOT_Table()
	if histo_.title is not None : ret._name = str(histo_.title)
		# add dep_var
	dep_var = ROOT_observable()
	dep_var.name = str(histo_._fZaxis._fTitle)
	dep_var.values = histo_.values
	dep_var.errors_up = np.sqrt( histo_.variances )
	dep_var.errors_dn = dep_var.errors_up
	ret.dep_var = dep_var
		# add indep_var for x-axis
	indep_var_x = ROOT_axis()
	indep_var_x.name = histo_._fXaxis._fTitle
	indep_var_x.bin_labels = []
	if histo_.xlabels is not None : indep_var_x.bin_labels = [ str(x) for x in histo_.xlabels ]
	if len(indep_var_x.bin_labels) == 0 : indep_var_x.bin_labels = [ "" for i in range(histo_.xnumbins) ]
	bin_edges_x = histo_.edges[0]
	indep_var_x.bin_centers = np.zeros(shape=(len(bin_edges_x)-1))
	for i in range(len(indep_var_x.bin_centers)) : indep_var_x.bin_centers[i] = 0.5 * ( bin_edges_x[i] + bin_edges_x[i+1] )
	indep_var_x.bin_widths_lo = np.array( [ indep_var_x.bin_centers[i] - bin_edges_x[i] for i in range(len(indep_var_x.bin_centers)) ] )
	indep_var_x.bin_widths_hi = np.array( [ bin_edges_x[i+1] - indep_var_x.bin_centers[i] for i in range(len(indep_var_x.bin_centers)) ] )
	ret.indep_vars.append(indep_var_x)
		# add indep_var for y-axis
	indep_var_y = ROOT_axis()
	indep_var_y.name = histo_._fYaxis._fTitle
	indep_var_y.bin_labels = []
	if histo_.ylabels is not None : indep_var_y.bin_labels = [ str(x) for x in histo_.ylabels ]
	if len(indep_var_y.bin_labels) == 0 : indep_var_y.bin_labels = [ "" for i in range(histo_.ynumbins) ]
	bin_edges_y = histo_.edges[1]
	indep_var_y.bin_centers = np.zeros(shape=(len(bin_edges_y)-1))
	for i in range(len(indep_var_y.bin_centers)) : indep_var_y.bin_centers[i] = 0.5 * ( bin_edges_y[i] + bin_edges_y[i+1] )
	indep_var_y.bin_widths_lo = np.array( [ indep_var_y.bin_centers[i] - bin_edges_y[i] for i in range(len(indep_var_y.bin_centers)) ] )
	indep_var_y.bin_widths_hi = np.array( [ bin_edges_y[i+1] - indep_var_y.bin_centers[i] for i in range(len(indep_var_y.bin_centers)) ] )
	ret.indep_vars.append(indep_var_y)
		# return table
	return ret


#  Brief: turn an uproot.rootio.TGraphErrors object into an instance of ROOT_Table
def get_ROOT_Table_from_uproot_TGraphErrors ( graph_ ) :
		# create table
	ret = ROOT_Table()
	if graph_.title is not None : ret._name = str(graph_.title)
		# create dep_var
	dep_var = ROOT_observable()
	if graph_.ylabel is not None : dep_var.name = str(graph_.ylabel)
	dep_var.values = graph_.yvalues
	dep_var.errors_up = graph_.yerrors
	dep_var.errors_dn = dep_var.errors_up
	ret.dep_var = dep_var
		# create indep_var
	indep_var = ROOT_axis()
	if graph_.xlabel is not None : indep_var.name = str(graph_.xlabel)
	indep_var.bin_labels = [ "" for i in range(len(dep_var.values)) ]
	indep_var.bin_centers = graph_.xvalues
	indep_var.bin_widths_lo = graph_.xerrors
	indep_var.bin_widths_hi = indep_var.bin_widths_lo
	ret.indep_vars.append(indep_var)
	return ret


#  Brief: turn an uproot.rootio.TGraphAsymmErrors object into an instance of ROOT_Table
def get_ROOT_Table_from_uproot_TGraphAsymmErrors ( graph_ ) :
		# create table
	ret = ROOT_Table()
	if graph_.title is not None : ret._name = str(graph_.title)
		# create dep_var
	dep_var = ROOT_observable()
	if graph_.ylabel is not None : dep_var.name = str(graph_.ylabel)
	dep_var.values = graph_.yvalues
	dep_var.errors_up = graph_.yerrorshigh
	dep_var.errors_dn = graph_.yerrorslow
	ret.dep_var = dep_var
		# create indep_var
	indep_var = ROOT_axis()
	if graph_.xlabel is not None : indep_var.name = str(graph_.xlabel)
	indep_var.bin_labels = [ "" for i in range(len(dep_var.values)) ]
	indep_var.bin_centers = graph_.xvalues
	indep_var.bin_widths_lo = graph_.xerrorslow
	indep_var.bin_widths_hi = graph_.xerrorshigh
	ret.indep_vars.append(indep_var)
	return ret


#  Brief: turn an uproot.rootio.TGraph object into an instance of ROOT_Table
def get_ROOT_Table_from_uproot_TGraph ( graph_ ) :
		# create table
	ret = ROOT_Table()
	if graph_.title is not None : ret._name = str(graph_.title)
		# create dep_var
	dep_var = ROOT_observable()
	if graph_.ylabel is not None : dep_var.name = str(graph_.ylabel)
	dep_var.values = graph_.yvalues
	dep_var.errors_up = np.zeros(shape=len(dep_var.values))
	dep_var.errors_dn = np.zeros(shape=len(dep_var.values))
	ret.dep_var = dep_var
		# create indep_var
	indep_var = ROOT_axis()
	if graph_.xlabel is not None : indep_var.name = str(graph_.xlabel)
	indep_var.bin_labels = [ "" for i in range(len(dep_var.values)) ]
	indep_var.bin_centers = graph_.xvalues
	indep_var.bin_widths_lo = np.zeros(shape=(len(indep_var.bin_centers)))
	indep_var.bin_widths_hi = np.zeros(shape=(len(indep_var.bin_centers)))
	ret.indep_vars.append(indep_var)
	return ret


#  Brief: turn an uproot.rootio.TGraph2DErrors object into an instance of ROOT_Table
def get_ROOT_Table_from_uproot_TGraph2DErrors ( graph_ ) :
		# create table
	ret = ROOT_Table()
	if graph_._fTitle is not None : ret._name = str(graph_._fTitle)
	elif graph_._fName is not None : ret._name = str(graph_._fName)
		# create dep_var
	dep_var = ROOT_observable()
	dep_var.values = graph_._fZ
	dep_var.errors_up = graph_._fEZ
	dep_var.errors_dn = dep_var.errors_up
	ret.dep_var = dep_var
		# create indep_var_x
	indep_var_x = ROOT_axis()
	indep_var_x.bin_centers = graph_._fX
	indep_var_x.bin_widths_hi = graph_._fEX
	indep_var_x.bin_widths_lo = indep_var_x.bin_widths_hi
	indep_var_x.bin_labels = [ "" for i in range(len(indep_var_x.bin_centers)) ]
	ret.indep_vars.append(indep_var_x)
		# create indep_var_y
	indep_var_y = ROOT_axis()
	indep_var_y.bin_centers = graph_._fY
	indep_var_y.bin_widths_hi = graph_._fEY
	indep_var_y.bin_widths_lo = indep_var_y.bin_widths_hi
	indep_var_y.bin_labels = [ "" for i in range(len(indep_var_y.bin_centers)) ]
	ret.indep_vars.append(indep_var_y)
	return ret


#  Brief: turn raw uproot data into data structures
def get_uproot_histograms ( data_ ) :
	ret = {}
	for key, entry in data_.items() :
		if str(type(entry)) == "<class 'uproot.rootio.TH1F'>" : ret[key] = get_ROOT_Table_from_uproot_TH1(entry)
		elif str(type(entry)) == "<class 'uproot.rootio.TH1D'>" : ret[key] = get_ROOT_Table_from_uproot_TH1(entry)
		elif str(type(entry)) == "<class 'uproot.rootio.TH2F'>" : ret[key] = get_ROOT_Table_from_uproot_TH2(entry)
		elif str(type(entry)) == "<class 'uproot.rootio.TH2D'>" : ret[key] = get_ROOT_Table_from_uproot_TH2(entry)
		elif str(type(entry)) == "<class 'uproot.rootio.TGraph'>" : ret[key] = get_ROOT_Table_from_uproot_TGraph(entry)
		elif str(type(entry)) == "<class 'uproot.rootio.TGraphErrors'>" : ret[key] = get_ROOT_Table_from_uproot_TGraphErrors(entry)
		elif str(type(entry)) == "<class 'uproot.rootio.TGraphAsymmErrors'>" : ret[key] = get_ROOT_Table_from_uproot_TGraphAsymmErrors(entry)
		elif str(type(entry)) == "<class 'uproot.rootio.TGraph2DErrors'>" : ret[key] = get_ROOT_Table_from_uproot_TGraph2DErrors(entry)
		else :
			msg.info("HEP_data_utils.ROOT_helpers.get_uproot_histograms","Object {0} of type {1} is ignored as I don't understand this format".format(key,type(entry)),verbose_level=0)
	return ret


#  Brief: convert a ROOT_observable object into a data_structures.DependentVariable one
def ROOT_observable_to_DependentVariable ( ROOT_y_ ) :
	ret = DependentVariable()
	ret._name = ROOT_y_.name
	ret._values = ROOT_y_.values.astype(np.float32)
	ret._asymerrors_up["err0"] = ROOT_y_.errors_up.astype(np.float32)
	ret._asymerrors_dn["err0"] = -1.0 * ROOT_y_.errors_dn.astype(np.float32)
	return ret


#  Brief: convert a ROOT_axis object into a data_structures.IndependentVariable one
def ROOT_axis_to_IndependentVariable ( ROOT_x_ ) :
	ret = IndependentVariable()
	ret._name = ROOT_x_.name
	ret._bin_labels = ROOT_x_.bin_labels
	ret._bin_centers = ROOT_x_.bin_centers.astype(np.float32)
	ret._bin_widths_lo = ROOT_x_.bin_widths_lo.astype(np.float32)
	ret._bin_widths_hi = ROOT_x_.bin_widths_hi.astype(np.float32)
	return ret


#  Brief: load ROOT_Table contents into a HEPDataTable object within dataset_ (of type DistributionContainer)
def load_distribution_from_ROOT_Table ( dataset_ , key_ , table_ , **kwargs ) :
	# Create the table object
	table = HEPDataTable()
	table._submission_file_meta = kwargs.get("submission_file_metadata",None)
	table._submission_file_table = kwargs.get("submission_file_table",None)
	# Set the variables
	table._dep_var = ROOT_observable_to_DependentVariable(table_.dep_var)
	for indep_var in table_.indep_vars :
		table._indep_vars.append(ROOT_axis_to_IndependentVariable(indep_var))
	# Figure out what key to give it
	if key_ in dataset_ :
		key_ = key_ + "-;1"
		while key_ in self : key_ = key_[:-1] + str(1+int(key_[-1:]))
	# Validate our table
	is_valid, validity_message = table.is_valid()
	if not is_valid :
		msg.error("ROOT_helpers.load_distribution_from_ROOT_Table","Error occured when loading table {0}... returning with nothing done.".format(key_))
		msg.error("ROOT_helpers.load_distribution_from_ROOT_Table",">>>>>>>>>>>>>>")
		msg.error("ROOT_helpers.load_distribution_from_ROOT_Table",validity_message)
		msg.error("ROOT_helpers.load_distribution_from_ROOT_Table","<<<<<<<<<<<<<<")
		return
	# Add to dataset
	n_dim = table.n_indep_vars()
	if n_dim == 0 : dataset_._inclusive_distributions[key_] = table
	elif n_dim == 1 : dataset_._1D_distributions[key_] = table
	elif n_dim == 2 : dataset_._2D_distributions[key_] = table
	else : dataset_._ND_distributions[key_] = table


#  Brief: use uproot to load a single root file based on the file path
def load_root_file ( dataset_ , path_ , **kwargs) :
	msg.info("ROOT_helpers.load_root_file","Opening root file {0}".format(path_),verbose_level=0)
	raw_uproot_data = open_root_file(path_,path_)
	uproot_histos = get_uproot_histograms(raw_uproot_data)
	for key, table in uproot_histos.items() :
		load_distribution_from_ROOT_Table(dataset_,key,table,**kwargs)


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



