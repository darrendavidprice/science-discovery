# ====================================================================================================
#  Brief: data structure to hold the DistributionContainer class and associated methods
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================

import numpy as np
import configparser, yaml

import HEP_data_utils.messaging as msg
import HEP_data_utils.plotting as plotter
from HEP_data_utils.data_structures import *


#  Brief: store a number of HEPData tables, and be able to manipulate them
class DistributionContainer (object) :
	def clear_entries (self) :
		self._inclusive_distributions = {}
		self._1D_distributions = {}
		self._2D_distributions = {}
		self._ND_distributions = {}
	def clear (self) :
		self.clear_entries()
		self._name = ""
		self._make_matrix_if_possible = True
	def __init__ ( self , name_ = "" ) :
		self.clear()
		self._name = name_
	def __type__ (self) :
		return "HEP_data_utils.DistributionContainer"
	def __len__ (self) :
		return len(self._inclusive_distributions) + len(self._1D_distributions) + len(self._2D_distributions) + len(self._ND_distributions)
	def __str__ (self) :
		ret = "HEP_data_utils.DistributionContainer \"{0}\" with the following entries".format(self._name)
		ret = ret + "\n\033[1mINCLUSIVE DISTRIBUTIONS:\033[0m"
		for key, dist in self._inclusive_distributions.items() : ret = ret + "\n   key: \033[95m{0}\033[0m\n      --> name \"{1}\" with {2} bins".format(key,dist._dep_var.name(),dist.n_bins())
		ret = ret + "\n\033[1m1D DISTRIBUTIONS:\033[0m"
		for key, dist in self._1D_distributions.items() : ret = ret + "\n   key: \033[95m{0}\033[0m\n      --> name \"{1}\" with {2} bins".format(key,dist._dep_var.name(),dist.n_bins())
		ret = ret + "\n\033[1m2D DISTRIBUTIONS:\033[0m"
		for key, dist in self._2D_distributions.items() : ret = ret + "\n   key: \033[95m{0}\033[0m\n      --> name \"{1}\" with {2} bins".format(key,dist._dep_var.name(),dist.n_bins())
		ret = ret + "\n\033[1m>=3D DISTRIBUTIONS:\033[0m"
		for key, dist in self._ND_distributions.items() : ret = ret + "\n  key: \033[95m{0}\033[0m\n      --> name \"{1}\" with {2} bins".format(key,dist._dep_var.name(),dist.n_bins())
		return ret
	def __contains__( self , key_ ) :
		if key_ in self._inclusive_distributions : return True
		if key_ in self._1D_distributions : return True
		if key_ in self._2D_distributions : return True
		if key_ in self._ND_distributions : return True
		return False
	def get_table ( self , key_ ) :
		for key, table in self._inclusive_distributions.items() :
			if key == key_ : return table
		for key, table in self._1D_distributions.items() :
			if key == key_ : return table
		for key, table in self._2D_distributions.items() :
			if key == key_ : return table
		for key, table in self._ND_distributions.items() :
			if key == key_ : return table
		return None
	def __getitem__ ( self , key_ ) :
		ret = self.get_table(key_)
		if ret is None :
			raise KeyError("No distribution with key {0} in DistributionContainer {1}".format(key_,self._name))
		return ret
	def get_keys (self) :
		ret = []
		ret.append( [x for x in self._inclusive_distributions] )
		ret.append( [x for x in self._1D_distributions] )
		ret.append( [x for x in self._2D_distributions] )
		ret.append( [x for x in self._ND_distributions] )
		return ret
	def print_keys (self) :
		print(self)
	def print_all (self) :
		print("DistributionContainer \"{0}\" with the following entries".format(self._name))
		for key, dist in self._inclusive_distributions.items() :
			print("\n\033[1m\033[95mINCLUSIVE DISTRIBUTION with key {0}, name {1} and {2} bins\033[0m\n".format(key,dist._dep_var.name(),dist.n_bins()))
			print(dist)
		for key, dist in self._1D_distributions.items() :
			print("\n\033[1m\033[95m1D DISTRIBUTION with key {0}, name {1} and {2} bins\033[0m\n".format(key,dist._dep_var.name(),dist.n_bins()))
			print(dist)
		for key, dist in self._2D_distributions.items() :
			print("\n\033[1m\033[95m2D DISTRIBUTION with key {0}, name {1} and {2} bins\033[0m\n".format(key,dist._dep_var.name(),dist.n_bins()))
			print(dist)
		for key, dist in self._ND_distributions.items() :
			print("\n\033[1m\033[95mND DISTRIBUTION with key {0}, name {1} and {2} bins\033[0m\n".format(key,dist._dep_var.name(),dist.n_bins()))
			print(dist)
	def get_inclusive_keys (self) : return [ key for key in self._inclusive_distributions ]
	def get_1D_keys (self) : return [ key for key in self._1D_distributions ]
	def get_2D_keys (self) : return [ key for key in self._2D_distributions ]
	def get_ND_keys (self) : return [ key for key in self._ND_distributions ]
	def rename_key ( self , old_key_ , new_key_ ) :
		something_done = False
		old_key = r"{0}".format(old_key_)
		new_key = r"{0}".format(new_key_)
		for key in self._inclusive_distributions :
			if old_key != key : continue
			self._inclusive_distributions[new_key_] = self._inclusive_distributions.pop(old_key_)
			msg.info("HEP_data_utils.DistributionContainer.rename_key","Store \"{0}\" renaming inclusive distribution key {1} to {2}".format(self._name,old_key_,new_key_),verbose_level=1)
			something_done = True
		for key in self._1D_distributions :
			if old_key != key : continue
			self._1D_distributions[new_key_] = self._1D_distributions.pop(old_key_)
			msg.info("HEP_data_utils.DistributionContainer.rename_key","Store \"{0}\" renaming 1D distribution key {1} to {2}".format(self._name,old_key_,new_key_),verbose_level=1)
			something_done = True
		for key in self._2D_distributions :
			if old_key != key : continue
			self._2D_distributions[new_key_] = self._2D_distributions.pop(old_key_)
			msg.info("HEP_data_utils.DistributionContainer.rename_key","Store \"{0}\" renaming 2D distribution key {1} to {2}".format(self._name,old_key_,new_key_),verbose_level=1)
			something_done = True
		for key in self._ND_distributions :
			if old_key != key : continue
			self._ND_distributions[new_key_] = self._ND_distributions.pop(old_key_)
			msg.info("HEP_data_utils.DistributionContainer.rename_key","Store \"{0}\" renaming ND distribution key {1} to {2}".format(self._name,old_key_,new_key_),verbose_level=1)
			something_done = True
		if not something_done :
			msg.error("HEP_data_utils.DistributionContainer.rename_key","Store \"{0}\" with nothing done for old_key_={1}, new_key_={2}".format(self._name,old_key_,new_key_),verbose_level=1)
	def load_keys ( self , filename_ ) :
		config = configparser.ConfigParser()
		config.optionxform = str
		try : config.read(filename_)
		except Exception as exc :
			msg.check_verbosity_and_print ( exc , verbose_level=-1 )
			msg.error("HEP_data_utils.DistributionContainer.load_keys","An exception occured when parsing the config file... Continuing with nothing done")
			return
		if "KEYS" not in config.sections() :
			msg.error("HEP_data_utils.DistributionContainer.load_keys","No section titled \"KEYS\" in file {0}".format(filename_))
			return
		keys = config["KEYS"]
		for old_key in keys :
			self.rename_key(old_key,keys[old_key])
		self.print_keys()
	def generate_key ( self , table_ ) :
		table_doi = ""
		if table_._submission_file_table : table_doi = table_._submission_file_table.table_doi()
		hepdata_doi = ""
		if table_._submission_file_meta : table_._submission_file_meta.hepdata_doi()
		name = ""
		if table_._submission_file_table : table_.submission_file_table().name()
		if len(name) == 0 and type(table_._dep_var) is DependentVariable : name = name = table_._dep_var.name()
		key = ""
		if len(table_doi) > 0 : key = table_doi + "||" + name
		if len(key) == 0 and len(hepdata_doi) > 0 : key = hepdata_doi + "||" + name
		if len(key) == 0 : key = name
		if key in self :
			key = key + "-;1"
			while key in self : key = key[:-1] + str(1+int(key[-1:]))
		return r"{0}".format(key)
	def plot ( self , key_ , **kwargs ) :
		for key, dist in self._inclusive_distributions.items() :
			if key != key_ : continue
			try : plotter.plot_inclusive_distribution(dist,**kwargs)
			except Exception as e :
				print(e)
				msg.error("HEP_data_utils.DistributionContainer.plot","Error when plotting inclusive distribution with key {0}... skipping".format(key))
		for key, dist in self._1D_distributions.items() :
			if key != key_ : continue
			try : plotter.plot_1D_distribution(dist,**kwargs)
			except Exception as e :
				print(e)
				msg.error("HEP_data_utils.DistributionContainer.plot","Error when plotting 1D distribution with key {0}... skipping".format(key))
		for key, dist in self._2D_distributions.items() :
			if key != key_ : continue
			try : plotter.plot_2D_distribution(dist,**kwargs)
			except Exception as e :
				print(e)
				msg.error("HEP_data_utils.DistributionContainer.plot","Error when plotting 2D distribution with key {0}... skipping".format(key))
	def plot_all ( self , **kwargs ) :
		for d in [ self._inclusive_distributions , self._1D_distributions , self._2D_distributions ] :
			for key in d :
				self.plot(key,**kwargs)
	def plot_ratio ( self , key_num_ , key_den_ , **kwargs ) :
		table_num = self._1D_distributions.get(key_num_,None)
		if not table_num :
			msg.error("HEP_data_utils.DistributionContainer.plot_ratio","Error when plotting 1D distribution with key {0}... skipping".format(key_num_))
			raise KeyError("key {0} not in {1}".format(key_num_,self._name))
		table_den = self._1D_distributions.get(key_den_,None)
		if not table_den :
			msg.error("HEP_data_utils.DistributionContainer.plot_ratio","Error when plotting 1D distribution with key {0}... skipping".format(key_den_))
			raise KeyError("key {0} not in {1}".format(key_den_,self._name))
		try : plotter.plot_ratio(table_num,table_den,**kwargs)
		except Exception as e :
			print(e)
			msg.error("HEP_data_utils.DistributionContainer.plot_ratio","Error when plotting {0} / {1} ratio... skipping".format(key_num_,key_den_))

