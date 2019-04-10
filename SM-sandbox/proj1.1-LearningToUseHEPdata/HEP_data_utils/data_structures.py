import numpy as np

class Distribution (object) :
	def __init__ ( self ) :
		self._this_key = "unknown"
		self._other_keys = []
		self._values = np.array([])
		self._symm_errors = {}
		self._asymm_errors_up = {}
		self._asymm_errors_down = {}
		self._description = "unknown"
		self._units = "unknown"
	def __len__ ( self ) :
		return len(self._values)

class Distribution_1D (Distribution) :
	def __init__ ( self ) :
		super(Distribution_1D,self).__init__()
		self._bin_values = np.array([])
		self._bin_labels = []
	def __len__ ( self ) :
		return len(self._values)
	def __str__ ( self ) :
		ret = "1D Distribution\n   - keys:"
		for key in [self._this_key] + self._other_keys : ret = ret + "  " + key
		ret = ret + "\n   - description: " + self._description
		ret = ret + "\n   - units: " + self._units
		ret = ret + "\n   - values ({0}): ".format(len(self._values)) + str(self._values)
		for err in self._symm_errors :
			ret = ret + "\n   - symmetric error [{0}]: ".format(err) + str(self._symm_errors[err])
		for err in self._asymm_errors_up :
			ret = ret + "\n   - asymmetric error [{0}]_up  : ".format(err) + str(self._asymm_errors_up[err])
			ret = ret + "\n   - asymmetric error [{0}]_down: ".format(err) + str(self._asymm_errors_down[err])
		ret = ret + "\n   - bin labels: " + str(self._bin_labels)
		ret = ret + "\n   - bin values: " + str(self._bin_values)
		return ret

class Distribution_2D (Distribution) :
	def __init__ ( self ) :
		super(Distribution_2D,self).__init__()
		self._bin_labels_x = []
		self._bin_labels_y = []
	def __len__ ( self ) :
		return len(self._values)
	def __str__ ( self ) :
		ret = "2D Distribution\n   - keys:"
		for key in [self._this_key] + self._other_keys : ret = ret + "  " + key
		ret = ret + "\n   - description: " + self._description
		ret = ret + "\n   - units: " + self._units
		ret = ret + "\n   - values ({0}): ".format(len(self._values)) + str(self._values)
		for err in self._symm_errors :
			ret = ret + "\n   - symmetric error [{0}]: ".format(err) + str(self._symm_errors[err])
		for err in self._asymm_errors_up :
			ret = ret + "\n   - asymmetric error [{0}]_up  : ".format(err) + str(self._asymm_errors_up[err])
			ret = ret + "\n   - asymmetric error [{0}]_down: ".format(err) + str(self._asymm_errors_down[err])
		ret = ret + "\n   - bin labels (x,{0}): ".format(len(self._bin_labels_x)) + str(self._bin_labels_x)
		ret = ret + "\n   - bin labels (y,{0}): ".format(len(self._bin_labels_y)) + str(self._bin_labels_y)
		return ret


class Dataset (object) :
	def __init__ ( self ) :
		self._distributions = {}
		self._correlations = {}
		self._description = "unknown"
		self._location = "unknown"
		self._comment = "unknown"
		self._hepdata_doi = "unknown"
	def __len__ ( self ) :
		return len(self._distributions) + len(self._correlations)