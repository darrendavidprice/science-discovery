import numpy as np

class Distribution (object) :
	def __init__ ( self ) :
		self._this_key = "unknown"
		self._other_keys = []
		self._values = np.array([0])
		self._errors = {}
		self._description = "unknown"
		self._units = "unknown"
	def __len__ ( self ):
		return len(self._values)


class Dataset (object) :
	def __init__ ( self ) :
		self._distributions = {}
		self._covariances = {}
		self._description = "unknown"
		self._location = "unknown"
		self._comment = "unknown"
		self._hepdata_doi = "unknown"
	def __len__ ( self ) :
		return len(self._distributions) + len(self._covariances)