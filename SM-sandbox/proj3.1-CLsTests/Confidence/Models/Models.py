#  ========================================================================
#  Brief:   Classes which store model predictions as normalised PDFs
#  Author:  Stephen Menary (stmenary@cern.ch)
#  ========================================================================


import numpy as np
import Confidence.Plotting.plotting as plotting
import matplotlib.pyplot as plt


#  Brief:   Class to hold a single template    (N.B. new-style classes in Python3 don't require (object) inheritance)
class Template :
	def normalise (self) :
		integral = sum(self.values) 
		if integral == 0 : raise ValueError("Template array {0} has an integral of 0 and cannot be normalised".format(self)) 
		self.values /= sum(self.values)
	def normalize (self) :
		self.normalise()
	def load (self, vals) :
		if type(vals) is list : self.values = np.array(vals,dtype=np.double)
		elif type(vals) is np.ndarray : self.values = np.array(vals,dtype=np.double)
		else : self.values = np.array([vals],dtype=np.double)
		for el in self.values :
			if el >= 0 : continue
			raise ValueError("Template element {0} cannot be negative".format(el))
	def __init__ (self, vals=None) :
		self.values = np.zeros(shape=(0),dtype=np.double)
		if vals is None : return
		self.load(vals)
	def __str__ (self) :
		return str(self.values)
	def __len__ (self) :
		return len(self.values)


#  Brief:   Class to hold a model as the sum of several Templates    (N.B. new-style classes in Python3 don't require (object) inheritance)
class Model :
	def num_templates (self) :
		return len(self.coefficients)
	def append (self, temp) :
		if type(temp) is not Template :
			temp = Template(temp)
		if self.length is None :
			self.length = len(temp)
		elif len(temp) is not self.length :
			raise ValueError("Cannot append a new template of length {0} where {1} was expected".format(len(temp),self.length))
		self.coefficients = np.append( self.coefficients , [1.0] )
		self.templates.append(temp)
	def __init__ (self, temps=[]) :
		self.length = None
		self.templates = []
		self.coefficients = np.array([],dtype=np.double)
		for temp in temps :
			self.append(temp)
	def __len__ (self) :
		return self.length
	def __str__ (self) :
		return "  +  ".join( [ "{0} * {1}".format(p,l) for p,l in zip(self.coefficients,self.templates) ] )
	def generate_prediction (self) :
		y = np.zeros(shape=(len(self)))
		for i in range(self.num_templates()) :
			y = y + self.coefficients[i] * self.templates[i].values
		return y
	def generate_asimov (self) :
		y = self.generate_prediction()
		ey = np.sqrt(y)
		return y, ey
	def throw_toy (self) :
		y = self.generate_prediction()
		for i in range(len(y)) :
			y[i] = np.random.poisson(y[i])
		ey = np.sqrt(y)
		return y, ey
	def plot_asimov(self, **kwargs) :
		throw_toy = kwargs.get("throw_toy",False)
		if throw_toy is True :
			y, ey = self.throw_toy()
			label = "Toy dataset"
		else :
			y, ey = self.generate_asimov()
			label = "Asimov dataset"
		x_min = kwargs.get("x_min",0.)
		x_max = kwargs.get("x_max",0.)
		labels = kwargs.get("labels",())
		num_points = len(y)
		x = np.linspace(x_min, x_max, num_points, endpoint=True)
		spacing = 0.5 * 10. / (num_points-1)
		bins = np.linspace(5.-spacing, 15.+spacing, 1+num_points, endpoint=True)
		x_new = plotting.bin_edges_to_histogram_x(bins)
		y_new = []
		for i in range(self.num_templates()) :
			y_new.append( plotting.bin_contents_to_histogram_y(self.templates[i].values*self.coefficients[i]) )
		fig  = plt.figure(figsize=(7,7))
		ax = fig.add_subplot(111)
		ax.stackplot(x_new, y_new, labels = labels)
		ax.errorbar(x, y, yerr=ey, marker='o', linestyle='', elinewidth=2, c='k', label=label)
		ax.set_xlim(bins[0], bins[-1])
		if "title" in kwargs : plt.title(kwargs["title"])
		if "xlabel" in kwargs : plt.xlabel(kwargs["xlabel"])
		if "ylabel" in kwargs : plt.ylabel(kwargs["ylabel"])
		if len(labels) > 0 : plt.legend(loc="best")
		plt.show()
		if type(plotting.document) is not None :
			fig.savefig(plotting.document, format='pdf')
		plt.close()
	def plot_toy(self, **kwargs) :
		self.plot_asimov(throw_toy=True, **kwargs)




