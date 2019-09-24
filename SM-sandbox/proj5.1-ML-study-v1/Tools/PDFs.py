import numpy         as np
import scipy.stats   as stats
import scipy.special as special


rcParams = {
	"x_low"  : 105.,
	"x_high" : 160.,
	"N_sig"  : 1500.,
	"N_bkg"  : 300000.,
	"bkg_a"  : 0.,
	"bkg_b"  : 1.,
	"bkg_c"  : 0.,
	"sig_mass"  : 125.,
	"sig_width" : 1.8,
}


def resolve_rcParam (key, argument, rogue=None) :
	if argument == rogue :
		global rcParams
		return rcParams[key]
	return argument


class BkgPDF :
	def __init__ (self, a=None, b=None, c=None) :
		self.a = resolve_rcParam("bkg_a", a, rogue=None)
		self.b = resolve_rcParam("bkg_b", b, rogue=None)
		self.c = resolve_rcParam("bkg_c", c, rogue=None)
	def integral (self, x_low, x_high, x_range_low=None, x_range_high=None) :
		x_range_low  = resolve_rcParam("x_low" , x_range_low , rogue=None)
		x_range_high = resolve_rcParam("x_high", x_range_high, rogue=None)
		return self.unnormalised_integral(x_low, x_high) / self.unnormalised_integral(x_range_low,x_range_high)
	def pdf (self, x, x_low=None, x_high=None) :
		x_low  = resolve_rcParam("x_low" , x_low , rogue=None)
		x_high = resolve_rcParam("x_high", x_high, rogue=None)
		return self.unnormalised_eval(x) / self.unnormalised_integral(x_low, x_high)
	def unnormalised_eval (self, x) :
		return self.a + self.b/(x-self.c)
	def unnormalised_integral (self, x_low, x_high) :
		return self.a*(x_high-x_low) + self.b*(np.log(x_high-self.c) - np.log(x_low-self.c))


class SigPDF :
	def __init__ (self, mass=None, width=None) :
		self.mass  = resolve_rcParam("sig_mass" , mass , rogue=None)
		self.width = resolve_rcParam("sig_width", width, rogue=None)
	def pdf (self, x, x_low=None, x_high=None) :
		return stats.norm.pdf(x, self.mass, self.width)
	def integral (self, x_low, x_high) :
		return -0.5*( special.erf((self.mass-x_high)/(np.sqrt(2)*self.width)) - special.erf((self.mass-x_low)/(np.sqrt(2)*self.width)) )

class PDF :
	def __init__ (self, *argv, **kwargs) :
		self.bkgPDF = None
		self.sigPDF = None
		self.N_sig  = float(resolve_rcParam("N_sig", kwargs.get("N_sig", None), rogue=None))
		self.N_bkg  = float(resolve_rcParam("N_bkg", kwargs.get("N_bkg", None), rogue=None))
		for arg in argv :
			if type(arg) == BkgPDF :
				self.bkgPDF = arg
				continue
			if type(arg) == SigPDF :
				self.sigPDF = arg
				continue
			raise ArgumentError(f"{type(self)} argument \'{arg}\' of type {type(arg)} not recognised")
		if "a" in kwargs or "b" in kwargs :
			if "a" not in kwargs : raise ArgumentError(f"{type(self)} with specified argument \'b\'' also requires speified \'a\'")
			if "b" not in kwargs : raise ArgumentError(f"{type(self)} with specified argument \'a\'' also requires speified \'b\'")
			if self.bkgPDF != None  : raise ArgumentError(f"cannot specify \'a\' or \'b\' and also provide an object of type {type(BkgPDF)} for the constructor of {type(self)} - please choose one or the other")
			self.bkgPDF = BkgPDF(a=kwargs["a"], b=kwargs["b"])
		if self.bkgPDF == None :
			self.bkgPDF = BkgPDF()
		if "mass" in kwargs or "width" in kwargs :
			if self.sigPDF != None  : raise ArgumentError(f"cannot specify \'mass\' or \'width\' and also provide an object of type {type(SigPDF)} for the constructor of {type(self)} - please choose one or the other")
		if self.sigPDF == None :
			self.sigPDF = SigPDF(kwargs.get("mass", None), kwargs.get("width", None))
	def absolute_eval (self, x, x_low=None, x_high=None) :
		return self.N_sig*self.sigPDF.pdf(x, x_low=x_low, x_high=x_high) + self.N_bkg*self.bkgPDF.pdf(x, x_low=x_low, x_high=x_high)
	def pdf (self, x, x_low=None, x_high=None) :
		return self.absolute_eval(x, x_low=x_low, x_high=x_high) / (self.N_sig + self.N_bkg)
	def integral (self, x_low, x_high) :
		return (self.N_sig*self.sigPDF.integral(x_low, x_high) + self.N_bkg*self.bkgPDF.integral(x_low, x_high)) / (self.N_sig + self.N_bkg)
	def generate_binned_dataset (self, bins, asimov=False, extended=True) :
		x_low, x_high = bins[:-1], bins[1:]
		probabilities = self.integral(x_low, x_high)
		N_total = self.N_sig + self.N_bkg
		if extended : N_total = np.random.poisson(N_total)
		expected_yields =  N_total*probabilities
		if asimov   : return expected_yields
		for idx in range(len(expected_yields)) :
			expected_yields[idx] =  np.random.poisson(expected_yields[idx])
		return expected_yields





