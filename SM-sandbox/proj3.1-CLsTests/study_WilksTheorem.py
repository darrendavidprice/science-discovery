import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

import Confidence.Models.Simple2DModel as DM
import Confidence.DM_fitting_2 as fit
import Confidence.Plotting.plotting as plotting
import Confidence.messaging as msg
import Confidence.utils as utils
from study_CLs import generate_correlation_matrix, get_CL





#  Open pickled file
#
def open_from_file (fname, keys, **kwargs) :
	if utils.file_exists(fname) is False : return False, {}
	unpickled_file = pickle.load(open(fname, "rb"))
	for option, value in kwargs.items() :
		if option not in unpickled_file : return False, {}
		if unpickled_file[option] != value : return False, {}
	ret = {}
	for key in keys :
		if key not in unpickled_file : return False, {}
		ret[key] = unpickled_file[key]
	return True, ret


#  Save pickled dict to file
#
def save_to_file (fname, dictionary, **kwargs) :
	if utils.path_exists(fname) is True :
		msg.warning("save_to_file", "Path provided {0} already exists and will be overwritten".format(fname))
	to_save = {}
	for option, value in dictionary.items() :
		to_save[option] = value
	for option, value in kwargs.items() :
		to_save[option] = value
	try :
		pickle.dump(to_save, open(fname,"w+b"))
	except Exception as exc :
		msg.error("save_to_file", "File {0} could not be written. The following error was thrown:".format(fname))
		msg.check_verbosity_and_print(exc)


#  Load a file of toys and check the metadata. Return (True, [toys]) if everything is as expected.
#
def open_toys_file (fname, **kwargs) :
	if utils.file_exists(fname) is False : return False, [], [], [], []
	unpickled_file = pickle.load(open(fname,"rb"))
	if "toys_q"  not in unpickled_file : return False, [], [], [], []
	if "toys_dTNLL"  not in unpickled_file : return False, [], [], [], []
	if "toys_k1" not in unpickled_file : return False, [], [], [], []
	if "toys_k2" not in unpickled_file : return False, [], [], [], []
	for option, value in kwargs.items() :
		if option not in unpickled_file : return False, [], [], [], []
		if unpickled_file[option] != value : return False, [], [], [], []
	return True, unpickled_file["toys_q"], unpickled_file["toys_dTNLL"], unpickled_file["toys_k1"], unpickled_file["toys_k2"]


#  Save a file of toys and metadata (as a pickled dictionary).
#
def save_toys_file (toys_q, toys_dTNLL, toys_k1, toys_k2, fname, **kwargs) :
	if utils.path_exists(fname) is True :
		msg.warning("save_toys_file","Path provided {0} already exists and will be overwritten".format(fname))
	to_save = {}
	for option, value in kwargs.items() :
		to_save[option] = value
	to_save["toys_q"], to_save["toys_dTNLL"], to_save["toys_k1"], to_save["toys_k2"] = toys_q, toys_dTNLL, toys_k1, toys_k2
	try :
		pickle.dump(to_save,open(fname,"w+b"))
	except Exception as exc :
		msg.error("save_toys_file","File {0} could not be written. The following error was thrown:".format(fname))
		msg.check_verbosity_and_print(exc)


#  Save a file of toys and metadata (as a pickled dictionary).
#
def get_q_distribution_for_kappa(model, k1, k2, **kwargs) :
	force_new_toys = kwargs.get("force_new_toys",True)
	tag = kwargs.get("tag","untagged")
	n_bins = kwargs.get("n_bins",-1)
	n_toys = kwargs.get("n_toys",-1)
	x_low  = kwargs.get("x_low",-1.)
	x_high = kwargs.get("x_high",-1.)
	rnd_seed = kwargs.get("rnd_seed",-1)
	plot_pulls = kwargs.get("plot_pulls",False)
	toys_q, toys_dTNLL, toys_k1, toys_k2 = [], [], [], []
	is_NULL_hypothesis = (k1 == 1 and k2 == 0)

	if is_NULL_hypothesis is True :
		label = tag + "(NULL)"
		fname_toys = "tmp/{0}_toys_b.dat".format(tag)
	else :
		label = tag + "(k1 = {0:.2f}, k2 = {1:.2f})".format(k1, k2)
		fname_toys = "tmp/{0}_toys_s_{1:.2f}_{2:.2f}.dat".format(tag, k1, k2)

	if force_new_toys is False :
		toys_opened_success, toys_q, toys_dTNLL, toys_k1, toys_k2 = open_toys_file(fname_toys, n_bins=n_bins, n_toys=n_toys, x_low=x_low, x_high=x_high, rnd_seed=rnd_seed )
		force_new_toys = not toys_opened_success
	
	if force_new_toys is True :
		msg.info("study_WilksTheorem.py","Throwing {0} toys around the {1} hypothesis".format(n_toys, label))
		model.coefficients[1] = k1
		model.coefficients[2] = k2
		toys_q, toys_dTNLL, toys_k1, toys_k2 = fit.throw_and_fit_toys(model, n_toys)
		msg.info("study_WilksTheorem.py","Saving {0} hypothesis toys to file {1}".format(label, fname_toys))
		save_toys_file(toys_q, toys_dTNLL, toys_k1, toys_k2, fname_toys, n_bins=n_bins, n_toys=n_toys, x_low=x_low, x_high=x_high, rnd_seed=rnd_seed)

	if False in [len(x) == n_toys for x in [toys_q, toys_dTNLL, toys_k1, toys_k2]] :
		msg.fatal("study_WilksTheorem.py","Failure loading toys for the {1} hypothesis".format(n_toys, label))
	else :
		msg.info("study_WilksTheorem.py","{0} toys successfully loaded for the {1} hypothesis".format(n_toys, label))
	return toys_q, toys_dTNLL, toys_k1, toys_k2


#  Get coverage using Wilks theorem
#
def get_coverage_wilks2(toys_dTNLL) :
	num_covered = 0
	for dq in toys_dTNLL :
		#if dq > 2.2955 : continue    # 1 sigma
		if dq > 2.2789 : continue     # 68%
		#if dq > 5.9915 : continue     # 95%
		num_covered = num_covered + 1
	return num_covered / len(toys_dTNLL)


#  Draw Asimov expected likelihood contours
#
def draw_asimov_likelihood_contours (model) :
	kappa_lo = -4
	kappa_hi = 4
	kappa_interval = 0.1
	kappa_npoints_per_dof = 1 + int((kappa_hi-kappa_lo)/kappa_interval)
	im = np.zeros(shape=(kappa_npoints_per_dof, kappa_npoints_per_dof))
	model.coefficients = np.array([1., 1., 0.])
	opened, dic = open_from_file("tmp/.study_WilksTheorem.draw_asimov_likelihood_contours.dat", ["im"], kappa_lo=kappa_lo, kappa_hi=kappa_hi, kappa_interval=kappa_interval)
	x = []
	for i in range(kappa_npoints_per_dof) :
		x.append(kappa_lo + i * kappa_interval)
	if opened is False :
		y, ey = model.generate_asimov(fabs=True)
		fit.set_fit_model(model)
		fit.set_data(y)
		TNLL_NULL = fit.get_fitted_TNLL( (1.,1.,0.), (0.1,0.1,0.1), (False,True,True) )
		for i in range(kappa_npoints_per_dof) :
			k1 = kappa_lo + i * kappa_interval
			for j in range(kappa_npoints_per_dof) :
				k2 = kappa_lo + j * kappa_interval
				TNLL = fit.get_fitted_TNLL( (1.,k1,k2), (0.1,0.1,0.1), (False,True,True) )
				im[i][j] = TNLL - TNLL_NULL
		save_to_file("tmp/.study_WilksTheorem.draw_asimov_likelihood_contours.dat", {"im":im}, kappa_lo=kappa_lo, kappa_hi=kappa_hi, kappa_interval=kappa_interval)
	else :
		im = dic["im"]
	fig  = plt.figure(figsize=(7,7))
	ax = fig.add_subplot(111)
	CS = ax.contour(x, x, im, [2.29, 5.99], colors=["navy","crimson"])
	ax.clabel(CS, fmt={2.29:r"$1\sigma$", 5.99:r"$95 \%$"}, inline=1, fontsize=10)
	ax.set_ylabel(r"$\kappa_{1}$")
	ax.set_xlabel(r"$\kappa_{2}$")
	ax.set_title(r"Asimov sensitivity  ($\kappa_{1}^{true}=1$, $\kappa_{2}^{true}=0$)")
	plt.show()
	plt.close(fig)


#
#
def draw_coverage (x, z) :
	fig = plt.figure(figsize=(7,7))
	ax = fig.add_subplot(111)
	n_points = len(x)
	radius = 0.45 * (x[-1] - x[0]) / n_points
	for i in range(n_points) :
		for j in range(n_points) :
			cov = 100*z[j][i]
			color = "k"
			if cov > 72 : color = "r"
			if cov < 64 : color = "b"
			circle = plt.Circle((x[i], x[j]), radius=radius, color=color)
			plt.gca().add_patch(circle)
			plt.text(x[i]-0.55*radius, x[j]-0.35*radius, "{0:.0f}".format(cov), fontsize=7, color="white")
	ax.set_xlim(x[0]-1, x[-1]+1)
	ax.set_ylim(x[0]-1, x[-1]+1)
	ax.grid(False)
	plt.show()




#  Main script (importable)
#
def study_WilksTheorem () :
	plotting.set_style()
	msg.VERBOSE_LEVEL = 1
	plotting.set_save_file("out/study_WilksTheorem.pdf")
	# 
	# Config constants
	# 
	n_bins = 7
	n_toys = 1000
	x_low, x_high = 5, 15
	rnd_seed = 100
	plot_asimovs = False
	do_draw_likelihood_contours = True
	force_new_toys = False
	kappa_lo = -4.
	kappa_interval = 0.5
	kappa_npoints_per_dof = 1 + int(2.*np.fabs(kappa_lo)/kappa_interval)
	# 
	# Create sig + bkg expectations and plot them
	# 
	model = DM.create_simple_2D_model_1(scale_bkg=0.8, x_low=x_low, x_high=x_high, n_bins=n_bins)
	if plot_asimovs is True :
		model.plot_asimov(x_min=5., x_max=15., y_min=0, y_max=125,  xlabel="Observable", ylabel="Events  /  bin width", labels = ("SM bkg", r"SM signal ($\kappa_{1}^{\rm true}=1$)", r"BSM signal ($\kappa_{2}^{\rm true}=1$)"), title=r"Model with 2 degrees of freedom")
	# 
	# Plot expected (Asimov) likelihood contours
	# 
	if do_draw_likelihood_contours is True :
		draw_asimov_likelihood_contours(model)
	# 
	# Set random number seed
	# 
	utils.set_numpy_random_seed(rnd_seed)
	# 
	# Load toy distribution of q for the background only case, or throw+fit them if needed   (MODEL1)
	# 
	toys_NULL, toys_dTNLL_NULL, k1_NULL, k2_NULL = get_q_distribution_for_kappa(model, 1, 0, tag="model_2D", force_new_toys=force_new_toys, n_bins=n_bins, n_toys=n_toys, x_low=x_low, x_high=x_high)
	# 
	# Load toy distribution of q for the signal cases, or throw+fit them if needed   (MODEL1)
	#
	toys_s, toys_dTNLL, toys_k1, toys_k2 = {}, {}, {}, {}
	toys_s[(1,0)], toys_dTNLL[(1,0)], toys_k1[(1,0)], toys_k2[(1,0)] = toys_NULL, toys_dTNLL_NULL, k1_NULL, k2_NULL
	signal_points = [(1,0)]
	cov = np.zeros(shape=(kappa_npoints_per_dof, kappa_npoints_per_dof))
	k_points = np.zeros(shape=(kappa_npoints_per_dof))
	for i in range(kappa_npoints_per_dof) :
		k1 = kappa_lo + i * kappa_interval
		k_points[i] = k1
		for j in range(kappa_npoints_per_dof) :
			k2 = kappa_lo + j * kappa_interval
			if k1 == 1 and k2 == 0 : continue
			signal_points.append((k1,k2))
			toys_s[(k1,k2)], toys_dTNLL[(k1,k2)], toys_k1[(k1,k2)], toys_k2[(k1,k2)] = get_q_distribution_for_kappa(model, k1, k2, tag="model_2D", force_new_toys=force_new_toys, n_bins=n_bins, n_toys=n_toys, x_low=x_low, x_high=x_high)
			cov[i][j] = get_coverage_wilks2(toys_dTNLL[(k1,k2)])
	draw_coverage(k_points, cov)
	#
	# Clean up
	#
	plotting.close_save_file()



'''
	# 
	# Look at CLs and CLs+b coverage at each mu   (MODEL1)
	#
	if do_analyse_coverage is True :
		if plot_CL_for_median_toys is True :
			for mu_sig in signal_points :
				plot_CL_profile(toys_model1_s, signal_points, mu_sig, int(n_toys/2), title="Model1: toy with median $q_{{\\rm obs}}$, $\mu_{{\\rm sig}}^{{\\rm true}} = {0:.1f}$".format(mu_sig))
		analyse_coverage(toys_model1_s, signal_points, title=r"Model1: CL < 95% interval")
	# 
	# Get distribution of toys for model2
	#
	toys_model2_NULL = get_q_distribution_for_signal_strength(model2, correlation, 0, tag="model2", force_new_toys=force_new_toys, n_bins=n_bins, n_toys=n_toys, x_low=x_low, x_high=x_high, plot_pulls=plot_pulls)
	toys_model2_s = {}
	toys_model2_s[0] = toys_model2_NULL
	for mu in signal_points[1:] :
		toys_model2_s[mu] = get_q_distribution_for_signal_strength(model2, correlation, mu, tag="model2", force_new_toys=force_new_toys, n_bins=n_bins, n_toys=n_toys, x_low=x_low, x_high=x_high, plot_pulls=plot_pulls)
	# 
	# Look at CLs and CLs+b coverage at each mu   (MODEL2)
	#
	if do_analyse_coverage is True :
		if plot_CL_for_median_toys is True :
			for mu_sig in signal_points :
				plot_CL_profile(toys_model2_s, signal_points, mu_sig, int(n_toys/2), title="Model1: toy with median $q_{{\\rm obs}}$, $\mu_{{\\rm sig}}^{{\\rm true}} = {0:.1f}$".format(mu_sig))
		analyse_coverage(toys_model2_s, signal_points, title=r"Model2: CL < 95% interval")
	# 
	# Look at comparitive sensitivities
	#
	if do_analyse_sensitivities is True :
		analyse_sensitivities(toys_model1_s, toys_model2_s, signal_points)
	#
	# Clean up
	plotting.close_save_file()
'''



#  Fall back to running script if called as __main__
#
if __name__ == "__main__" :
	study_WilksTheorem()
