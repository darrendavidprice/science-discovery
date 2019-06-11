import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

import Confidence.Models.SimpleDarkMatterModel as DM
import Confidence.DM_fitting as fit
import Confidence.Plotting.plotting as plotting
import Confidence.messaging as msg
import Confidence.utils as utils



#  Generate a random correlation matrix of size ndim X ndim
#
def generate_correlation_matrix (ndim, **kwargs) :
	ret = np.eye(ndim)
	if kwargs.get("randomise",False) is True :
		sd = kwargs.get("sd",0.1)
		for i in range(ndim) :
			for j in range(ndim) :
				if i == j :
					continue
				if i > j :
					ret[i][j] = ret[j][i]
					continue
				ret[i][j] = np.random.normal(0, sd)
	return ret


#  return fraction of sorted list which lies below q
#
def get_CL (q, sorted_dist) :
	length = float(len(sorted_dist))
	if q < sorted_dist[0] :
		return 0.
	for i in range(len(sorted_dist)) :
		if q > sorted_dist[i] : continue
		return float(i) / length
	return 1.


#  Load a file of toys and check the metadata. Return (True, [toys]) if everything is as expected.
#
def open_toys_file (fname, **kwargs) :
	if utils.file_exists(fname) is False : return False, []
	unpickled_file = pickle.load(open(fname,"rb"))
	if "toys" not in unpickled_file : return False, []
	for option, value in kwargs.items() :
		if option not in unpickled_file : return False, []
		if unpickled_file[option] != value : return False, []
	return True, unpickled_file["toys"]


#  Save a file of toys and metadata (as a pickled dictionary).
#
def save_toys_file (toys, fname, **kwargs) :
	if utils.path_exists(fname) is True :
		msg.warning("save_toys_file","Path provided {0} already exists and will be overwritten".format(fname))
	to_save = {}
	for option, value in kwargs.items() :
		to_save[option] = value
	to_save["toys"] = toys
	try :
		pickle.dump(to_save,open(fname,"w+b"))
	except Exception as exc :
		msg.error("save_toys_file","File {0} could not be written. The following error was thrown:".format(fname))
		msg.check_verbosity_and_print(exc)


#  Save a file of toys and metadata (as a pickled dictionary).
#
def get_q_distribution_for_signal_strength(model, correlation, mu_sig, **kwargs) :
	force_new_toys = kwargs.get("force_new_toys",True)
	tag = kwargs.get("tag","untagged")
	n_bins = kwargs.get("n_bins",-1)
	n_toys = kwargs.get("n_toys",-1)
	x_low  = kwargs.get("x_low",-1.)
	x_high = kwargs.get("x_high",-1.)
	rnd_seed = kwargs.get("rnd_seed",-1)
	plot_pulls = kwargs.get("plot_pulls",False)
	toys = []
	is_NULL_hypothesis = mu_sig == 0

	if is_NULL_hypothesis is True :
		label = tag + "(NULL)"
		fname_toys = "tmp/{0}_toys_b.dat".format(tag)
	else :
		label = tag + "(mu_sig = {0:.5f})".format(mu_sig)
		fname_toys = "tmp/{0}_toys_s_{1:.5f}.dat".format(tag,mu_sig)

	if force_new_toys is False :
		toys_opened_success, toys = open_toys_file(fname_toys, n_bins=n_bins, n_toys=n_toys, x_low=x_low, x_high=x_high, rnd_seed=rnd_seed )
		force_new_toys = not toys_opened_success
	
	if force_new_toys is True :
		msg.info("study_CLs.py","Throwing {0} toys around the {1} hypothesis".format(n_toys, label))
		model.coefficients[1] = mu_sig
		toys = fit.throw_and_fit_toys(model, correlation, n_toys, plot_pulls=plot_pulls, pulls_xlabel=r"$\frac{\hat{\mu}_{\rm sig} - \mu^{\rm true}_{\rm sig}}{\hat{\sigma}_{\rm sig}}$", pulls_label=r"Toys with $\mu^{\rm true}_{\rm sig} = "+"{0:.1f}$".format(mu_sig))
		msg.info("study_CLs.py","Saving {0} hypothesis toys to file {1}".format(label, fname_toys))
		save_toys_file(toys, fname_toys, n_bins=n_bins, n_toys=n_toys, x_low=x_low, x_high=x_high, rnd_seed=rnd_seed)

	if len(toys) == n_toys :
		msg.info("study_CLs.py","{0} toys successfully loaded for the {1} hypothesis".format(n_toys, label))
	return toys


#  Get coverage for a particular truth value of mu_sig
#
def get_coverage(toys_s_b, all_mu_sig, mu_sig, **kwargs) :
	coverage_CLs, coverage_CLs_b = 0, 0
	n_toys = len(toys_s_b[0])
	for q in toys_s_b[mu_sig] :
		CL_b = get_CL(q, toys_s_b[0])                  #  CL_b for this toy
		CL_s_b = get_CL(q, toys_s_b[mu_sig])
		if CL_b == 0 : CL_s = 1
		else : CL_s = CL_s_b / CL_b
		if CL_s_b > 0.05 : coverage_CLs_b = coverage_CLs_b + 1
		if CL_s > 0.05 : coverage_CLs = coverage_CLs + 1
	return coverage_CLs/n_toys, coverage_CLs_b/n_toys



#  Plot CL as a function of mu_sig for a particular toy
#
def plot_CL_profile(toys_s_b, all_mu_sig, mu_sig, k, **kwargs) :
	q = toys_s_b[mu_sig][k]
	CL_b = get_CL(q, toys_s_b[0])
	y_CLs, y_CLs_b = [], []
	for test_mu in all_mu_sig :
		CL_s_b = get_CL(q, toys_s_b[test_mu])
		try : CL_s = CL_s_b / CL_b
		except : CL_s = 1
		y_CLs.append(CL_s)
		y_CLs_b.append(CL_s_b)
	fig  = plt.figure(figsize=(7,7))
	ax = fig.add_subplot(111)
	ax.plot( all_mu_sig, y_CLs_b, '--', c='k', label=r"${\rm CL}_{s+b}$" )
	ax.plot( all_mu_sig, y_CLs, '-', c='r', label=r"${\rm CL}_{s}$" )
	ax.set_ylim(0,1.19)
	ax.set_xlim(0,all_mu_sig[-1])
	plt.ylabel("Confidence level")
	plt.xlabel(r"$\mu_{\rm sig}$")
	plt.title(kwargs.get("title", "CL profile of median toy @ $\mu_{{\\rm sig}}^{{\\rm true}} = {0:.1f}$".format(mu_sig)))
	plt.legend(loc="upper right")
	plt.show()
	if type(plotting.document) is not None :
		fig.savefig(plotting.document, format='pdf')
	plt.close()


#  Get upper limits on mu_sig using the CL_s and CLs+b methods
#
def get_upper_limits(toys_s_b, all_mu_sig, mu_sig, **kwargs) :
	msg.info("get_upper_limits", "Getting upper limits for mu_sig = {0:.4f}".format(mu_sig), verbose_level=1)
	upp_lim_CLs, upp_lim_CLs_b = [], []
	for q in toys_s_b[mu_sig] :
		lim_CLs, lim_CLs_b = 0., 0.
		CL_b = get_CL(q, toys_s_b[0])
		for test_mu in all_mu_sig :
			CL_s_b = get_CL(q, toys_s_b[test_mu])
			if CL_b == 0 : CL_s = 1
			else : CL_s = CL_s_b / CL_b
			if CL_s_b > 0.05 : lim_CLs_b = test_mu
			if CL_s > 0.05 : lim_CLs = test_mu
		upp_lim_CLs.append(lim_CLs)
		upp_lim_CLs_b.append(lim_CLs_b)
	return upp_lim_CLs, upp_lim_CLs_b


#  Coverage analysis
#
def analyse_coverage(toys_s_b, all_mu_sig, **kwargs) :
	coverage_CLs, coverage_CLs_b = [], []
	for mu_sig in all_mu_sig :
		cov_CLs, cov_CLs_b = get_coverage(toys_s_b, all_mu_sig, mu_sig, **kwargs)
		coverage_CLs.append(cov_CLs)
		coverage_CLs_b.append(cov_CLs_b)
		msg.info("analyse_coverage", "mu_sig = {0:.4f} with Coverage(CLs) = {1:.4f} and Coverage(CL_s+b) = {2:.4f}".format(mu_sig, cov_CLs, cov_CLs_b), verbose_level=1)
	fig  = plt.figure(figsize=(7,7))
	ax = fig.add_subplot(111)
	ax.plot( all_mu_sig , coverage_CLs_b , '--', c='k' , label=r"${\rm CL}_{s+b}$" )
	ax.plot( all_mu_sig , coverage_CLs , '-', c='r' , label=r"${\rm CL}_{s}$" )
	ax.set_xlim(all_mu_sig[0], all_mu_sig[-1])
	ax.set_ylim(0.8, 1.05)
	plt.ylabel("Expected coverage")
	plt.xlabel(r"$\mu_{\rm sig}^{\rm true}$")
	plt.title(kwargs.get("title","CL < 95%% interval"))
	plt.legend(loc="lower left")
	plt.show()
	if type(plotting.document) is not None :
		fig.savefig(plotting.document, format='pdf')
	plt.close()


#  Sensitivity analysis
#
def analyse_sensitivities(toys_model1, toys_model2, all_mu_sig, **kwargs) :
	limits_CLs_model1, limits_CLs_b_model1 = get_upper_limits(toys_model1, all_mu_sig, 3.)
	limits_CLs_model2, limits_CLs_b_model2 = get_upper_limits(toys_model2, all_mu_sig, 3.)
	plotting.plot_histo_from_data(80, [limits_CLs_model1, limits_CLs_model2], logy=True, xmin=0, xmax=8, xlabel=r"95 % upper confidence limit on $\mu_{\rm sig}$", ylabel="Num. toys", labels=[r"Moderate bkg model", r"Large bkg model"])


#  Main script (importable)
#
def test () :
	plotting.set_style()
	msg.VERBOSE_LEVEL = 1
	plotting.set_save_file("out/test.pdf")
	# 
	# Config constants
	# 
	n_bins = 11
	n_toys = 4000
	x_low, x_high = 5, 15
	rnd_seed = 100
	plot_asimovs = False
	plot_pulls = False
	plot_q = False
	plot_a_toy = False
	plot_CL_for_median_toys = False
	force_new_toys = False
	mu_sig_interval = 0.2
	mu_sig_npoints = 30
	do_analyse_coverage = True
	do_analyse_sensitivities = False
	# 
	# Create sig + bkg expectations for moderate and large backgrounds, and plot them
	# 
	model1 = DM.create_simple_DM_model_1(scale_bkg=0.8, x_low=x_low, x_high=x_high, n_bins=n_bins)
	model2 = DM.create_simple_DM_model_1(scale_bkg=5., x_low=x_low, x_high=x_high, n_bins=n_bins)
	if plot_asimovs is True :
		model1.plot_asimov(x_min=5., x_max=15., y_min=0, y_max=50,  xlabel="Observable", ylabel="Events", labels = ("SM bkg (moderate)", "DM signal ($\mu_{sig}=1$)"), title=r"Model1")
		model2.plot_asimov(x_min=5., x_max=15., y_min=0, y_max=260, xlabel="Observable", ylabel="Events  /  bin width", labels = ("SM bkg (large)", "DM signal ($\mu_{sig}=1$)"), title=r"Model2")
	# 
	# Create an experimental correlation between the fitted bins
	# 
	utils.set_numpy_random_seed(rnd_seed)
	correlation = generate_correlation_matrix(n_bins, randomise=False)
	# 
	# Plot a toy if desired
	#
	if plot_a_toy is True :
		model1.plot_toy(x_min=5., x_max=15., y_max=50,  xlabel="Observable", ylabel="Events", labels = ("SM bkg (moderate)", "DM signal ($\mu_{sig}=1$)"), title=r"Model1")
		model2.plot_toy(x_min=5., x_max=15., y_max=260, xlabel="Observable", ylabel="Events  /  bin width", labels = ("SM bkg (large)", "DM signal ($\mu_{sig}=1$)"), title=r"Model2")
		utils.set_numpy_random_seed(rnd_seed)
	# 
	# Evaluate the expected significance for the two models
	# 
	expected_significance_model1 = fit.get_expected_mu_sig_error(model1, correlation)
	expected_significance_model2 = fit.get_expected_mu_sig_error(model2, correlation)
	msg.info("study_CLs.py","Expected significance of model 1 is {0}".format(1./expected_significance_model1))
	msg.info("study_CLs.py","Expected significance of model 2 is {0}".format(1./expected_significance_model2))
	# 
	# Load toy distribution of q for the background only case, or throw+fit them if needed   (MODEL1)
	# 
	toys_NULL = get_q_distribution_for_signal_strength(model1, correlation, 0, tag="model1", force_new_toys=force_new_toys, n_bins=n_bins, n_toys=n_toys, x_low=x_low, x_high=x_high, plot_pulls=plot_pulls)
	if plot_q is True :
		plotting.plot_histo_from_data(400,[toys_NULL],logy=True,xmax=40, xlabel=r"Test statistic:  $q = \frac{L(s+b)}{L(b)}$", ylabel="Num. toys", labels=["Model1, NULL hypothesis"])
	# 
	# Load toy distribution of q for the signal cases, or throw+fit them if needed   (MODEL1)
	#
	toys_model1_s = {}
	toys_model1_s[0] = toys_NULL
	signal_points = [0]
	for i in range(mu_sig_npoints) :
		mu_sig = (1+i)*mu_sig_interval
		signal_points.append(mu_sig)
		toys_model1_s[mu_sig] = get_q_distribution_for_signal_strength(model1, correlation, mu_sig, tag="model1", force_new_toys=force_new_toys, n_bins=n_bins, n_toys=n_toys, x_low=x_low, x_high=x_high, plot_pulls=plot_pulls)
		if plot_q is False : continue
		plotting.plot_histo_from_data(400,[toys_model1_s[mu_sig]],logy=True,xmax=40, xlabel=r"Test statistic:  $q = \frac{L(s+b)}{L(b)}$", ylabel="Num. toys", labels=["Model1, $\mu_{{sig}}={0:.5f}$".format(mu_sig)])
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


	# Clean up
	plotting.close_save_file()



#  Fall back to running script if called as __main__
#
if __name__ == "__main__" :
	test()