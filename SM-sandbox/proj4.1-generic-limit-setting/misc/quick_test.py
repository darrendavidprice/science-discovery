import sys

import matplotlib.pyplot as plt
import numpy             as np
import scipy.stats       as stats
import pickle


#  ====================================================================================================
#      CREATE HAND-PICKED FAKE DATA (2 bins with correlation)
#  ====================================================================================================

meas         = np.array( [13.1, 10.9, 9.0] )
meas_err     = np.array( [0.5 , 0.9 , 0.8] )
meas_corr    = np.array( [[1.0, 0.5, -0.3], [0.5, 1.0, 0.6], [-0.3, 0.6, 1.0]] )
meas_cov     = np.matmul(np.diag(meas_err), np.matmul(meas_corr, np.diag(meas_err)))
meas_cov_inv = np.linalg.inv(meas_cov)   # cheaper to do this once now, rather than everytime we evaluate a chi2 (N.B. I am ignoring prediction errors, assuming they are small compared with experimental ones)
n_dof = len(meas)

SM             = np.array( [13.5, 11.6, 8.7] )
SM_exp_err     = np.array( [0.35, 1.0 , 0.7] )
SM_exp_corr    = meas_corr
SM_exp_cov     = np.matmul(np.diag(SM_exp_err), np.matmul(SM_exp_corr, np.diag(SM_exp_err)))
SM_exp_cov_inv = np.linalg.inv(SM_exp_cov)   # cheaper to do this once now, rather than everytime we evaluate a chi2 (N.B. I am ignoring prediction errors, assuming they are small compared with experimental ones)

SM_thry_err  = np.random.normal(0, 0.1, (3, 3))
SM_thry_cov  = np.diag(np.multiply(SM_thry_err, SM_thry_err))

BSM_400     = np.array([1., 2., 5.])
BSM_400_err = np.random.normal(0, 0.1, (3, 3))
BSM_400_cov = np.diag(np.multiply(BSM_400_err, BSM_400_err))


#  ====================================================================================================
#      DEFINE SOME UTILITY FUNCTIONS
#  ====================================================================================================

def get_chi2 (meas, pred, cov=None, cov_inv=None) :
	if cov is None and cov_inv is None : raise ValueError("get_chi2(): must provide cov or cov_inv argument (cov is ignored if both are provided)")
	if cov_inv is None : cov_inv = np.linalg.inv(cov)
	res = meas - pred
	return np.matmul(res, np.matmul(cov_inv, res))

def get_frequentist_CL (chi2) :
	return 1.0 - stats.chi2.cdf(chi2, 2)

def get_CLs_limit(this_meas, this_cov, coverage=0.95) :
	if coverage <= 0 or coverage >= 1 : raise ValueError("get_CLs_limit(): provided coverage {coverage} is out of the allowed range [0, 1]")
	x = np.linspace(300, 700, 81)
	CLs = []
	for xt in x :
		scale_factor = (400./xt) ** 6     #   using model where BSM part of the prediction scales as x^{-6}
		BSM      = BSM_400 * scale_factor
		BSM_cov  = BSM_400_cov * scale_factor
		chi2_BSM = get_chi2(this_meas, BSM + SM, cov=this_cov + BSM_cov + SM_thry_cov)
		chi2_SM  = get_chi2(this_meas, SM      , cov=this_cov + SM_thry_cov)
		CLs.append( get_frequentist_CL(chi2_BSM) / get_frequentist_CL(chi2_SM) )
	return np.interp([1.0 - coverage], CLs, x)[0]

def throw_toys(central_values, cov, n_toys=100) :
	w, v = np.linalg.eig(cov)
	w = np.sqrt(w)
	toy_shifts = np.random.normal(0, 1, (n_toys, len(central_values)))  #arguments are mean, std. dev., shape of object to create
	toys = []
	for i in range(n_toys) :
		toys.append(np.matmul(v, np.multiply(w, toy_shifts[i])) + central_values)
		if 100*(i+1) % n_toys != 0 : continue
		sys.stdout.write("\rThrowing toys {:.0f}%".format(100*(i+1)/n_toys))
		sys.stdout.flush()
	sys.stdout.write("\n")
	return toys

def save_model_to_pickle () :
	print("Saving test values to pickle file so that results of this script can be used as a benchmark to test larger framework")
	dict_to_save = {
					"meas"                : meas,
					"meas_cov"            : meas_cov,
					"SM"                  : SM,
					"SM_exp_cov"          : SM_exp_cov,
					"SM_theory_cov"       : SM_thry_cov,
					"BSM_400"             : BSM_400,
					"BSM_400_cov"         : BSM_400_cov,
					"BSM_plus_SM_400"     : SM + BSM_400,
					"BSM_plus_SM_400_cov" : SM_thry_cov + BSM_400_cov
	}
	pickle.dump(dict_to_save, open("stupid_test/.test_values.pickle","wb"))


#  ====================================================================================================
#      ACTUAL SCRIPT
#  ====================================================================================================

def quick_test () :
	print(f"chi2 probability of measurement vs. SM is {get_frequentist_CL(get_chi2(meas, SM, meas_cov))}")
	obs_limit = get_CLs_limit(meas, meas_cov)
	print(f"Obs. 95% limit is {obs_limit:.1f}")
	SM_limit = get_CLs_limit(SM, SM_exp_cov)
	print(f"Exp. 95% limit is {SM_limit:.1f}")

	n_toys = 5000
	toy_measurements = throw_toys(SM, SM_exp_cov, n_toys=n_toys)

	print("Evaluating toy chi2s as a cross-check...")
	SM_toy_chi2s = [get_chi2(t, SM, cov_inv=SM_exp_cov_inv) for t in toy_measurements]
	plt.hist(SM_toy_chi2s, bins=np.linspace(0, 10, 50), label="Toys around SM")
	plt.plot(np.linspace(0,10,500), 0.2*n_toys*stats.chi2.pdf(np.linspace(0,10,500), n_dof), linestyle="--", label=f"$\chi^{2}$ (ndof = {n_dof})")
	plt.gca().set_xlabel("$\\chi^{2}$")
	plt.gca().set_ylabel("Num. toys")
	plt.gca().set_xlim(0, 10)
	plt.title("Check: do SM toys reproduce a $\\chi^{2}$ distribution when compared with SM?")
	plt.legend(loc="upper right")
	plt.show()
	plt.close()

	sys.stdout.write("Evaluating toy confidence limits...")
	SM_toy_limits = []
	for idx, t in enumerate(toy_measurements) :
		SM_toy_limits.append(get_CLs_limit(t, SM_exp_cov))
		if 100*(idx+1) % n_toys != 0 : continue
		sys.stdout.write("\rEvaluating toy confidence limits...  {:.0f}%".format(100*(idx+1)/n_toys))
		sys.stdout.flush()
	sys.stdout.write("\n")
	SM_toy_limits.sort()
	h = plt.hist(SM_toy_limits, bins=np.linspace(400, 700, 75), label="Toys around SM")
	plt.axvline(obs_limit, linestyle="--", color="darkred", label="observed")
	plt.axvline(SM_limit, linestyle="--", color="darkgreen", label="SM")
	plt.gca().set_xlabel("$CL_{s}$ limit on $x$")
	plt.gca().set_ylabel("Num. toys")
	plt.gca().set_xlim(400, 700)
	plt.legend(loc="upper right")
	plt.show()
	plt.close()

	print(f"Exp. 95% limit [MEDIAN toys] is {SM_toy_limits[int(0.5*len(SM_toy_limits))]:.1f}")
	print(f"Exp. 95% limit [16% toys] is {SM_toy_limits[int(0.16*len(SM_toy_limits))]:.1f}")
	print(f"Exp. 95% limit [84% toys] is {SM_toy_limits[int(0.84*len(SM_toy_limits))]:.1f}")

	save_model_to_pickle()

if __name__ == "__main__" :
	quick_test()














'''
	bin_centers = [0.5*(h[1][i] + h[1][i+1]) for i in range(len(h[0]))]
	v = np.interp([obs_limit], bin_centers, h[0])[0]
	integral = sum([entry for entry in h[0] if entry <= v])

	p_value_of_limit = np.interp([obs_limit], SM_toy_limits, np.linspace(0, 1, n_toys))[0]
	if p_value_of_limit > 0.5 : p_value_of_limit = 1.0 - p_value_of_limit
	print(p_value_of_limit)
	print(integral/n_toys)
'''