##  =======================================================================================================================
##  Brief: reproduce CLs limits from EPJC (2017) 77:765 (Figure 6, top): DMEFT D7a EFT limits
##         - instead of the full grid of m_chi vs. lambda, we use only one BSM input at m_chi = 10 GeV, lambda = 400 GeV
##         - a 1D scan as a function of lambda can be obtained by scaling this prediction by (lambda [GeV]/400)**6, as the EFT operator has no kinematic lambda dependence
##         - since m_chi = 10 GeV, we are effectively trying to reproduce a horizontal slice at the very bottom of the plot referenced above
##  Run command: python3 quick_test.py
##  External dependencies: numpy, scipy, matplotlib, pickle
##  Author: Stephen Menary
##  Email: sbmenary@gmail.com
##  =======================================================================================================================

import sys

import matplotlib.pyplot as plt
import numpy             as np
import scipy.stats       as stats
import pickle


#  ====================================================================================================
#      GLOBAL: READ DATA, SM AND BSM PREDICTIONS FROM PICKLE FILE
#  ====================================================================================================

inputs = pickle.load(open("paper_values_and_DMEFT_D7a_m10_l400_prediction.pickle", "rb"))

meas         = inputs["meas_values"]              #  This is the measurement (central values)
meas_cov     = inputs["meas_cov"]                 #  This is the measurement (covariance)
meas_cov_inv = np.linalg.inv(meas_cov)            #  This is the inverse covariance
n_dof        = len(meas)                          #  This is the number of bins (=24)

SM           = inputs["SM_model_values"]          #  This is the SM expectation (central values)
SM_exp_cov   = inputs["SM_exp_cov"]               #  This is the expected experimental covariance
SM_thry_cov  = inputs["SM_model_cov"]             #  This is the covariance on the central values due to theory modelling
SM_thry_cov  = np.zeros(shape=(n_dof, n_dof))               # Uncomment this line if you want to ignore theory uncertainty on SM
SM_total_cov = SM_exp_cov + SM_thry_cov           #  This is the sum of the theory covariance with the expected experimental covariance
SM_exp_cov_inv   = np.linalg.inv(SM_exp_cov)      
SM_total_cov_inv = np.linalg.inv(SM_total_cov)

BSM_400     = inputs["BSM400_values"]             #  This is Ste's MG5 (LO) expectation for DMEFT D7a: m_chi = 10 GeV, lambda = 400 GeV
BSM_400_cov = inputs["BSM400_cov"]                #  This is the covariance on the expectation
BSM_400_cov = np.zeros(shape=(n_dof, n_dof))               # Uncomment this line if you want to ignore theory uncertainty on BSM model


#  ====================================================================================================
#      DEFINE SOME UTILITY FUNCTIONS
#  ====================================================================================================

#  Calculate chi2 between two distributions with a specified covariance
#
def get_chi2 (meas, pred, cov=None, cov_inv=None) :
	if cov is None and cov_inv is None :
		raise ValueError("get_chi2(): must provide cov or cov_inv argument (cov is ignored if both are provided)")
	if cov_inv is None :
		cov_inv = np.linalg.inv(cov)
	res = meas - pred
	return np.matmul(res, np.matmul(cov_inv, res))

#  Calculate the confidence level of a chi2 test-statistic assuming that it follows the standard distribution
#
def get_frequentist_CL (chi2) :
	return 1.0 - stats.chi2.cdf(chi2, n_dof)

#  Calculate the CLs limit associated with a given measurement
#  - method: profile lambda, create a BSM prediction at each lambda value, add it to the SM and calculate CLs. Return value of lambda where profile crosses (1-coverage)
#
def get_CLs_limit(this_meas, this_cov, coverage=0.95) :
	if coverage <= 0 or coverage >= 1 : raise ValueError("get_CLs_limit(): provided coverage {coverage} is out of the allowed range [0, 1]")
	lambda_linspace = np.linspace(500, 1000, 51)
	CLs = []
	for this_lambda in lambda_linspace :
		scale_factor = (400./this_lambda) ** 6     #   using model where BSM part of the prediction scales as lambda^{-6}
		BSM          = BSM_400     * scale_factor
		BSM_cov      = BSM_400_cov * scale_factor * scale_factor
		chi2_BSM     = get_chi2(this_meas, SM + BSM, this_cov + SM_thry_cov + BSM_cov)
		chi2_SM      = get_chi2(this_meas, SM      , this_cov + SM_thry_cov)
		CLs.append( get_frequentist_CL(chi2_BSM) / get_frequentist_CL(chi2_SM) )
	return np.interp([1.0 - coverage], CLs, lambda_linspace)[0]    # interpolate profile of lambda vs. CLs to find where CLs=(1-coverage)

#  Throw toys around a distribution according to the given covariance
#  - method: diagonalise covariance matrix and throw orthognal shifts down its eigendirections
#
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

#  Save the model values to a pickle file
#
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
	pickle.dump(dict_to_save, open(".quick_test_3__models.pickle","wb"))


#  ====================================================================================================
#      ACTUAL SCRIPT
#  ====================================================================================================

#  MAIN SCRIPT
#  -  calculate observed CLs limit
#  -  calculate expected CLs limit (by setting data exactly to SM expectation)
#  -  throw toys around SM (expected covariance)
#      -  check that toys were throw correctly by showing that their spread can be used to reproduce the chi2 distribution
#      -  investigate the spread of CLs limits obtained from the toys
def quick_test () :
	#
	#  Print chi2 probability comparing measurements with SM
	#
	obs_chi2_prob = get_frequentist_CL(get_chi2(meas, SM, meas_cov + SM_thry_cov))
	print(f"chi2 probability of measurement vs. SM is {obs_chi2_prob}")
	#
	#  Print observed CLs limit
	#
	obs_limit = get_CLs_limit(meas, meas_cov)
	print(f"Obs. 95% limit is {obs_limit:.1f}")
	#
	#  Print expected CLs limit (where data = SM)
	#
	SM_limit = get_CLs_limit(SM, SM_exp_cov)
	print(f"Exp. 95% limit is {SM_limit:.1f}")
	#
	#  Throw toys around SM expectation
	#      - toy variations are drawn from the measured covariance
	#
	n_toys           = 10000
	toy_cov          = SM_exp_cov
	toy_cov_inv      = np.linalg.inv(toy_cov)
	toy_measurements = throw_toys(SM, toy_cov, n_toys=n_toys)
	#
	#  Get distribution of chi2s comparing the toys with the SM (using the same cov as used to generate the toys)
	#      - if no bug, we expect closure (i.e. we can reproduce the chi2 distribution with appropriate num degrees of freedom)
	#
	print("Evaluating toy chi2s as a cross-check...")
	SM_toy_chi2s = [get_chi2(t, SM, cov_inv=toy_cov_inv) for t in toy_measurements]
	#
	#  Plot the chi2 distribution to check that closure is obtained
	#
	plt.hist(SM_toy_chi2s, bins=np.linspace(0, 60, 120), label="Toys around SM")
	plt.plot(np.linspace(0,60,500), 0.5*n_toys*stats.chi2.pdf(np.linspace(0,60,500), n_dof), linestyle="--", label=f"$\chi^{2}$ (ndof = {n_dof})")
	plt.gca().set_xlabel("$\\chi^{2}$ (toy vs. SM)")
	plt.gca().set_ylabel("Num. toys")
	plt.gca().set_xlim(0, 60)
	plt.title("Check: do SM toys reproduce a $\\chi^{2}$ distribution when compared with SM?")
	plt.legend(loc="upper right")
	plt.show()
	plt.close()
	#
	#  Get distribution of CLs limits for all of our toys, and put them in a sorted list
	#      - the covariance of each toy is assumed to be that expected by the SM
	#      - this is because I don't think it makes sense for measured and expected syst covariances to differ
	#        (they would be contradictory statements about our belief in the same unknown quantities)
	#      - strictly speaking, the expected stat covariance should vary with the expectated event yield, but I am ignoring this for now
	#
	SM_toy_limits = []
	sys.stdout.write("Evaluating toy confidence limits...")
	for idx, t in enumerate(toy_measurements) :
		SM_toy_limits.append(get_CLs_limit(t, SM_exp_cov))
		if 100*(idx+1) % n_toys != 0 : continue
		sys.stdout.write("\rEvaluating toy confidence limits...  {:.0f}%".format(100*(idx+1)/n_toys))
		sys.stdout.flush()
	sys.stdout.write("\n")
	SM_toy_limits.sort()
	#
	#  Print the median and +/-1sigma variations
	#
	print(f"Exp. 95% limit [MEDIAN toys] is {SM_toy_limits[int(0.5*len(SM_toy_limits))]:.1f}")
	print(f"Exp. 95% limit [16% toys] is {SM_toy_limits[int(0.16*len(SM_toy_limits))]:.1f}")
	print(f"Exp. 95% limit [84% toys] is {SM_toy_limits[int(0.84*len(SM_toy_limits))]:.1f}")
	#
	#  Plot the CLs limit distribution
	#
	plt.hist(SM_toy_limits, bins=np.linspace(500, 1000, 100), label="Toys around SM")
	plt.axvline(obs_limit, linestyle="--", color="darkred"  , label="observed")
	plt.axvline(SM_limit , linestyle="--", color="darkgreen", label="SM")
	plt.gca().set_xlabel("$CL_{s}$ limit on $\Lambda_{EFT}$")
	plt.gca().set_ylabel("Num. toys")
	plt.gca().set_xlim(500, 1000)
	plt.legend(loc="upper left")
	plt.show()
	plt.close()
	#
	#  Save the model to a pickle file in case you want to use it for benchmarking other code
	#      - useful when we use fake data for this script
	#
	do_save_model = False
	if do_save_model : save_model_to_pickle()
	#
	#  End
	#


#  Fallback: ensures that quick_test() is run automatically when this file is called as a script (but not if it is imported into some other code)
#
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




