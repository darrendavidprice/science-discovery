import os, sys, yaml
import general_utils.messaging as msg
import general_utils.helpers as hlp
import HEP_data_utils.helpers as HEPData_hlp
from HEP_data_utils.data_structures import *


def print_help () :
	msg.info("study_DM_paper.py:print_help","Usage is: python study_DM_paper.py <single yaml-file OR directory with yaml-files OR submission.yaml steering file>")
	msg.info("study_DM_paper.py:print_help","I'm currently configured specifically for the DM paper format")

if __name__ == "__main__" :
	#  Welcome, check arguments and set verbosity (to be a config in future)
	msg.info("study_DM_paper.py","Running program")
	msg.VERBOSE_LEVEL = -1
	if len(sys.argv) != 2 :
		print_help()
		msg.fatal("study_DM_paper.py","{0} argument(s) provided where 2 were expected".format(len(sys.argv)))
	if sys.argv[1] == "--help" or sys.argv[1] == "-h" :
		print_help()
		exit()
	#  Create dataset container and load from yaml file(s)
	dataset = Distribution_store("Darren DM paper")
	HEPData_hlp.load_dataset ( dataset , sys.argv[1] )
	msg.info("study_DM_paper.py","Dataset loaded with the following entries")
	dataset.print_keys()
	#  Set a table key to something intelligible. If I don't know what info is contained within a file, I will print it's metadata to find out. This will help me set a nicely informative key.
	dataset.print_meta("|10.17182/hepdata.78366.v2/t1|Table1.yaml|measured $R^\\text{miss}$|")
	dataset.rename("|10.17182/hepdata.78366.v2/t1|Table1.yaml|measured $R^\\text{miss}$|","R_pT_miss_geq1j_meas")
	#  Do for the rest of the tables. I put them in a file to make our life easier.
	dataset.load_keys("DM_paper_keys.dat")
	dataset.print_keys()
	#  Let's see what we have loaded using print(dataset) or dataset.print_all()
	print(dataset)
	#  Set 2D table keys so I know what distribution they correspond to. First I need to print the meta_data so I know what they are.
	dataset.print_meta("stat_corr")
	dataset._distributions_2D["stat_corr"].set_local_key("garbage",0,10)
	dataset.print_keys()
	# oops, we set a rubbish key, let's remove it...
	dataset._distributions_2D["stat_corr"].remove_local_key("garbage")
	dataset.print_keys()
	# if you want to change a local key name, you can
	dataset._distributions_2D["stat_corr"].set_local_key("pT_miss",0,6)
	dataset._distributions_2D["stat_corr"].change_local_key("pT_miss","pT_miss_geq1j")
	dataset.print_keys()
	# now to set the rest...
	dataset._distributions_2D["stat_corr"].set_local_key("pT_miss_VBF",7,12)
	dataset._distributions_2D["stat_corr"].set_local_key("m_jj_VBF",13,17)
	dataset._distributions_2D["stat_corr"].set_local_key("dphi_jj_VBF",18,23)
	dataset.copy_2D_local_keys("stat_corr","stat_cov","syst_cov_lep_eff","syst_cov_jets","syst_cov_W->taunu_CR","syst_cov_multijet","syst_cov_corr_fac_stat","syst_cov_W_stat","syst_cov_W_theory","syst_cov_top_xs","syst_cov_Z->ll_bkg","syst_cov_total")
	dataset.print_keys()
#	make some plots
	dataset.plot_1D_distribution("R_pT_miss_geq1j_meas",legend_loc="lower left",ylabel="R_miss (>=1j region)",xlabel="pT_miss  [GeV]",label="Measured",xlim=[200,1400],ylim=[0,13])
	dataset.plot_data_vs_prediction("R_pT_miss_geq1j_meas","R_pT_miss_geq1j_exp",legend_loc="lower left",ylabel="R_miss (>=1j region)",xlabel="pT_miss  [GeV]",xlim=[200,1400],ylim=[0,13])
	dataset.plot_data_vs_prediction("R_pT_miss_VBF_meas","R_pT_miss_VBF_exp",legend_loc="lower left",ylabel="R_miss (VBF region)",xlabel="pT_miss  [GeV]",xlim=[200,1400],ylim=[0,13])
	dataset.plot_data_vs_prediction("R_m_jj_VBF_meas","R_m_jj_VBF_exp",legend_loc="lower left",ylabel="R_miss (VBF region)",xlabel="m_jj  [GeV]",xlim=[200,4000],ylim=[0,13])
	dataset.plot_data_vs_prediction("R_dphi_jj_VBF_meas","R_dphi_jj_VBF_exp",legend_loc="lower left",ylabel="R_miss (VBF region)",xlabel="Dphi_jj  [GeV]",xlim=[0,3.15],ylim=[0,13])
	dataset.plot_matrix("stat_corr",title="Statistical correlation",flt_precision=1)
	dataset.plot_matrix("stat_cov",title="Statistical covariance",flt_precision=1)
	dataset.plot_matrix("syst_cov_jets",title="Systematic covariance (jets)",flt_precision=2)
	dataset.plot_matrix("syst_cov_Z->ll_bkg",title="Systematic covariance (Z->ll bkg)",flt_precision=2)
	dataset.plot_matrix("syst_cov_total",title="Systematic covariance (total)",flt_precision=1)
#	make_plot_m_jj_correlated(dataset)
#	get_global_chi2(dataset)
#	create submatrix and get a local chi2 (1 or 2 distributions)
