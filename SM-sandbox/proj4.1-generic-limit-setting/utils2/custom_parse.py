import sys
import getopt

import utils2.utils.utils                 as     utils
import utils2.utils.globals_and_fallbacks as     glob
import utils2.utils.plotting              as     plotting



##  Parse command line arguments
#
def parse_command_line_arguments (*argv, **kwargs) :
	utils.info("parse_command_line_arguments()", "Parsing arguments")
	try :
		opts, rest = getopt.getopt(sys.argv[1:], "", [f"{k}=" for k in kwargs] + ["save=", "tag=", "show="] + [f"{a}=" for a in argv] )
	except getopt.GetoptError as err :
		utils.error("parse_command_line_arguments()", "The following error was thrown whilst parsing command-line arguments")
		utils.fatal("parse_command_line_arguments()", err)
	if len(rest) is not 1 :
		raise ValueError(f"parse_command_line_arguments(): expected 1 unlabelled argument where {len(argv)} provided")
	cfg_name = rest[0]
	save_fname, do_show, tag = None, True, None
	ret = {}
	if not utils.is_file(cfg_name) :
		raise RuntimeError(f"parse_command_line_arguments(): config file {cfg_name} not found")
	for option, value in opts :
		if option in ["--tag"]  :
			tag = str(value)
			utils.info("parse_command_line_arguments()", f"Labelling temporary files using the tag: {tag}")
		if option in ["--save"] :
			save_fname = str(value)
			utils.info("parse_command_line_arguments()", f"Opening plots file {save_fname}")
			plotting.open_plots_pdf(save_fname)
		if option in ["--show"] :
			do_show = utils.string_to_object(value)
			if type(do_show) != bool : raise ValueError(f"parse_command_line_arguments(): --show value \"{value}\" could not be cast to a bool")
		if option in argv :
			ret[option] = True

	glob.custom_store["config name"   ] = cfg_name
	glob.custom_store["do show plots" ] = do_show
	glob.custom_store["plots filename"] = save_fname
	glob.custom_store["tag"           ] = tag
	return ret