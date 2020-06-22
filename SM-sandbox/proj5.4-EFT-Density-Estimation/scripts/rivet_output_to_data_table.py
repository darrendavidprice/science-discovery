#   ==================================================================================
#   Brief : Convert Rivet output file to a pickled DataTable file 
#           - can be run as a script
#   Author: Stephen Menary (sbmenary@gmail.com)
#   ==================================================================================


#  Required imports

import argparse, sys

sys.path.append("/Users/Ste/PostDoc/git-with-DP/SM-sandbox/proj5.4-EFT-Density-Estimation/backends")
from data_preparation import DataTable


#  Convert the dat file to a pickle file, and plot the contents if requested
#
def convert_dat_to_pickle (in_fname, out_fname, show, save) :
	#
	#  Load input file
	#
	sys.stdout.write(f"Loading input file {in_fname}...")
	data = DataTable(in_fname)
	sys.stdout.write(" done.\n\n")
	sys.stdout.flush()
	#
	#  Print it's contents to terminal
	#
	print(f"Printing contents...")
	data.print_summary()
	sys.stdout.flush()
	#
	#  Save the pickle file
	#
	sys.stdout.write(f"\nSaving data to pickle file {out_fname}...")
	data.save_to_pickle_file(out_fname)
	sys.stdout.write(" done.\n")
	sys.stdout.flush()
	#
	#  Make plot
	#
	if show or type(save) != type(None) :
		print(f"\nPlotting contents...")
		data.plot_contents(show=show, save=save)


#  Parse the command-line arguments using argparse module
#
def parse_args () :
	parser = argparse.ArgumentParser(description="Convert a .dat file, created by the Rivet routine, into a pickle file storing a DataTable object.")
	parser.add_argument("input"      , type=str           , help="Name of input file to run over."      )
	parser.add_argument("output"     , type=str           , help="Name of output file to create."       )
	parser.add_argument("--plot"     , action="store_true", help="Plot the data entries.", default=False)
	parser.add_argument("--save_plot", type=str           , help="Plot save file."       , default=""   )
	return parser.parse_args()


#  Parse the command line arguments, and call convert_dat_to_pickle()
#
def run_program () :
	args = parse_args ()
	in_fname, out_fname, show, save = args.input, args.output, args.plot, args.save_plot
	if len(save) == 0 : save = None
	convert_dat_to_pickle (in_fname, out_fname, show, save)


#  Fallback: call run_program() if module executedf as a script
#
if __name__ == "__main__" :
	run_program()
