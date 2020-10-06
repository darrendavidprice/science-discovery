#   ==================================================================================
#   Brief : Create Rivet output file dat file with specified name
#           - cleans up the YODA file output, which we don't want
#   Author: Stephen Menary (sbmenary@gmail.com)
#   ==================================================================================

import argparse, docker, os, shutil, subprocess



#  Parse the command-line arguments using argparse module
#
def parse_args () :
	parser = argparse.ArgumentParser(description="Run the Rivet routine over a list of inputs, and specify the .dat file it creates.")
	parser.add_argument("inputs"  , type=str  , help="Name of input file to run over." , nargs="+")
	parser.add_argument("output"  , type=str  , help="Name of output file to create."  , default="event_parser.dat")
	parser.add_argument("routine" , type=str  , help="Path to the rivet routine"       , default="event_parser.cxx")
	parser.add_argument("name"    , type=str  , help="Name of rivet routine"           , default="event_parser"    )
	return parser.parse_args()


#  Run a Rivet routine using our docker image
#
def run_rivet (inputs, routine_name="event_parser", docker_image="hepstore/rivet:2.7.2", buildplugin=None) :   # "hepstore/rivet:2.7.2"  or   "hepstore/rivet"
	cwd = os.getcwd()
	client = docker.from_env()
	print(f"Registered Rivet inputs : {', '.join(inputs)}")
	print(f"Registered Rivet routine: {routine_name}")
	if type(buildplugin) != type(None) :
		print(f"Building Rivet plugin {buildplugin} inside docker image '{docker_image}' - warning, we will only print the output at the very end, so the terminal will be silent until Rivet has finished")
		stdout = client.containers.run(docker_image, 
									   f"rivet-buildplugin {buildplugin}",    # rivet-build  or  rivet-buildplugin
									   remove=True, 
									   volumes={cwd: {'bind': cwd, 'mode': 'rw'}}, 
									   working_dir=cwd)
		print("Printing rivet-buildplugin output")
		stdout = str(stdout)[2:-1]
		print(stdout.replace(r"\n", "\n").replace(r"\t", "\t"))
	print(f"Running Rivet inside docker image '{docker_image}' - warning, we will only print the output at the very end, so the terminal will be silent until Rivet has finished")
	stdout = client.containers.run(docker_image, 
								   f"rivet --pwd -a {routine_name} {' '.join(inputs)}", 
								   remove=True, 
								   volumes={cwd: {'bind': cwd, 'mode': 'rw'}}, 
								   working_dir=cwd)
	print("Printing Rivet output")
	stdout = str(stdout)[2:-1]
	print(stdout.replace(r"\n", "\n").replace(r"\t", "\t"))


#  Specify a bunch of files to copy / remove
#
def tidy_up_files (copy_files=[], remove_files=[]) :
	#  Make sure valid arguments were provided
	if len(copy_files) + len(remove_files) == 0 :
		raise RuntimeWarning("tidy_up_files() called with no arguments")
		return
	for pair in copy_files :
		if (type(pair) not in [list, tuple]) or (len(pair) != 2 ) :
			raise RuntimeError("Expected copy_files for be a list of pairs, e.g. [('old_fname1', 'new_fname1'), ('old_fname2', 'new_fname2')]")
	#  Make sure input files exist
	for fname in [x[0] for x in copy_files] + remove_files  :
		if os.path.isfile(fname) : continue
		raise RuntimeError(f"File {fname} does not exist")
	#  Copy files
	for pair in copy_files :
		shutil.copyfile(pair[0], pair[1])
	#  Remove files
	for fname in remove_files :
		os.remove(fname)


#  Parse the command line arguments, call run_rivet(), then tidy up its ouputs using tidy_up_files()
#
def run_program () :
	args = parse_args()
	run_rivet(args.inputs, buildplugin=args.routine, routine_name=args.name)
	tidy_up_files(copy_files   = [("event_parser_output.dat", args.output)], 
				  remove_files = ["event_parser_output.dat"])  #  "Rivet.yoda"


#  Fallback: call run_program() if module executedf as a script
#
if __name__ == "__main__" :
	run_program()

