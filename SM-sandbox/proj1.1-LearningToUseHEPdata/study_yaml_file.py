import os, sys, yaml
import general_utils.messaging as msg
import general_utils.helpers as hlp
import HEP_data_utils.helpers as HEPData_hlp
from HEP_data_utils.data_structures import *


if __name__ == "__main__" :
	msg.info("study_yaml.py","Running program")
	msg.VERBOSE_LEVEL = 0
	if len(sys.argv) != 2 :
		msg.fatal("study_yaml.py","{0} argument(s) provided where 2 were expected".format(len(sys.argv)))
	in_str = sys.argv[1]
	in_str = hlp.remove_subleading(in_str,"/")
	dataset = Dataset()
	if os.path.isdir(in_str) :
		msg.info("study_yaml.py","{0} is a directory... I am expanding the entries (but will only go one directory deep!)".format(in_str))
		if os.path.isfile(in_str+"/submission.yaml") :
			msg.info("study_yaml.py","submission.yaml file found in directory {0}... I will use this to steer the directory".format(in_str))
			HEPData_hlp.load_submission_file ( dataset , in_str , "submission.yaml" )
		else :
			msg.info("study_yaml.py","no submission.yaml file found in directory {0}... I will open all available yaml files".format(in_str))
			HEPData_hlp.load_all_yaml_files ( dataset , in_str )
	else :
		if HEPData_hlp.is_yaml_file(in_str) == False :
			msg.fatal("study_yaml.py","{0} doesn't seem to be a yaml file or a directory... I don't know what to do with it".format(in_str))
		if HEPData_hlp.is_submission_file(in_str) :
			msg.info("study_yaml.py","{0} is a submission.yaml file... I will use this to steer the directory".format(in_str))
			HEPData_hlp.load_submission_file ( dataset , in_str )
		else :
			msg.info("study_yaml.py","Interpreting {0} as a yaml file... I will use it as my only input".format(in_str))
			HEPData_hlp.load_yaml_file ( dataset , in_str )




'''
if __name__ == "__main__" :
	msg.info("study_yaml.py","Running program")
	if len(sys.argv) != 2 :
		msg.fatal("study_yaml.py","{0} argument(s) provided where 2 were expected".format(len(sys.argv)))
	in_str = sys.argv[1]
	in_str = hlp.remove_subleading(in_str,"/")
	yaml_filenames = []
	if os.path.isdir(in_str) :
		msg.info("study_yaml.py","{0} is a directory... I am expanding the entries (but will only go one directory deep!)".format(in_str))
		for f in os.listdir(in_str) :
			f = in_str + "/" + f
			f_extension = os.path.splitext(f)[1][1:]
			if hlp.check_extension(f_extension,"yaml") :
				msg.info("study_yaml.py","{0} is a yaml file... I have added it to my list of inputs".format(f))
				yaml_filenames.append(f)
			else :
				msg.info("study_yaml.py","{0} does not seem to be a yaml file... I will ignore it".format(f))
	else :
		if hlp.check_extension(in_str,"yaml") :
			msg.info("study_yaml.py","Interpreting {0} as a yaml file... I will use it as my only input".format(in_str))
			yaml_filenames.append(in_str)
		else :
			msg.fatal("study_yaml.py","Input {0} does not seem to be a yaml file nor a directory".format(in_str))
	for yaml_filename in yaml_filenames :
		msg.info("study_yaml.py","Opening {0} as yaml file".format(yaml_filename))
		with open(yaml_filename, 'r') as yaml_file:
		    try:
		        print(yaml.safe_load(yaml_file))
		    except yaml.YAMLError as exc:
		        print(exc)
'''