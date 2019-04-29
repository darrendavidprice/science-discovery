# ====================================================================================================
#  Brief: some useful messaging functions with configurable global settings
#  Author: Stephen Menary (stmenary@cern.ch)
# ====================================================================================================

import sys


VERBOSE_LEVEL = -1
#  VERBOSE_LEVEL : Set the level of verbosity { -1 , 0 , 1 , 2 }
#      <= -1 provides only the minimum necessary updates to the terminal
#      0 provides reasonably granular updates (e.g. for keeping track of progress)
#      1 provides some debug-level information
#      >= 2 provides the maximum information possible (e.g. for debugging, be prepared to shift through the logs!)


ENABLE_FATAL = True
#  ENABLE_FATAL :
#      True means that throwing a fatal will cause the program to close
#      False means that throwing a fatal will allow the program to continue in a possibly-broken state, e.g. during interactive sessions


#  Brief: print message only if the global verbosity setting dictates
def check_verbosity_and_print ( message_ , **kwargs ) :
	_verbose_level = kwargs.get("verbose_level",-1)
	if _verbose_level <= -1 or VERBOSE_LEVEL >= _verbose_level :
		print ( message_ )


#  Brief: print message only if the global verbosity setting dictates
def print_message ( type_ , whereFrom_ , message_ , **kwargs ) :
	check_verbosity_and_print ( "{0}\t{1}\t{2}".format(type_,whereFrom_,message_) , **kwargs )


#  Brief: print error message only if the global verbosity setting dictates
def error ( whereFrom_ , message_ , **kwargs ) :
	print_message ( "ERROR" , whereFrom_ , message_ , **kwargs )


#  Brief: print fatal message only if the global verbosity setting dictates, and exit if the global setting ENABLE_FATAL 
def fatal ( whereFrom_ , message_ , **kwargs ) :
	print_message ( "FATAL" , whereFrom_ , message_ , **kwargs )
	if ENABLE_FATAL :
		sys.exit(kwargs.get("exit_code",0))


#  Brief: print info message only if the global verbosity setting dictates
def info ( whereFrom_ , message_ , **kwargs ) :
	print_message ( "INFO" , whereFrom_ , message_ , **kwargs )


#  Brief: print warning message only if the global verbosity setting dictates
def warning ( whereFrom_ , message_ , **kwargs ) :
	print_message ( "WARNING" , whereFrom_ , message_ , **kwargs )


#  Brief: forbid fatals
def forbid_fatal_errors () :
	ENABLE_FATAL = False


#  Brief: allow fatals
def allow_fatal_errors () :
	ENABLE_FATAL = True
