
VERBOSE_LEVEL = -1
#  VERBOSE_LEVEL : Set the level of verbosity { -1 , 0 , 1 }
#      <= -1 provides only the minimum necessary updates to the terminal
#      0 provides reasonably granular updates (e.g. for keeping track of progress)
#      >= 1 provides the maximum information possible (e.g. for debugging, be prepared to shift through the logs!)


def check_verbosity_and_print ( message_ , **kwargs ) :
	_verbose_level = kwargs.get ( "_verbose_level" , -2 )
	if _verbose_level < -1 or VERBOSE_LEVEL >= _verbose_level :
		print ( message_ )

def print_message ( type_ , whereFrom_ , message_ , **kwargs ) :
	check_verbosity_and_print ( "{0}\t{1}\t{2}".format(type_,whereFrom_,message_) , **kwargs )

def error ( whereFrom_ , message_ , **kwargs ) :
	print_message ( "ERROR" , whereFrom_ , message_ , **kwargs )

def fatal ( whereFrom_ , message_ , code_=0 ) :
	print_message ( "FATAL" , whereFrom_ , message_ )
	exit(code_)

def info ( whereFrom_ , message_ , **kwargs ) :
	print_message ( "INFO" , whereFrom_ , message_ , **kwargs )

def warning ( whereFrom_ , message_ , **kwargs ) :
	print_message ( "WARNING" , whereFrom_ , message_ , **kwargs )