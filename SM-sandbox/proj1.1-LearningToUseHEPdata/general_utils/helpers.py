import os


def get_extension ( f ) :
	os.path.splitext(f)[1][1:]
	return f

def get_directory ( f ) :
	return os.path.dirname(f)

def check_extension ( f , ext ) :
	f_ext = get_extension(os.path.splitext(f)[1][1:])
	if f_ext == ext :
		return True
	return False

def remove_subleading ( f , pattern ) :
	while f[len(f)-len(pattern)] == pattern :
		f = f[:len(f)-len(pattern)]
	return f