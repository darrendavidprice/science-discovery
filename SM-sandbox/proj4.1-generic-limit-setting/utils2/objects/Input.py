##  ===========================================================================================================================
##  Brief :  Input class; wrapper for Distribution with labels describing metadata from the input file
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  ===========================================================================================================================


##  Stores a single Distribution object read from a file
#
class Input :
	def __init__ (self, name="", type="", origin_file="", keys=[], params=None, dist=None) :
		self.name        = name
		self.type        = type
		self.origin_file = origin_file
		self.keys        = keys
		self.dist        = dist
		self.params      = params
	def __str__ (self) :
		return f"Input \'{self.name}\' of type \'{self.type}\' with {len(self.dist)} entries"