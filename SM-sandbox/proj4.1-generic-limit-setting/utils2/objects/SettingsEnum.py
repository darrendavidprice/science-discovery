##  =======================================================================================================================
##  Brief :  SettingEnum class; wrapper for python enum allowing for extra methods
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  =======================================================================================================================

from enum import Enum, unique


##  Base class for enum describing a settings configuration
##  (protects against non-defined values)
#
@unique
class SettingsEnum (Enum) :
	## String
	# 
	def __str__ (self) :
		return f"{self.name}"
	## Class method return list of allowed enums
	# 
	@classmethod
	def allowed_values (cls) :
		return [str(x) for x in cls.__members__]