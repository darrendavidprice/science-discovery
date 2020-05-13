#   Implementation of Param structure
#   Author:  Stephen Menary  (stmenary@cern.ch)


#  Class:  Store the value of a parameter
#          - here this is used to represent model parameters
#
class Param :
    def __init__ (self, name=None, value=None, lower_lim=None, upper_lim=None) :
        self.clear()
        if type(name     ) != type(None) : self.name      = name
        if type(value    ) != type(None) : self.value     = value
        if type(lower_lim) != type(None) : self.lower_lim = lower_lim
        if type(upper_lim) != type(None) : self.upper_lim = upper_lim
    def clear (self) :
        self.name       = ""
        self.value      = 0.
        self.upper_lim  = None
        self.lower_lim  = None