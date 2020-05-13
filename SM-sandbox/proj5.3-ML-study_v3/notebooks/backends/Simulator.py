#   Implementation of Simulator base class
#   Author:  Stephen Menary  (stmenary@cern.ch)


from .Param import Param



#  Base Class:  Generate random datapoints according to some generative model
#               - derived classes could use analytic or NN models
#
class Simulator :
    def __init__ (self) :
        self.clear()
    def add_param (self, name, value=0, lower_lim=None, upper_lim=None) :
        self.params.append(Param(name, value, lower_lim, upper_lim))
    def clear (self) :
        self.params    = []
    def get_param (self, name) :
        for param in self.params :
            if param.name != name : continue
            return param
        raise KeyError(f"Simulator object contains no param called '{name}'")
    def set_param_value (self, name, value=None) :
        self.get_param(name).value = value
    def generate (self, n_points=1, *argv, **kwargs) :
        if hasattr(self, "generator") is False :
            raise RuntimeError("Simulator.generate(...) requires Simulator.generator to have been configured")
        if type(self.generator) == type(None) :
            raise RuntimeError("Simulator.generate(...) requires Simulator.generator to have been configured")
        return self.generator(self, n_points, *argv, **kwargs)