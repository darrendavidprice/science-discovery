#   ==================================================================================
#   Brief : Classes and methods which help us create and train the density model
#   Author: Stephen Menary (sbmenary@gmail.com)
#   ==================================================================================


#  Required imports

import pickle, time

import numpy as np
import tensorflow as tf

from   keras.activations import softplus
from   keras.layers      import BatchNormalization, Dense, Dropout, Input, LeakyReLU, Concatenate, Lambda, Reshape, Softmax
from   keras.models      import Model
from   keras.optimizers  import Adam, SGD, Adadelta
from   keras.callbacks   import EarlyStopping
import keras.backend     as     K

from .utils import joint_shuffle, make_sure_dir_exists_for_filename


#  Configurable constants
#
Gauss_width_reduction_factor = 8.


#  Brief: tf function for adding offsets to the Gaussian means, so they start equally spaced (not all at 0)
# 
def add_gauss_mean_offsets (x, num_gauss, offset_min, offset_max):
    c = tf.convert_to_tensor([offset_min + (offset_max-offset_min)*i/(num_gauss-1.) for i in range(num_gauss)])
    return x + c


#  Brief: tf function for adding offsets to the Gaussian amplitudes, so they start equal
# 
def add_gauss_fraction_offsets (x, num_gauss):
    c = tf.convert_to_tensor([1./num_gauss for i in range(num_gauss)])
    return 0.8*x + 0.2*c


#  Brief: tf function for adding offsets to the Gaussian widths, so they are not allowed to collapse to 0
# 
def add_gauss_sigma_offsets (x, num_gauss):
    c = tf.convert_to_tensor([1e-4 for i in range(num_gauss)])
    return x + c


#  Brief: tf function for initialising the Gaussian widths, so we control the starting conditions
# 
def add_gauss_sigma_offsets2 (x, num_gauss, offset_min, offset_max):
    offset_range = float(offset_max - offset_min)
    target_width = offset_range / num_gauss / Gauss_width_reduction_factor
    offset       = float(np.log(np.exp(target_width) - 1))
    c = tf.convert_to_tensor([offset for i in range(num_gauss)])
    return x + c


#  Brief: create a continuous keras model describing the density over one observable
# 
def create_continuous_density_keras_model (name, **kwargs) :
    #  Parse arguments
    #
    num_conditions_in  = int  (kwargs.get("num_conditions_in"         ))
    num_observables_in = int  (kwargs.get("num_observables_in", 0     ))
    num_gaussians      = int  (kwargs.get("num_gaussians"     , 5     ))
    verbose            = bool (kwargs.get("verbose"           , True  ))
    learning_rate      = float(kwargs.get("learning_rate"     , 0.001 ))
    optimiser          = str  (kwargs.get("optimiser"         , "adam"))
    range_min          = float(kwargs.get("range_min"         , -5.   ))
    range_max          = float(kwargs.get("range_max"         , 5.    ))
    
    #  Print a status message
    #
    if verbose : 
        print(f"Creating continuous density model: {name}")
        print(f"  - num_conditions_in  is {num_conditions_in}")
        print(f"  - num_observables_in is {num_observables_in}")
        print(f"  - num_gaussians      is {num_gaussians}")
        print(f"  - range              is {range_min:.4f} - {range_max:.4f}")
    
    #  Create model
    #
    #  Format Gaussian means so they start equally placed, 
    #       sigmas so they are positive nonzero, and fractions so they sum to one
    #
    conditions_input  = Input((num_conditions_in ,))
    model_conditions  = Dense      (5 + 2*num_conditions_in)(conditions_input) 
    model_conditions  = LeakyReLU  (0.2                    )(model_conditions )
    if num_observables_in > 0 :
        observables_input = Input((num_observables_in,))
        model_observables = Dense      (3*num_observables_in)(observables_input)    
        model_observables = LeakyReLU  (0.2                 )(model_observables)
        model             = Concatenate()([model_conditions, model_observables])
    else :
        model = model_conditions
    '''model = Dense     (10*num_gaussians)(model)
                model = LeakyReLU (0.2             )(model)'''
        
    gauss_means     = Dense      (2*num_gaussians )(model      )
    gauss_means     = LeakyReLU  (0.2             )(gauss_means)
    gauss_means     = Dense      (num_gaussians, activation="linear"  )(gauss_means)
    add_initial_mean_offsets = lambda x : add_gauss_mean_offsets(x, num_gaussians, range_min, range_max)
    gauss_means     = Lambda(add_initial_mean_offsets)(gauss_means)
    
    gauss_sigmas       = Dense      (2*num_gaussians   )(model        )
    gauss_sigmas       = LeakyReLU  (0.2               )(gauss_sigmas )
    gauss_sigmas       = Dense      (num_gaussians     )(gauss_sigmas )
    add_sigma_offsets2 = lambda x : add_gauss_sigma_offsets2(x, num_gaussians, range_min, range_max)
    gauss_sigmas       = Lambda     (add_sigma_offsets2)(gauss_sigmas )

    lambda_softplus    = lambda x : K.log(1. + K.exp(x))
    gauss_sigmas       = Lambda     (lambda_softplus   )(gauss_sigmas )
    #gauss_sigmas       = softplus                       (gauss_sigmas )

    add_sigma_offsets  = lambda x : add_gauss_sigma_offsets(x, num_gaussians)
    gauss_sigmas       = Lambda     (add_sigma_offsets )(gauss_sigmas )
    
    gauss_fractions = Dense      (2*num_gaussians )(model           )
    gauss_fractions = LeakyReLU  (0.2             )(gauss_fractions )
    gauss_fractions = Dense      (num_gaussians, activation="softmax" )(gauss_fractions)
    add_fraction_offsets = lambda x : add_gauss_fraction_offsets(x, num_gaussians)
    gauss_fractions = Lambda(add_fraction_offsets)(gauss_fractions)
    
    #  Concatenate model output
    #
    model             = Concatenate()([gauss_fractions, gauss_means, gauss_sigmas])
    if num_observables_in > 0 : model = Model ([conditions_input, observables_input], model, name=name)
    else                      : model = Model (conditions_input, model, name=name)
    
    #  Compile model
    #
    loss_function = lambda y_true, y_pred : -1. * K_dataset_log_likelihood (y_true, y_pred, num_gaussians)
    if   optimiser.lower() == "sgd"      : model.compile(loss=loss_function, optimizer=SGD     (learning_rate=learning_rate))    
    elif optimiser.lower() == "adadelta" : model.compile(loss=loss_function, optimizer=Adadelta(learning_rate=learning_rate))    
    elif optimiser.lower() == "adam"     : model.compile(loss=loss_function, optimizer=Adam    (learning_rate=learning_rate))   
    else : raise ValueError(f"Optimiser '{optimiser}' not recognised") 
    if verbose : model.summary()
     
    #  Return model
    #   
    return model, (num_conditions_in, num_observables_in, num_gaussians)


#  Brief: create a discrete keras model describing the porbability distribution over one observable
# 
def create_discrete_density_keras_model (name, **kwargs) :
    #  Parse arguments
    #
    num_conditions_in  = int  (kwargs.get("num_conditions_in"         ))
    num_observables_in = int  (kwargs.get("num_observables_in", 0     ))
    num_outputs        = int  (kwargs.get("num_outputs"       , 5     ))
    verbose            = bool (kwargs.get("verbose"           , True  ))
    learning_rate      = float(kwargs.get("learning_rate"     , 0.001 ))
    optimiser          = str  (kwargs.get("optimiser"         , "adam"))
    
    #  Print a status message
    #
    if verbose : 
        print(f"Creating discrete density model: {name}")
        print(f"  - num_conditions_in  is {num_conditions_in}")
        print(f"  - num_observables_in is {num_observables_in}")
        print(f"  - num_outputs        is {num_outputs}")
    
    #  Create model layers
    #     output layer must sum to one
    #
    conditions_input  = Input((num_conditions_in ,))
    model_conditions  = Dense      (5 + 2*num_conditions_in)(conditions_input ) 
    model_conditions  = LeakyReLU  (0.2)(model_conditions )
    if num_observables_in > 0 :
        observables_input = Input((num_observables_in,))
        model_observables = Dense      (3*num_observables_in)(observables_input)    
        model_observables = LeakyReLU  (0.2                 )(model_observables)
        model             = Concatenate(   )([model_conditions, model_observables])
    else :
        model = model_conditions
    model = Dense      (3*num_outputs)(model)
    model = LeakyReLU  (0.2          )(model)
    model = Dense      (2*num_outputs)(model)
    model = LeakyReLU  (0.2          )(model)
    
    #normalise_to_unity = lambda x : x / K.sum(x)
    model = Dense (num_outputs, activation="softmax" )(model)
    #model = Lambda(normalise_to_unity                )(model)
    
    #  Compile model
    #
    if num_observables_in > 0 : model = Model ([conditions_input, observables_input], model, name=name)
    else                      : model = Model (conditions_input, model, name=name)
    #loss_function = lambda y_true, y_pred : -1. * K.sum(y_true*y_pred)
    if   optimiser.lower() == "sgd"      : model.compile(loss="categorical_crossentropy", optimizer=SGD     (learning_rate=learning_rate))    
    elif optimiser.lower() == "adadelta" : model.compile(loss="categorical_crossentropy", optimizer=Adadelta(learning_rate=learning_rate))    
    elif optimiser.lower() == "adam"     : model.compile(loss="categorical_crossentropy", optimizer=Adam    (learning_rate=learning_rate))   
    else : raise ValueError(f"Optimiser '{optimiser}' not recognised")   
    if verbose : model.summary()
     
    #  Return model
    #   
    return model, (num_conditions_in, num_observables_in, num_outputs)


#  Brief: return the density at position x, given Gaussian parameters params
# 
def get_sum_gauss_density (x, params) :
    num_gauss = int(len(params) / 3)
    fracs, means, sigmas = params[:num_gauss], params[num_gauss:2*num_gauss], params[2*num_gauss:3*num_gauss]
    return np_datapoint_likelihood (x, num_gauss, fracs, means, sigmas)


#  Brief: keras implementation of Gaussian probability density
# 
def K_gauss_prob (x, mean, sigma) :
    prob = K.exp(-0.5*(x - mean)*(x - mean)/(sigma*sigma)) / K.sqrt(2*np.pi*sigma*sigma)
    return tf.where(tf.is_nan(prob), 1e-20*tf.ones_like(prob), prob)


#  Brief: keras implemention returning the likelihood for individual datapoints
# 
def K_datapoint_likelihood (x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas) :
    prob = 0.
    x = x[:,0]
    for i in range(num_gauss) :
        prob = prob + gauss_fracs[:,i] * K_gauss_prob(x, gauss_means[:,i], gauss_sigmas[:,i])
    return prob


#  Brief: keras implemention returning the log likelihood for individual datapoints
# 
def K_datapoint_log_likelihood (x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas) :
    log_L = K.log(K_datapoint_likelihood(x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas))
    return tf.where(tf.is_nan(log_L), -1e20*tf.ones_like(log_L), log_L)


#  Brief: keras implemention returning the mean likelihood over a set of datapoints
# 
def K_dataset_mean_likelihood  (x, params, num_gauss) :
    return K.mean(K_datapoint_likelihood(x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas))  


#  Brief: keras implemention returning the total likelihood over a set of datapoints
# 
def K_dataset_likelihood (x, params, num_gauss) :
    prod_L = K.prod(K_datapoint_likelihood(x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas))
    return tf.where(tf.is_nan(prod_L), tf.zeros_like(prod_L), prod_L)
        

#  Brief: keras implemention returning the mean log likelihood over a set of datapoints
# 
def K_dataset_log_likelihood (x, params, num_gauss) :
    gauss_fracs, gauss_means, gauss_sigmas = params[:,:num_gauss], params[:,num_gauss:2*num_gauss], params[:,2*num_gauss:3*num_gauss]
    return K_datapoint_log_likelihood(x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas)


#  Brief: numpy implemention returning a single Gaussian  probability density
# 
def np_gauss_prob (x, mean, sigma) :
    return np.exp(-0.5*(x - mean)*(x - mean)/(sigma*sigma)) / np.sqrt(2*np.pi*sigma*sigma)


#  Brief: numpy implemention returning the likelihood for individual datapoints
# 
def np_datapoint_likelihood (x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas) :
    prob = 0.
    for i in range(num_gauss) :
        prob = prob + gauss_fracs[i] * np_gauss_prob(x, gauss_means[i], gauss_sigmas[i])
    return prob


#  Brief: create a one-hot vector representing a given index, assuming a given size
# 
def onehot (size, index) :
    ret = np.zeros(shape=(size,))
    ret [index] = 1
    return ret


#  Brief: randomly sample a Gaussian mixture model n_points times using the model parameters provided
# 
def sample_sum_gaussians (n_points, params) :
    num_gauss = int(len(params) / 3)
    fracs, means, sigmas = params[:num_gauss], params[num_gauss:2*num_gauss], params[2*num_gauss:3*num_gauss]
    which_gauss     = np.random.choice(num_gauss, size=(n_points,), p=fracs)
    n_pts_per_gauss = [len([x for x in which_gauss if x == i]) for i in range(num_gauss)]
    points = []
    for i in range(num_gauss) :
        points.append(np.random.normal(means[i], sigmas[i], n_pts_per_gauss[i]))
    return np.concatenate(points)


#  Class ContinuousDensityModel
#    -  wrapper for constructing and training a density model for a continuous observable
#    -  allow it to be evaluated, sampled, fit and rebuilt
#
class ContinuousDensityModel () :
    
    def __init__ (self, name, **kwargs) :
        self.rebuild(name, **kwargs)

    def evaluate (self, conditions, observables, conditional_observables=[]) :
        ds_size = len(observables)
        conditions = [conditions for i in range(ds_size)]
        gauss_params_list = self.get_gauss_params(conditions, conditional_observables)
        return np.array([get_sum_gauss_density (xp, gauss_params) for xp, gauss_params in zip(observables, gauss_params_list)])
    
    def fit (self, *argv, **kwargs) :
        self.model.fit(*argv, **kwargs)
    
    def get_gauss_params (self, conditional_params, conditional_observables=[]) :
        if self.num_observables == 0 :
            return self.model.predict(conditional_params)
        return self.model.predict([conditional_params, conditional_observables])
        
    def rebuild (self, name, **kwargs) :
        self.model, model_constants = create_continuous_density_keras_model (name, **kwargs)
        self.name            = name
        self.num_conditions  = model_constants[0]
        self.num_observables = model_constants[1]
        self.num_gaussians   = model_constants[2]
        
    def sample (self, n_points, conditional_params, conditional_observables=[]) :
        gauss_params_list = self.get_gauss_params(conditional_params, conditional_observables)
        return np.array([sample_sum_gaussians (n_points, gauss_params) for gauss_params in gauss_params_list])


#  Class DiscreteDensityModel
#    -  wrapper for constructing and training a density model for a discrete observable
#    -  allow it to be evaluated, sampled, fit and rebuilt
#
class DiscreteDensityModel :
    
    def __init__ (self, name, minimum, maximum, **kwargs) :
        self.rebuild(name, minimum, maximum, **kwargs)
        
    def evaluate (self, x, conditional_params, conditional_observables=[]) :
        probs_list = self.get_categorical_probabilities(conditional_params, conditional_observables)
        x = self._x_to_idices(x)
        return np.array([probs[xp] for xp, probs in zip(x, probs_list)])
    
    def fit (self, X, Y, *argv, **kwargs) :
        Y = self._x_to_onehot(Y)
        self.model.fit(X, Y, *argv, **kwargs)
    
    def get_categorical_probabilities (self, conditional_params, conditional_observables=[]) :
        if self.num_observables == 0 :
            return self.model.predict(conditional_params)
        return self.model.predict([conditional_params, conditional_observables])
        
    def rebuild (self, name, minimum, maximum, **kwargs) :
        self.name            = name
        self.minimum         = int(minimum)
        self.maximum         = int(maximum)
        self.model, model_constants = create_discrete_density_keras_model (name, num_outputs=1+maximum-minimum, **kwargs)
        self.num_conditions  = model_constants[0]
        self.num_observables = model_constants[1]
        
    def sample (self, n_points, conditional_params, conditional_observables=[]) :
        probs_list = self.get_categorical_probabilities(conditional_params, conditional_observables)
        return np.array([np.random.choice(np.arange(self.minimum, self.maximum+1), n_points, p=probs/np.sum(probs)) for probs in probs_list])
    
    def _x_to_indices (self, x) :
        indices = []            
        if type(x) not in [list, np.ndarray] : x = [x]
        for val in x :
            if val > self.maximum : raise ValueError(f"Value {x} exceeds self.maximum ({self.maximum})")
            if val < self.minimum : raise ValueError(f"Value {x} is less than self.minimum ({self.minimum})")
            indices.append(int(val - self.minimum))
        return indices
    
    def _x_to_onehot (self, x) :
        indices    = self._x_to_indices(x)
        row_length = 1 + self.maximum - self.minimum
        rows       = [onehot(row_length, idx) for idx in indices]
        return np.array(rows)
    

#  Class DiscreteDensityModel
#    -  store, construct and train an autoregressive density model over many observables
#    -  allow it to be evaluated, sampled, fit and rebuilt
#    -  allow it to be saved to / loaded from pickle files
#
class DensityModel :
    def __init__ (self, **kwargs) :
        self.construct(**kwargs)
    def build (self, build_settings=None, verbose=True) :
        if type(build_settings) == type(None) :
            build_settings    = {"name":self.name, 
                                 "num_gaussians":self.num_gaussians, 
                                 "num_conditions":self.num_conditions, 
                                 "num_observables":self.num_observables, 
                                 "types":self.types, 
                                 "int_limits":self.int_limits, 
                                 "range_limits":self.range_limits}
        likelihood_models = []
        for i in range(build_settings["num_observables"]) :
            model_segment_name = build_settings["name"]+f"_observable{i}"
            if verbose : print("INFO".ljust(8) + "   " + "DensityModel.build".ljust(25) + "   " + f"Building model segment: {model_segment_name} for observable index {i}")
            if self.types[i] == float :
                range_min, range_max = build_settings.get("range_limits", {}).get(i, [-5., 5.])
                density_model = ContinuousDensityModel      (model_segment_name,
                                                             num_gaussians      = build_settings["num_gaussians" ] ,
                                                             num_conditions_in  = build_settings["num_conditions"] ,
                                                             num_observables_in = i                                ,
                                                             verbose            = verbose                          ,
                                                             range_min          = range_min                        ,
                                                             range_max          = range_max                        )
            elif self.types[i] == int :
                density_model = DiscreteDensityModel        (model_segment_name,
                                                             minimum            = build_settings["int_limits"][i][0],
                                                             maximum            = build_settings["int_limits"][i][1],
                                                             num_conditions_in  = build_settings["num_conditions"]  ,
                                                             num_observables_in = i                                 ,
                                                             verbose            = verbose                           )
            else :
                raise TypeError(f"Observable index {i} requested an unrecognised type {self.types[i]}")
            likelihood_models.append(density_model)
        if verbose : print("INFO".ljust(8) + "   " + "DensityModel.build".ljust(25) + "   " + f"{len(likelihood_models)} partial density models constructed")
        self.build_settings    = build_settings
        self.likelihood_models = likelihood_models
    def construct (self, **kwargs) :
        name            = kwargs.get("name"           , None )
        num_gaussians   = kwargs.get("num_gaussians"  , None )
        num_conditions  = kwargs.get("num_conditions" , None )
        num_observables = kwargs.get("num_observables", None )
        types           = kwargs.get("types"          , None )
        int_limits      = kwargs.get("int_limits"     , {}   )
        range_limits    = kwargs.get("range_limits"   , {}   )   
        verbose         = kwargs.get("verbose"        , True ) 
        do_build        = kwargs.get("build"          , True )     
        if (type(name)            == type(None)) and (hasattr(self, "name"           )) :
            if verbose      : print(f"No name argument provided - using stored value")
            name            = self.name
        if (type(num_gaussians)   == type(None)) and (hasattr(self, "num_gaussians"  )) :
            if verbose      : print(f"No num_gaussians argument provided - using stored value")
            num_gaussians   = self.num_gaussians
        if (type(num_conditions)  == type(None)) and (hasattr(self, "num_conditions" )) :
            if verbose      : print(f"No num_conditions argument provided - using stored value")
            num_conditions  = self.num_conditions
        if (type(num_observables) == type(None)) and (hasattr(self, "num_observables")) :
            if verbose      : print(f"No num_observables argument provided - using stored value")
            num_observables = self.num_observables
        if (type(types) == type(None)) and (hasattr(self, "types")) :
            if verbose      : print(f"No types argument provided - using stored value")
            types = self.types
        if type(types) == type(None) :
            if verbose      : print(f"No types argument provided - assuming all are floats")
            types = [float for i in range(num_observables)]
        if type(name)            != str : raise TypeError(f"name argument {name} of type {type(name)} where {type(int)} expected")
        if type(num_gaussians)   != int : raise TypeError(f"num_gaussians argument {num_gaussians} of type {type(num_gaussians)} where {type(int)} expected")
        if type(num_conditions)  != int : raise TypeError(f"num_conditions argument {num_conditions} of type {type(num_conditions)} where {type(int)} expected")
        if type(num_observables) != int : raise TypeError(f"num_observables argument {num_observables} of type {type(num_observables)} where {type(int)} expected")
        if len(types) != num_observables : raise TypeError(f"length of types argument {types} ({len(types)}) != num_observables ({num_observables})")
        if num_gaussians   < 1 : raise ValueError(f"num_gaussians must be > 0, but {num_gaussians} provided")
        if num_conditions  < 1 : raise ValueError(f"num_conditions must be > 0, but {num_conditions} provided")
        if num_observables < 1 : raise ValueError(f"num_observables must be > 0, but {num_observables} provided")
        print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set model name           : {name}"           )
        print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set model num_gaussians  : {num_gaussians}"  )
        print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set model num_conditions : {num_conditions}" )
        print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set model num_observables: {num_observables}")
        print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set observable types     : {types}")
        self.name            = name
        self.num_gaussians   = num_gaussians
        self.num_conditions  = num_conditions
        self.num_observables = num_observables
        self.int_limits      = int_limits
        self.range_limits    = range_limits
        self.types           = types
        if do_build is False : return
        self.build(verbose=verbose)
    def evaluate (self, conditions, observables) :
        num_observables = observables.shape[1]
        if len(observables.shape) == 1 :
            observables = observables.copy()
            observables.reshape(len(observables), 1)
        density = self.likelihood_models[0].evaluate(conditions, observables[:,0])
        for obs_idx in range(1, num_observables) :
            density = density * self.likelihood_models[obs_idx].evaluate(conditions, observables[:,obs_idx], observables[:,:obs_idx])
        return density
    def fit (self, dataset, weights=None, **kwargs) :                                     
        #  Parse settings
        #
        observable_idx            = kwargs.get("observable"               , None )
        max_epochs_per_observable = kwargs.get("max_epochs_per_observable", 2000 )
        early_stopping_patience   = kwargs.get("early_stopping_patience"  , 100  )
        early_stopping_min_delta  = kwargs.get("early_stopping_min_delta" , 0    )
        batch_size_per_observable = kwargs.get("batch_size_per_observable", -1   )
        validation_split          = kwargs.get("validation_split"         , 0.3  )
        do_build                  = kwargs.get("build"                    , False)
        verbose                   = kwargs.get("verbose"                  , True )
        tf_verbose                = kwargs.get("tf_verbose"               , 1    )
        monitor = "val_loss"
        if validation_split <= 0 : monitor = "loss"
        
        early_stopping_min_delta
                                                        
        #  (Re-)build model if requested
        #
        if do_build : self.build(**kwargs)
                                                           
        #  Make sure model has been built, and with the same settings as currently set
        #                                     
        if hasattr(self, "build_settings") is False : raise RuntimeError(f"self.build_settings does not exist - you must call self.build() before self.fit(), or specify self.fit(build=True)")                                     
        build_settings = {"name":self.name, "num_gaussians":self.num_gaussians, "num_conditions":self.num_conditions, "num_observables":self.num_observables}
        for setting, value in build_settings.items() :
            built_value = self.build_settings [setting]
            if built_value == value : continue
            raise ValueError(f"Setting {setting}={value} has changed since last build (with {setting}={built_value}. You must specify self.fit(build=True) if you want to re-build the model.")
                                                                
        #  Collect list of observable indices to be trained
        #                                       
        if   type(observable_idx) == type(None) : observable_indices = [i for i in range(self.num_observables)]
        elif type(observable_idx) == int        : observable_indices = [observable_idx]
        elif type(observable_idx) == list       : observable_indices = observable_idx
        elif type(observable_idx) == np.ndarray : observable_indices = [i for i in observable_idx]
        else : raise TypeError(f"Observable {observable_idx} of type {type(observable_idx)} could not be interpreted as an integer, or list of integers")
        for index in observable_indices :
            if (type(index) == int) and (index >= 0) and (index < self.num_observables) : continue
            raise TypeError(f"Observable indices must be integers but {type(index)}={index} provided")  
        if verbose : print("INFO".ljust(8) + "   " + "DensityModel.fit".ljust(25) + "   " + f"Queued the following observable indices to train: {', '.join([str(x) for x in observable_indices])}")                                    
                                                                  
        #  Create weights if None provided, and make sure all samples normalised equally
        # 
        if type(weights) == type(None) :
            if verbose : print("INFO".ljust(8) + "   " + "DensityModel.fit".ljust(25) + "   " + "Creating weights, as None provided")
            weights = {}
            for condition, datapoints in dataset.items() :
                weights [condition] = np.full(fill_value=1./len(datapoints), shape=(len(datapoints),))
        #if verbose : print("INFO".ljust(8) + "   " + "DensityModel.fit".ljust(25) + "   " + "Normalising samples to equal weight")
        '''for condition, weights_vec in weights.items() :
                                    weights_vec = weights_vec / np.sum(weights_vec)'''

        #  Parse dataset to be fit
        #             
        if verbose : print("INFO".ljust(8) + "   " + "DensityModel.fit".ljust(25) + "   " + "Parsing training data")
        train_data_cond, train_data_obs, train_data_weights = [], [], []
        for condition, datapoints in dataset.items() :
            train_data_weights.append(weights [condition])
            for datapoint in datapoints :
                train_data_cond.append(condition)
                train_data_obs .append(datapoint)
        train_data_weights = np.concatenate(train_data_weights)
        train_data_cond, train_data_obs, train_data_weights = joint_shuffle(train_data_cond, train_data_obs, train_data_weights)
        train_data_cond, train_data_obs = np.array(train_data_cond), np.array(train_data_obs)
                                                     
        #  Loop over target observables
        #  
        for observable_idx in observable_indices :
            if verbose :
                print("INFO".ljust(8) + "   " + "DensityModel.fit".ljust(25) + "   " + f"Training observable index {observable_idx}")
                print("INFO".ljust(8) + "   " + "DensityModel.fit".ljust(25) + "   " + f"  -  Training setting: epochs = {max_epochs_per_observable}")
                print("INFO".ljust(8) + "   " + "DensityModel.fit".ljust(25) + "   " + f"  -  Training setting: batch_size = {batch_size_per_observable}")
                print("INFO".ljust(8) + "   " + "DensityModel.fit".ljust(25) + "   " + f"  -  Training setting: validation_split = {validation_split}")
                print("INFO".ljust(8) + "   " + "DensityModel.fit".ljust(25) + "   " + f"  -  Training setting: early_stopping_patience = {early_stopping_patience}")     
                print("INFO".ljust(8) + "   " + "DensityModel.fit".ljust(25) + "   " + f"  -  Training setting: early_stopping_min_delta = {early_stopping_min_delta}")                                           
            #  Project training data along appropriate axes
            #  
            if observable_idx == 0 :
                train_data_X = train_data_cond
            else :
                train_data_X_obs = train_data_obs[:, :observable_idx]
                train_data_X = [train_data_cond, train_data_X_obs]
            n_data = len(train_data_cond)
            if (batch_size_per_observable > n_data) or (batch_size_per_observable <= 0) :
                print("INFO".ljust(8) + "   " + "DensityModel.fit".ljust(25) + "   " + f"Setting batch size to size of dataset, which is {n_data}")
                batch_size = n_data
            else : batch_size = batch_size_per_observable
            train_data_Y = train_data_obs[:,observable_idx]
            start_time = time.time()
            #print(" ".join([f"{c:.1f}" for c in train_data_cond[:1000]]))   # check that data is shuffled properly
            self.likelihood_models[observable_idx].fit(train_data_X,
                                                       train_data_Y,
                                                       sample_weight    = train_data_weights,
                                                       validation_split = validation_split,
                                                       epochs           = max_epochs_per_observable,
                                                       shuffle          = True,
                                                       batch_size       = batch_size,
                                                       callbacks        = [EarlyStopping(patience=early_stopping_patience, restore_best_weights=True, monitor=monitor, min_delta=early_stopping_min_delta)])
            print(f"Fit completed in {int(time.time() - start_time):.0f}s")
    @classmethod
    def from_dir (cls, dirname, verbose=False) :
        ret = cls(name="dflt", num_gaussians=2, num_observables=1, num_conditions=1, do_build=False, verbose=verbose)
        ret.load_from_dir(dirname)
        return ret

    def get_NLL (self, params, dataset, weights=None) :
        num_observables = dataset.shape[1]
        params          = np.full(fill_value=params, shape=dataset[:,0].shape)
        NLL             = self.likelihood_models[0].model.evaluate(params, dataset[:,0], sample_weight=weights, verbose=0)
        for obs_idx in range(1, num_observables) :
            NLL = NLL + self.likelihood_models[obs_idx].model.evaluate([params, dataset[:,:obs_idx]], dataset[:,0], sample_weight=weights, verbose=0)
        return NLL

    def load_from_dir (self, dirname) :
        pfile_name = dirname + "/density_model.pickle"
        to_load = pickle.load(open(pfile_name, "rb"))
        for required_key in ["build_settings", "model_files", "name", "num_gaussians", "num_conditions", "num_observables", "types", "int_limits"] :
            if required_key in to_load : continue
            raise RuntimeError(f"Required entry '{required_key}' not found in file '{pfile_name}'")
        build_settings    = to_load ["build_settings"   ]
        self.name              = to_load ["name"             ]
        self.num_gaussians     = to_load ["num_gaussians"    ]
        self.num_conditions    = to_load ["num_conditions"   ]
        self.num_observables   = to_load ["num_observables"  ]
        self.types             = to_load ["types"            ]
        self.int_limits        = to_load ["int_limits"       ]
        self.range_limits      = to_load.get ("range_limits", [-5, 5])
        self.build (build_settings=build_settings, verbose=False)
        for idx, (likelihood_model, model_fname) in enumerate(zip(self.likelihood_models, to_load["model_files"])) :
            likelihood_model.model.load_weights(model_fname)
    def sample (self, n_points, conditions, num_observables=None, verbose=True) :
        if type(num_observables) == type(None) :
            num_observables = self.num_observables
        if verbose : print("INFO".ljust(8) + "   " + "DensityModel.sample".ljust(25) + "   " + f"Sampling {n_points} datapoints, observable index is 0")  
        X1_to_XN = [self.likelihood_models[0].sample (n_points, [conditions]).reshape(n_points)]
        dp_conditions = [conditions for i in range(n_points)]
        for obs_idx in range(1, num_observables) :
            if verbose : print("INFO".ljust(8) + "   " + "DensityModel.sample".ljust(25) + "   " + f"Sampling {n_points} datapoints, observable index is {obs_idx}")  
            XN = self.likelihood_models[obs_idx].sample(1, dp_conditions, np.array(X1_to_XN).transpose()).reshape(n_points)
            X1_to_XN.append(XN)
        return np.array(X1_to_XN).transpose()
    def save_to_dir (self, dirname) :
        pfile_name = dirname + "/density_model.pickle"
        make_sure_dir_exists_for_filename(pfile_name)
        to_pickle = {}
        to_pickle ["build_settings"   ] = self.build_settings
        model_files = []
        for idx, likelihood_model in enumerate(self.likelihood_models) :
            model_fname = dirname + f"/tf_model_weights_obs{idx}"
            likelihood_model.model.save_weights(model_fname)
            model_files.append(model_fname)
        to_pickle ["model_files"      ] = model_files
        to_pickle ["name"             ] = self.name
        to_pickle ["num_gaussians"    ] = self.num_gaussians
        to_pickle ["num_conditions"   ] = self.num_conditions
        to_pickle ["num_observables"  ] = self.num_observables
        to_pickle ["types"            ] = self.types
        to_pickle ["int_limits"       ] = self.int_limits
        pickle.dump(to_pickle, open(pfile_name, "wb"))




