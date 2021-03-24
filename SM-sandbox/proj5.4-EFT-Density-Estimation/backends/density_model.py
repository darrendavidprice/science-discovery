#   ==================================================================================
#   Brief : Classes and methods which help us create and train the density model
#   Author: Stephen Menary (sbmenary@gmail.com)
#   ==================================================================================


#  Required imports

import pickle, time
from   multiprocessing import Process, Queue
from   threading       import Thread

import numpy as np
import tensorflow as tf

from   keras.activations import softplus
from   keras.layers      import BatchNormalization, Dense, Dropout, Input, LeakyReLU, Concatenate, Lambda, Reshape, Softmax
from   keras.models      import Model
from   keras.optimizers  import Adam, SGD, Adadelta
from   keras.callbacks   import Callback, EarlyStopping
import keras.backend     as     K

from .utils import joint_shuffle, make_sure_dir_exists_for_filename


#  Disbale eager execution so first training epoch does not take forever in tf2 (?!)
#
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


#  Configurable constants
#
Gauss_width_reduction_factor = 8.


#  Non-configurable constants
#
sqrt_2pi = np.sqrt(2*np.pi)


#  Brief: tf function for adding offsets to the Gaussian widths, so they are not allowed to collapse to 0
# 
def add_epsilon_to_gauss_sigmas (x, num_gauss, epsilon=1e-4) :
    """TF method: for input x of size [?, num_gauss], add epsilon to every value"""
    c = tf.convert_to_tensor([float(epsilon) for i in range(num_gauss)])
    return x + c


#  Brief: tf function for adding offsets to the Gaussian means, so they start equally spaced (not all at 0)
# 
def add_gauss_mean_offsets (x, num_gauss, offset_min, offset_max) :
    """TF method: for input x of size [?, num_gauss], add evenly spaced offsets between [offset_min, offset_max]"""
    c = tf.convert_to_tensor([offset_min + (offset_max-offset_min)*i/(num_gauss-1.) for i in range(num_gauss)])
    return x + c


#  Brief: tf function for adding offsets to the Gaussian amplitudes, so they do not approach 0
# 
def add_gauss_fraction_offsets (x, num_gauss, const_frac=0.2) :
    """TF method: for input x of size [?, num_gauss], where x is a multinomial(num_gauss) probability distribution, add a constant term to prevent probabilities going to 0"""
    c = tf.convert_to_tensor([1./num_gauss for i in range(num_gauss)])
    return (1.-const_frac)*x + const_frac*c


#  Brief: create a continuous keras model describing the density over one observable
# 
def create_continuous_density_keras_model (name, **kwargs) :
    """create a Keras model which outputs the parameters of a Gaussian mixture model, conditioned on external parameters and other observables"""
    #
    #  =============================
    #  ===    Parse arguments    ===
    #  =============================
    #
    num_conditions_in        = int  (kwargs.get("num_conditions_in"                   ))
    num_observables_in       = int  (kwargs.get("num_observables_in", 0               ))
    num_gaussians            = int  (kwargs.get("num_gaussians"     , 5               ))
    verbose                  = bool (kwargs.get("verbose"           , True            ))
    learning_rate            = float(kwargs.get("learning_rate"     , 0.001           ))
    optimiser                = str  (kwargs.get("optimiser"         , "adam"          ))
    activation               = str  (kwargs.get("activation"        , "leakyrelu"     ))
    range_min                = float(kwargs.get("range_min"         , -5.             ))
    range_max                = float(kwargs.get("range_max"         , 5.              ))
    transform_min            = float(kwargs.get("transform_min"     , -2.             ))
    transform_max            = float(kwargs.get("transform_max"     , 2.              ))
    min_gauss_amplitude_frac = float(kwargs.get("min_gauss_amplitude_frac", 0.        ))
    bias_initializer         = kwargs.get("bias_initializer"        , "zeros"         )
    condition_limits         = kwargs.get("condition_limits"        , None            )
    observables_limits       = kwargs.get("observables_limits"      , None            )
    #
    #  Parse arguments to configure number of layers
    #
    A1, A2 = int(kwargs.get("A1", 5)), int(kwargs.get("A2", 5))
    B1, B2 = int(kwargs.get("B1", 5)), int(kwargs.get("B2", 5))
    C      = int(kwargs.get("C" , 1))
    D2     = int(kwargs.get("D2", 2))
    #
    #  Parse arguments to configure what constants to scale the output layers by (smaller numbers = smaller initial perturbations on Gaussian means, widths and fractions)
    #
    gauss_frac_scale  = float(kwargs.get("gauss_frac_scale" , 1. / 8.))
    gauss_mean_scale  = float(kwargs.get("gauss_mean_scale" , 1. / 8.))
    gauss_sigma_scale = float(kwargs.get("gauss_sigma_scale", 1. / 8.))
    #
    #  If condition_limits provided, we want to transform the condition inputs, so we should make sure the input has the correct format
    #
    do_transform_conditions = False
    if type(condition_limits) != type(None) :
        do_transform_conditions = True
        transform_range = transform_max - transform_min
        assert len(condition_limits) == num_conditions_in, f"{condition_limits} expected to have shape [{num_conditions_in}, 2]"
        assert transform_range > 0, f"Transform range transform_max - transform_min = {transform_max} - {transform_min} = {transform_range} must be > 0"
        for row_idx, row in enumerate(condition_limits) :
            assert len(row) == 2, f"condition_limits row index {row_idx} is {row} where object of length 2 expected"
            assert row[1] >= row[0], f"condition_limits row index {row_idx} is {row} expected upper limit >= lower limit"
            if verbose :
                print(f"Projecting external parameter index {row_idx} from interval {row} onto [{transform_min}, {transform_max}]")
        def transform_conditions (x) :    # x shape is (None, num_conditions_in)
            cond_ranges = [(row[1]-row[0]) for row in condition_limits]
            cond_ranges = np.array([r if r > 0 else 1 for r in cond_ranges])
            out_min     = tf.constant([transform_min if row[1]>row[0] else 0.5*(transform_min+transform_max) for row in condition_limits])
            out_scale   = tf.constant([transform_range/row_range for row_range in cond_ranges])
            in_min      = tf.constant([row[0] for row in condition_limits])
            return out_min + ((x - in_min)*out_scale)
    #
    #  If observables_limits provided, we want to transform the observables inputs, so we should make sure the input has the correct format
    #
    do_transform_observables = False
    if type(observables_limits) != type(None) :
        do_transform_observables = True
        transform_range = transform_max - transform_min
        assert len(observables_limits) == num_observables_in, f"{observables_limits} expected to have shape [{num_observables_in}, 2]"
        assert transform_range > 0, f"Transform range transform_max - transform_min = {transform_max} - {transform_min} = {transform_range} must be > 0"
        for row_idx, row in enumerate(observables_limits) :
            assert len(row) == 2, f"observables_limits row index {row_idx} is {row} where object of length 2 expected"
            assert row[1] >= row[0], f"observables_limits row index {row_idx} is {row} expected upper limit >= lower limit"
            if verbose :
                print(f"Projecting observable index {row_idx} from interval {row} onto [{transform_min}, {transform_max}]")
        def transform_observables (x) :    # x shape is (None, num_conditions_in)
            cond_ranges = [(row[1]-row[0]) for row in observables_limits]
            cond_ranges = np.array([r if r > 0 else 1 for r in cond_ranges])
            out_min     = tf.constant([transform_min if row[1]>row[0] else 0.5*(transform_min+transform_max) for row in observables_limits])
            out_scale   = tf.constant([transform_range/row_range for row_range in cond_ranges])
            in_min      = tf.constant([row[0] for row in observables_limits])
            return out_min + ((x - in_min)*out_scale)
    #
    #  Print the configured settings
    #
    if verbose : 
        print(f"Creating continuous density model: {name}")
        print(f"  - num_conditions_in        is {num_conditions_in}")
        print(f"  - num_observables_in       is {num_observables_in}")
        print(f"  - num_gaussians            is {num_gaussians}")
        print(f"  - learning_rate            is {learning_rate}")
        print(f"  - optimiser                is {optimiser}")
        print(f"  - activation               is {activation}")
        print(f"  - range                    is {range_min:.4f}  to  {range_max:.4f}")
        print(f"  - transform range          is {transform_min:.4f}  to  {transform_max:.4f}")
        print(f"  - min_gauss_amplitude_frac is {min_gauss_amplitude_frac}")
        print(f"  - bias_initializer         is {bias_initializer}")
        print(f"  - gauss_frac_scale         is {gauss_frac_scale}")
        print(f"  - gauss_mean_scale         is {gauss_mean_scale}")
        print(f"  - gauss_sigma_scale        is {gauss_sigma_scale}")
        print(f"  - adding hidden layer of size {A1 + A2*num_conditions_in} to pre-process condition inputs")
        if num_observables_in > 0 :
            print(f"  - adding hidden layer of size {B1 + B2*num_observables_in} to pre-process observables inputs")
        for c in range(C) :
            print(f"  - adding hidden layer of size {A1 + A2*num_conditions_in + B1 + B2*num_observables_in}")
    #
    #  If LeakyReLU used as activation function, set activation of Dense layers to "linear", then LeakyReLU will be applied as a separate layer
    #
    use_leaky_relu = False
    if activation.lower() == "leakyrelu" :
        activation     = "linear"
        use_leaky_relu = True
    #
    #  ================================
    #  ===    Create Keras model    ===
    #  ================================
    #
    #  Create an input layer for the external parameter dependence
    #  -  if configured, add a layer which transforms these inputs onto the given domain
    #  -  add a layer to process just these inputs
    #
    conditions_input  = Input((num_conditions_in ,))
    model_conditions  = conditions_input
    if do_transform_conditions : 
        model_conditions = Lambda(transform_conditions)(model_conditions)
    model_conditions  = Dense (A1 + A2*num_conditions_in, kernel_initializer=custom_weight_init_initial, bias_initializer=bias_initializer, activation=activation)(model_conditions) 
    if use_leaky_relu : 
        model_conditions  = LeakyReLU (0.2)(model_conditions )
    #
    #  If they exist, create an input layer for other input observables
    #  -  if configured, add a layer which transforms these inputs onto the given domain
    #  -  add a layer to process just these inputs
    #  -  concatenate the resulting hidden layer with that from the external parameter dependence
    #  If they don't exist, skip this step
    #
    if num_observables_in > 0 :
        observables_input = Input((num_observables_in,))
        model_observables = observables_input
        if do_transform_observables : 
            model_observables = Lambda(transform_observables)(model_observables)
        model_observables = Dense      (B1 + B2*num_observables_in, kernel_initializer=custom_weight_init_initial, bias_initializer=bias_initializer, activation=activation)(model_observables)    
        if use_leaky_relu : model_observables = LeakyReLU (0.2)(model_observables)
        model             = Concatenate()([model_conditions, model_observables])
    else :
        model = model_conditions
    #
    #  Add the configured number of additional hidden layers
    #
    for c in range(C) :
        model = Dense (A1 + A2*num_conditions_in + B1 + B2*num_observables_in, kernel_initializer=custom_weight_init_hidden, bias_initializer=bias_initializer, activation=activation)(model)
        if use_leaky_relu : model = LeakyReLU (0.2)(model)
    #
    #  Calculate Gaussian means with two more hidden layers
    #
    gauss_means     = Dense (D2*num_gaussians, kernel_initializer=custom_weight_init_hidden, bias_initializer=bias_initializer, activation=activation)(model)
    if use_leaky_relu : gauss_means = LeakyReLU (0.2)(gauss_means)
    gauss_means     = Dense (num_gaussians, kernel_initializer=custom_weight_init_hidden, bias_initializer=bias_initializer, activation="linear")(gauss_means)
    gauss_means     = Lambda(lambda x : gauss_mean_scale*x)                                            (gauss_means)
    gauss_means     = Lambda(lambda x : add_gauss_mean_offsets(x, num_gaussians, range_min, range_max))(gauss_means)
    #
    #  Calculate Gaussian widths with two more hidden layers
    #
    gauss_sigmas       = Dense (D2*num_gaussians   , kernel_initializer=custom_weight_init_hidden, bias_initializer=bias_initializer, activation=activation)(model)
    if use_leaky_relu : gauss_sigmas = LeakyReLU (0.2)(gauss_sigmas )
    gauss_sigmas       = Dense (num_gaussians     , kernel_initializer=custom_weight_init_hidden, bias_initializer=bias_initializer, activation=activation)(gauss_sigmas)
    gauss_sigmas = Lambda (lambda x : gauss_sigma_scale*x )                                            (gauss_sigmas)
    gauss_sigmas = Lambda (lambda x : set_initial_gauss_sigmas(x, num_gaussians, range_min, range_max))(gauss_sigmas)
    gauss_sigmas = Lambda (lambda x : K.log(1. + K.exp(x)))                                            (gauss_sigmas)
    gauss_sigmas = Lambda (lambda x : add_epsilon_to_gauss_sigmas(x, num_gaussians))                   (gauss_sigmas)
    #
    #  Calculate Gaussian fractions with two more hidden layers
    #
    gauss_fractions = Dense      (D2*num_gaussians , kernel_initializer=custom_weight_init_hidden, bias_initializer=bias_initializer, activation=activation)(model)
    if use_leaky_relu : gauss_fractions = LeakyReLU  (0.2)(gauss_fractions)
    gauss_fractions = Dense      (num_gaussians, kernel_initializer=custom_weight_init_hidden, bias_initializer=bias_initializer, activation="linear")(gauss_fractions)
    gauss_fractions = Lambda(lambda x : gauss_frac_scale*x)                                                    (gauss_fractions)
    gauss_fractions = Softmax()                                                                                (gauss_fractions)
    gauss_fractions = Lambda(lambda x : add_gauss_fraction_offsets(x, num_gaussians, min_gauss_amplitude_frac))(gauss_fractions)
    #
    #  Create Keras Model from inputs (external parameter and conditional observables) --> outputs (the Gaussian paramaters)
    #
    model = Concatenate()([gauss_fractions, gauss_means, gauss_sigmas])
    if num_observables_in > 0 : model = Model ([conditions_input, observables_input], model, name=name)
    else                      : model = Model (conditions_input, model, name=name)
    #
    #  Compile model
    #  -  loss function is the negative log-likelihood, where the PDF is a Gaussian mixture model
    #
    loss_function = lambda y_true, y_pred : -1. * K_dataset_log_likelihood (y_true, y_pred, num_gaussians)
    #loss_function = lambda y_true, y_pred : K_dataset_PlogP (y_true, y_pred, num_gaussians)
    if   optimiser.lower() == "sgd"      : model.compile(loss=loss_function, optimizer=SGD     (learning_rate=learning_rate))    
    elif optimiser.lower() == "adadelta" : model.compile(loss=loss_function, optimizer=Adadelta(learning_rate=learning_rate))    
    elif optimiser.lower() == "adam"     : model.compile(loss=loss_function, optimizer=Adam    (learning_rate=learning_rate))   
    else : raise ValueError(f"Optimiser '{optimiser}' not recognised") 
    if verbose : model.summary()
    #
    #  Return the Keras model, and its defining properties
    #   
    return model, (num_conditions_in, num_observables_in, num_gaussians)


#  Brief: create a discrete keras model describing the probability distribution over one observable
# 
def create_discrete_density_keras_model (name, **kwargs) :
    """create a Keras model which outputs the values of a multinomial probability distribution, conditioned on external parameters and other observables"""
    #
    #  =============================
    #  ===    Parse arguments    ===
    #  =============================
    #
    num_conditions_in        = int  (kwargs.get("num_conditions_in"                   ))
    num_observables_in       = int  (kwargs.get("num_observables_in", 0               ))
    num_outputs              = int  (kwargs.get("num_outputs"       , -1              ))
    verbose                  = bool (kwargs.get("verbose"           , True            ))
    learning_rate            = float(kwargs.get("learning_rate"     , 0.001           ))
    optimiser                = str  (kwargs.get("optimiser"         , "adam"          ))
    activation               = str  (kwargs.get("activation"        , "leakyrelu"     ))
    transform_min            = float(kwargs.get("transform_min"     , -2.             ))
    transform_max            = float(kwargs.get("transform_max"     , 2.              ))
    bias_initializer         = kwargs.get("bias_initializer"        , "zeros"         )
    condition_limits         = kwargs.get("condition_limits"        , None            )
    observables_limits       = kwargs.get("observables_limits"      , None            )
    #
    #  Sanitise inputs
    #
    if num_outputs < 2 :
        raise ValueError(f"num_outputs is {num_outputs}, but I cannot create a model with less than 2")
    #
    #  Parse arguments to configure number of layers
    #
    A1, A2   = int(kwargs.get("A1", 5)), int(kwargs.get("A2", 5))
    B1, B2   = int(kwargs.get("B1", 5)), int(kwargs.get("B2", 5))
    C_layers = kwargs.get("C_layers" , [10 + 3*num_outputs, 10 + 2*num_outputs])
    #
    #  If condition_limits provided, we want to transform the condition inputs, so we should make sure the input has the correct format
    #
    do_transform_conditions = False
    if type(condition_limits) != type(None) :
        do_transform_conditions = True
        transform_range = transform_max - transform_min
        assert len(condition_limits) == num_conditions_in, f"{condition_limits} expected to have shape [{num_conditions_in}, 2]"
        assert transform_range > 0, f"Transform range transform_max - transform_min = {transform_max} - {transform_min} = {transform_range} must be > 0"
        for row_idx, row in enumerate(condition_limits) :
            assert len(row) == 2, f"condition_limits row index {row_idx} is {row} where object of length 2 expected"
            assert row[1] >= row[0], f"condition_limits row index {row_idx} is {row} expected upper limit >= lower limit"
            if verbose :
                print(f"Projecting external parameter index {row_idx} from interval {row} onto [{transform_min}, {transform_max}]")
        def transform_conditions (x) :    # x shape is (None, num_conditions_in)
            cond_ranges = [(row[1]-row[0]) for row in condition_limits]
            cond_ranges = np.array([r if r > 0 else 1 for r in cond_ranges])
            out_min     = tf.constant([transform_min if row[1]>row[0] else 0.5*(transform_min+transform_max) for row in condition_limits])
            out_scale   = tf.constant([transform_range/row_range for row_range in cond_ranges])
            in_min      = tf.constant([row[0] for row in condition_limits])
            return out_min + ((x - in_min)*out_scale)
    #
    #  If observables_limits provided, we want to transform the observables inputs, so we should make sure the input has the correct format
    #
    do_transform_observables = False
    if type(observables_limits) != type(None) :
        do_transform_observables = True
        transform_range = transform_max - transform_min
        assert len(observables_limits) == num_observables_in, f"{observables_limits} expected to have shape [{num_observables_in}, 2]"
        assert transform_range > 0, f"Transform range transform_max - transform_min = {transform_max} - {transform_min} = {transform_range} must be > 0"
        for row_idx, row in enumerate(observables_limits) :
            assert len(row) == 2, f"observables_limits row index {row_idx} is {row} where object of length 2 expected"
            assert row[1] >= row[0], f"observables_limits row index {row_idx} is {row} expected upper limit >= lower limit"
            if verbose :
                print(f"Projecting observable index {row_idx} from interval {row} onto [{transform_min}, {transform_max}]")
        def transform_observables (x) :    # x shape is (None, num_conditions_in)
            cond_ranges = [(row[1]-row[0]) for row in observables_limits]
            cond_ranges = np.array([r if r > 0 else 1 for r in cond_ranges])
            out_min     = tf.constant([transform_min if row[1]>row[0] else 0.5*(transform_min+transform_max) for row in observables_limits])
            out_scale   = tf.constant([transform_range/row_range for row_range in cond_ranges])
            in_min      = tf.constant([row[0] for row in observables_limits])
            return out_min + ((x - in_min)*out_scale)
    #
    #  Print the configured settings
    #
    if verbose : 
        print(f"Creating continuous density model: {name}")
        print(f"  - num_conditions_in        is {num_conditions_in}")
        print(f"  - num_observables_in       is {num_observables_in}")
        print(f"  - num_outputs              is {num_outputs}")
        print(f"  - learning_rate            is {learning_rate}")
        print(f"  - optimiser                is {optimiser}")
        print(f"  - activation               is {activation}")
        print(f"  - transform range          is {transform_min:.4f}  to  {transform_max:.4f}")
        print(f"  - bias_initializer         is {bias_initializer}")
        print(f"  - adding hidden layer of size {A1 + A2*num_conditions_in} to pre-process condition inputs")
        if num_observables_in > 0 :
            print(f"  - adding hidden layer of size {B1 + B2*num_observables_in} to pre-process observables inputs")
        for c in C_layers :
            print(f"  - adding hidden layer of size {c}")
    #
    #  If LeakyReLU used as activation function, set activation of Dense layers to "linear", then LeakyReLU will be applied as a separate layer
    #
    use_leaky_relu = False
    if activation.lower() == "leakyrelu" :
        activation     = "linear"
        use_leaky_relu = True
    #
    #  ================================
    #  ===    Create Keras model    ===
    #  ================================
    #
    #  Create an input layer for the external parameter dependence
    #  -  if configured, add a layer which transforms these inputs onto the given domain
    #  -  add a layer to process just these inputs
    #
    conditions_input  = Input((num_conditions_in,))
    model_conditions  = conditions_input
    if do_transform_conditions : 
        model_conditions = Lambda(transform_conditions)(model_conditions)
    model_conditions  = Dense (A1 + A2*num_conditions_in, kernel_initializer=custom_weight_init_initial, bias_initializer=bias_initializer, activation=activation)(model_conditions) 
    if use_leaky_relu : 
        model_conditions  = LeakyReLU (0.2)(model_conditions )
    #
    #  If they exist, create an input layer for other input observables
    #  -  if configured, add a layer which transforms these inputs onto the given domain
    #  -  add a layer to process just these inputs
    #  -  concatenate the resulting hidden layer with that from the external parameter dependence
    #  If they don't exist, skip this step
    #
    if num_observables_in > 0 :
        observables_input = Input((num_observables_in,))
        model_observables = observables_input
        if do_transform_observables : 
            model_observables = Lambda(transform_observables)(model_observables)
        model_observables = Dense      (B1 + B2*num_observables_in, kernel_initializer=custom_weight_init_initial, bias_initializer=bias_initializer, activation=activation)(model_observables)    
        if use_leaky_relu : model_observables = LeakyReLU (0.2)(model_observables)
        model             = Concatenate()([model_conditions, model_observables])
    else :
        model = model_conditions
    #
    #  Add the configured number of additional hidden layers
    #
    for c in C_layers :
        model = Dense (c, kernel_initializer=custom_weight_init_hidden, bias_initializer=bias_initializer, activation=activation)(model)
        if use_leaky_relu : model = LeakyReLU (0.2)(model)
    #
    #  Add final layer of size num_outputs, with softmax activation to represent multinomial prob distribution
    #
    model = Dense (num_outputs, activation="softmax", kernel_initializer=custom_weight_init_hidden, bias_initializer=bias_initializer)(model)
    #
    #  Create Keras Model from inputs (external parameter and conditional observables) --> outputs (the Gaussian paramaters)
    #
    if num_observables_in > 0 : model = Model ([conditions_input, observables_input], model, name=name)
    else                      : model = Model (conditions_input, model, name=name)
    #
    #  Compile model
    #  -  loss function is the negative log-likelihood, where the PDF is a Gaussian mixture model
    #
    if   optimiser.lower() == "sgd"      : model.compile(loss="categorical_crossentropy", optimizer=SGD     (learning_rate=learning_rate))    
    elif optimiser.lower() == "adadelta" : model.compile(loss="categorical_crossentropy", optimizer=Adadelta(learning_rate=learning_rate))    
    elif optimiser.lower() == "adam"     : model.compile(loss="categorical_crossentropy", optimizer=Adam    (learning_rate=learning_rate))   
    else : raise ValueError(f"Optimiser '{optimiser}' not recognised") 
    if verbose : model.summary()
    #
    #  Return the Keras model, and its defining properties
    #   
    return model, (num_conditions_in, num_observables_in, num_outputs)


#  Brief: custom weight initialiser
# 
def custom_weight_init_hidden (shape, dtype=None, g=0.2) :
    """Keras: custom weight initialiser function for leaky relu layer with gradient g (not first layer in network)"""
    limit = 3. / np.sqrt(shape[0]) / (1 + g)
    return K.random_uniform(shape, -limit, limit, dtype=dtype)


#  Brief: custom weight initialiser
# 
def custom_weight_init_initial (shape, dtype=None, g=0.2) :
    """Keras: custom weight initialiser function for leaky relu layer with gradient g (first layer in network)"""
    limit = 4. / np.sqrt(shape[0]) / (1 + g)
    return K.random_uniform(shape, -limit, limit, dtype=dtype)


#  Brief: return the density at position x, given Gaussian parameters params
# 
def get_sum_gauss_density (x, params) :
    """return the density of datapoints x with shape [?, 1] for the Gaussian mixture model described by the 3*num_gaussians params"""
    num_gauss = int(len(params) / 3)
    fracs, means, sigmas = params[:num_gauss], params[num_gauss:2*num_gauss], params[2*num_gauss:3*num_gauss]
    return np_datapoint_likelihood (x, num_gauss, fracs, means, sigmas)


#  Brief: keras implementation of Gaussian probability density
# 
def K_gauss_prob (x, mean, sigma) :
    """return the Gaussian probability density for datapoints x"""
    prob = K.exp(-0.5*(x - mean)*(x - mean)/(sigma*sigma)) / K.sqrt(2*np.pi*sigma*sigma)
    return prob
    #return tf.where(tf.math.is_nan(prob), 1e-20*tf.ones_like(prob), prob)


#  Brief: keras implemention returning the likelihood for individual datapoints
# 
def K_datapoint_likelihood (x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas) :
    """Keras: return the probability density for datapoints x as described by the Gaussian mixture model"""
    prob = 0.
    x = x[:,0]
    for i in range(num_gauss) :
        prob = prob + gauss_fracs[:,i] * K_gauss_prob(x, gauss_means[:,i], gauss_sigmas[:,i])
    return prob


#  Brief: keras implemention returning the log likelihood for individual datapoints
# 
def K_datapoint_log_likelihood (x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas) :
    """Keras: return the log probability density for datapoints x as described by the Gaussian mixture model"""
    return K.log(K_datapoint_likelihood(x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas))


#  Brief: keras implemention returning the mean likelihood over a set of datapoints
# 
def K_dataset_mean_likelihood  (x, params, num_gauss) :
    """Keras: return the mean probability density for datapoints x as described by the Gaussian mixture model"""
    return K.mean(K_datapoint_likelihood(x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas))  


#  Brief: keras implemention returning the total likelihood over a set of datapoints
# 
def K_dataset_likelihood (x, params, num_gauss) :
    """Keras: return the combined probability density for datapoints x as described by the Gaussian mixture model (unstable - likely to underflow!)"""
    return K.prod(K_datapoint_likelihood(x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas))
        

#  Brief: keras implemention returning the mean log likelihood over a set of datapoints
# 
def K_dataset_log_likelihood (x, params, num_gauss) :
    """Keras: return the log probability density for datapoints x as described by the Gaussian mixture model"""
    gauss_fracs, gauss_means, gauss_sigmas = params[:,:num_gauss], params[:,num_gauss:2*num_gauss], params[:,2*num_gauss:3*num_gauss]
    return K_datapoint_log_likelihood(x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas)
        

#  Brief: keras implemention returning the mean P(x)*logP(x) over a set of datapoints
# 
def K_dataset_PlogP (x, params, num_gauss) :
    """Keras: return the log probability density for datapoints x as described by the Gaussian mixture model"""
    gauss_fracs, gauss_means, gauss_sigmas = params[:,:num_gauss], params[:,num_gauss:2*num_gauss], params[:,2*num_gauss:3*num_gauss]
    P = K_datapoint_likelihood(x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas)
    return P * K.log(P)


#  Brief: keras implemention returning the likelihood for individual datapoints
# 
def K_datapoint_likelihood_2 (x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas) :
    prob = 0.
    #x = x[:,0]
    for i in range(num_gauss) :
        prob = prob + gauss_fracs[:,i] * K_gauss_prob(x, gauss_means[:,i], gauss_sigmas[:,i])
    return prob


#  Brief: keras implemention returning the log likelihood for individual datapoints
# 
def K_datapoint_log_likelihood_2 (x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas) :
    return K.log(K_datapoint_likelihood_2(x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas))
        

#  Brief: keras implemention returning the mean log likelihood over a set of datapoints
# 
def K_dataset_log_likelihood_2 (x, params, num_gauss) :
    gauss_fracs, gauss_means, gauss_sigmas = params[:,:num_gauss], params[:,num_gauss:2*num_gauss], params[:,2*num_gauss:3*num_gauss]
    return K_datapoint_log_likelihood_2(x, num_gauss, gauss_fracs, gauss_means, gauss_sigmas)


#  Brief: Keras calculation of log-likelihood from a list of [floating point datapoints, Gaussian params], suitable for a keras lambda layer
#
def K_get_log_likelihood_for_float_datapoints (inputs, num_gauss) :
    """return log-likelihood from a list of [floating point datapoints, Gaussian params], suitable for a keras lambda layer"""
    eval_p, params_p = inputs[0], inputs[1]
    return K_dataset_log_likelihood_2 (eval_p, params_p, num_gauss)


#  Brief: Keras calculation of log-likelihood from a list of [integer datapoints as one-hot vector, catergorical probabilities], suitable for a keras lambda layer
#
def K_get_log_likelihood_for_int_datapoints (inputs, eval_p_min, eval_p_max) :
    """return log-likelihood from a list of [integer datapoints as one-hot vector, catergorical probabilities], suitable for a keras lambda layer"""
    eval_p, probs_p = inputs[0], inputs[1]
    one_hot_eval_p = tf.one_hot(tf.cast(eval_p-eval_p_min, "int32"), 1+int(eval_p_max-eval_p_min))
    one_hot_eval_p = tf.reshape(tensor=one_hot_eval_p, shape=(-1, 1+int(eval_p_max-eval_p_min)))
    likelihood = K.max(one_hot_eval_p * probs_p, axis=-1)
    return K.log(likelihood)


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


#  Brief: tf function for initialising the Gaussian widths (as a fraction of the initial range)
# 
def set_initial_gauss_sigmas (x, num_gauss, offset_min, offset_max) :
    """TF method: for input x of size [?, num_gauss], add a constant factor which sets initial Gaussian widths as (offset_max-offset_min) / num_gauss / Gauss_width_reduction_factor
       - to be applied before a Softmax function, so offset addition is performed in a logarithmic basis"""
    offset_range = float(offset_max - offset_min)
    target_width = offset_range / num_gauss / Gauss_width_reduction_factor
    offset       = float(np.log(np.exp(target_width) - 1))
    c = tf.convert_to_tensor([offset for i in range(num_gauss)])
    return x + c


#  Class to handle multiprocessing of Gaussian sampling
#  -  allows sampling in parallel for multicore optimisation
#
class GaussianSamplingProcess (Process) :
    def __init__ (self, gauss_params_queue, out_queue, n_points=1) :
        Process.__init__(self)
        self.gauss_params_queue = gauss_params_queue
        self.out_queue          = out_queue
        self.n_points           = n_points
    def run (self) :
        while not self.gauss_params_queue.empty() :
            batch_idx, gauss_params_batch = self.gauss_params_queue.get()
            to_return = (batch_idx, [sample_sum_gaussians(self.n_points, gauss_params) for gauss_params in gauss_params_batch])
            self.out_queue.put(to_return)

#  Class to handle piping of multiprocess outputs into a single list
#  -  prevents MP Queue buffer from creating deadlock
#
class OutputPipe (Thread) :
    def __init__ (self, queue, delay=0.2) :
        Thread.__init__(self)
        self.queue = queue
        self.delay = delay
    def complete (self) :
        self.run = False
    def run (self) :
        self.run    = True
        self.output = []
        while self.run or (not self.queue.empty()) :
            if self.queue.empty() :
                time.sleep(self.delay)
                continue
            self.output.append(self.queue.get())


#  Class ContinuousDensityModel
#    -  wrapper for constructing and training a density model for a continuous observable
#    -  allow it to be evaluated, sampled, fit and rebuilt
#
class ContinuousDensityModel () :
    #
    #  __init__
    #
    def __init__ (self, name, **kwargs) :
        """build a new Gaussian mixture model by passing the arguments provided onto self.rebuild"""
        self.rebuild(name, **kwargs)
    #
    #  evaluate
    #
    def evaluate (self, x, conditions, conditional_observables=[]) :
        """evaluate the likelihood of datapoints x (shape [?]) for given conditional inputs (shape [num_conditions]) and conditional observables (shape [?, num_observables]"""
        print("Warning - ContinuousDensityModel.evaluate is deprecated (please use DensityModel.evaluate)")
        ds_size    = len(x)
        conditions = [conditions for i in range(ds_size)]
        gauss_params_list = self.get_gauss_params(conditions, conditional_observables)
        return np.array([get_sum_gauss_density (xp, gauss_params) for xp, gauss_params in zip(x, gauss_params_list)])
    #
    #  fit
    #
    def fit (self, *argv, **kwargs) :
        """pass arguments on to the keras fit method"""
        self.model.fit(*argv, **kwargs)
    #
    #  get_gauss_params
    #
    def get_gauss_params (self, conditional_params, conditional_observables=[]) :
        """get the 3*num_gaussians parameters of the Gaussian mixture model for the given inputs"""
        if self.num_observables == 0 :
            return self.model.predict([conditional_params])
        return self.model.predict([conditional_params, conditional_observables])
    #
    #  rebuild
    #
    def rebuild (self, name, **kwargs) :
        """assign internal variables and recreate the keras model"""
        self.model, model_constants = create_continuous_density_keras_model (name, **kwargs)
        self.name            = name
        self.num_conditions  = model_constants[0]
        self.num_observables = model_constants[1]
        self.num_gaussians   = model_constants[2]
    #
    #  reset_weights
    #
    def reset_weights (self) :
        """reset the weights and biases of the keras model by running the initialisers in the current TF session"""
        session = tf.compat.v1.keras.backend.get_session()
        for layer in self.model.layers: 
            if hasattr(layer, "kernel") :
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, "bias") :
                layer.bias.initializer.run(session=session)
    #
    #  sample
    #
    def sample (self, n_points, conditional_params, conditional_observables=[], n_processes=1) :
        """draw n_points samples from the probability distribution evaluated with the given inputs"""
        #  If only one set of inputs, don't bother multiprocessing
        if (n_processes == 1) or (len(conditional_params) == 1) :
            gauss_params_list = self.get_gauss_params(conditional_params, conditional_observables)
            return np.array([sample_sum_gaussians (n_points, gauss_params) for gauss_params in gauss_params_list])
        #  Otherwise multiprocess if requested
        return self.sample_MP(n_points, conditional_params, conditional_observables, n_processes=n_processes)
    #
    #  sample_MP
    #
    def sample_MP (self, n_points, conditional_params, conditional_observables=[], n_processes=1) :
        """draw n_points samples from the probability distribution evaluated with the given inputs, accelerated with multiprocessing"""
        gauss_params_list = self.get_gauss_params(conditional_params, conditional_observables)
        #  Create queue of datapoints for parallel processing
        #  -  process in batches of 50 to reduce rate of Queue queries
        #  -  include batch index so we can order datapoints correctly after joining processes
        batch_idx, gauss_params_queue, sampled_datapoints_queue = 0, Queue(), Queue()
        while batch_idx < len(gauss_params_list) :
            next_idx = min(batch_idx+50, len(gauss_params_list))
            gauss_params_queue.put((batch_idx, gauss_params_list[batch_idx:next_idx]))
            batch_idx = next_idx
        #  Create and run processes
        #  -  OS blocks pipe if output Queue too large, leading to hanging on Process.join()
        #  -  instead we use a parallel thread OutputPipe to empty the queue as new results are created
        processes = []
        for proc_idx in range(n_processes) : processes.append(GaussianSamplingProcess(gauss_params_queue, sampled_datapoints_queue, n_points=n_points))
        output_pipe = OutputPipe(sampled_datapoints_queue)
        output_pipe.start()
        for process in processes : process.start()
        for process in processes : process.join()
        output_pipe.complete()
        output_pipe.join()
        #  Pull results from output pipe
        #  -  sorted() command sorts by the first index in the tuples, which are the batch indices, to ensure datapoints are ordered consistently with inputs
        sampled_datapoints_list = []
        for tup in sorted(output_pipe.output) :
            for sampled_datapoint in tup[1] :
                sampled_datapoints_list.append(sampled_datapoint)
        #  Return as array
        return np.array(sampled_datapoints_list)


#  Class DiscreteDensityModel
#
class DiscreteDensityModel :
    """Store, construct, train and sample a multinomial probability distribution for a discrete observable
       -  allow it to be evaluated, sampled, fit and rebuilt
       -  allow it to be saved to / loaded from pickle files"""
    #
    #  __init__
    #
    def __init__ (self, name, minimum, maximum, **kwargs) :
        """create discrete probability model between minimum and maximum integer values"""
        self.rebuild(name, minimum, maximum, **kwargs)
    #
    #  evaluate
    #
    def evaluate (self, x, conditional_params, conditional_observables=[]) :
        """evaluate probability model for observables x with shape [?, num_categories]"""
        print("Warning - DiscreteDensityModel.evaluate is deprecated (please use DensityModel.evaluate)")
        probs_list = self.get_categorical_probabilities(conditional_params, conditional_observables)
        x = self._x_to_idices(x)
        return np.array([probs[xp] for xp, probs in zip(x, probs_list)])
    #
    #  fit
    #
    def fit (self, X, Y, *argv, **kwargs) :
        """fit probability model to where X are inputs and Y are datapoints with shape [?, num_categories]"""
        Y = self._x_to_onehot(Y)
        self.model.fit(X, Y, *argv, **kwargs)
    #
    #  get_categorical_probabilities
    #
    def get_categorical_probabilities (self, conditional_params, conditional_observables=[]) :
        """return the probability spectrum for the given inputs"""
        if self.num_observables == 0 :
            return self.model.predict(conditional_params)
        return self.model.predict([conditional_params, conditional_observables])
    #
    #  rebuild
    #
    def rebuild (self, name, minimum, maximum, **kwargs) :
        """assign internal variables and create the keras model"""
        self.name            = name
        self.minimum         = int(minimum)
        self.maximum         = int(maximum)
        self.model, model_constants = create_discrete_density_keras_model (name, num_outputs=1+maximum-minimum, **kwargs)
        self.num_conditions  = model_constants[0]
        self.num_observables = model_constants[1]
    #
    #  reset_weights
    #
    def reset_weights (self) :
        """reset the weights and biases of the keras model by running the initialisers in the current TF session"""
        session = tf.compat.v1.keras.backend.get_session()
        for layer in self.model.layers: 
            if hasattr(layer, "kernel") :
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, "bias") :
                layer.bias.initializer.run(session=session)
    #
    #  sample
    #
    def sample (self, n_points, conditional_params, conditional_observables=[], n_processes=1) :
        """draw n_points samples from the probability distribution evaluated with the given inputs"""
        if n_processes > 1 :
            print(f"Warning - n_processes={n_processes} requested but multiprocessing not implemented for DiscreteDensityModel.sample()")
        probs_list = self.get_categorical_probabilities(conditional_params, conditional_observables)
        return np.array([np.random.choice(np.arange(self.minimum, self.maximum+1), n_points, p=probs/np.sum(probs)) for probs in probs_list])
    #
    #  _x_to_indices
    #
    def _x_to_indices (self, x) :
        """internal method which subtracts self.minimum from every x, which is a one-dimensional array of datapoints"""
        indices = []            
        if type(x) not in [list, np.ndarray] : x = [x]
        for val in x :
            if val > self.maximum : raise ValueError(f"Value {x} exceeds self.maximum ({self.maximum})")
            if val < self.minimum : raise ValueError(f"Value {x} is less than self.minimum ({self.minimum})")
            indices.append(int(val - self.minimum))
        return indices
    #
    #  _x_to_onehot
    #
    def _x_to_onehot (self, x) :
        """convert datapoints x with shape [?] into one-hot vectors with shape [?, num_categories]"""
        indices    = self._x_to_indices(x)
        row_length = 1 + self.maximum - self.minimum
        rows       = [onehot(row_length, idx) for idx in indices]
        return np.array(rows)

 
#  Class EvolvingLearningRate
#
class EvolvingLearningRate (Callback) :
    """Reduce the learning rate when the monitoring metric stops improving.

  Arguments:
      factor: Factor by which the lr is multiplied at each update.
      min_delta: Smallest tolerated improvement in the monitor.
      monitor: Metric to be monitored for improvement.
      patience: Number of epochs the monitor must improve over.
  """

    def __init__(self, factor=0.5, min_delta=0, min_lr=0, monitor="none", patience=0) :
        super(EvolvingLearningRate, self).__init__()
        self.factor    = factor
        self.min_delta = min_delta
        self.min_lr    = min_lr
        self.monitor   = monitor
        self.patience  = patience

    def on_train_begin(self, logs=None) :
        self.epochs_since_min = 0
        self.monitor_best_val = np.inf
        self.learning_rate    = self.model.optimizer.learning_rate
        self.initial_lr       = self.learning_rate.eval(session=tf.compat.v1.keras.backend.get_session())
        self.model.lr_record  = [(0, self.learning_rate)]

    def on_epoch_end(self, epoch, logs=None) :
        if self.monitor == "none" :
            if "val_loss" in logs :
                print(" - no monitor metric set, defaulting to val_loss")
                self.monitor = "val_loss"
            elif "loss" in logs : 
                print(" - no monitor metric set, defaulting to loss as val_loss is not available")
                self.monitor = "loss"
            else :
                raise ValueError(f"No monitor metric set")
        monitor_val = logs.get(self.monitor)
        if monitor_val < self.monitor_best_val :
            self.monitor_best_val = monitor_val
            self.epochs_since_min = 0
        else:
            self.epochs_since_min += 1
            if self.epochs_since_min >= self.patience :
                session    = tf.compat.v1.keras.backend.get_session()
                current_lr = self.learning_rate.eval(session=session)
                new_lr     = np.max([self.factor*current_lr, self.min_lr])
                session.run(self.learning_rate.assign(new_lr))
                self.monitor_best_val = monitor_val
                self.epochs_since_min = 0
                self.model.lr_record.append((epoch, new_lr))

    def on_train_end(self, logs=None) :
        session = tf.compat.v1.keras.backend.get_session()
        session.run(self.learning_rate.assign(self.initial_lr))
    

#  Class DensityModel
#
class DensityModel :
    """Store, construct, train and sample an autoregressive Gaussian mixture density model over many observables
       -  allow it to be evaluated, sampled, fit and rebuilt
       -  allow it to be saved to / loaded from pickle files"""
    #
    #  __init__
    #
    def __init__ (self, **kwargs) :
        """Initialise object by passing **kwargs to self.construct method"""
        self.construct(**kwargs)
    #
    #  build
    #
    def build (self, build_settings=None, verbose=True, **kwargs) :
        """Build (or re-build) the Keras models associated with this object
           -  build_settings is a dictionary of configuration settings if provided, otherwise load internally stored settings"""
        if type(build_settings) == type(None) :
            build_settings = {"name"                     : self.name                     , 
                              "num_gaussians"            : self.num_gaussians            , 
                              "num_conditions"           : self.num_conditions           , 
                              "num_observables"          : self.num_observables          , 
                              "types"                    : self.types                    , 
                              "condition_limits"         : self.condition_limits         ,
                              "observables_limits"       : self.observables_limits       ,
                              "transform_min"            : self.transform_min            ,
                              "transform_max"            : self.transform_max            ,
                              "bias_initializer"         : self.bias_initializer         ,
                              "optimiser"                : self.optimiser                ,
                              "learning_rate"            : self.learning_rate            ,
                              "learning_rate_evo_factor" : self.learning_rate_evo_factor ,
                              "activation"               : self.activation               ,
                              "A1"                       : self.A1                       ,
                              "A2"                       : self.A2                       ,
                              "B1"                       : self.B1                       ,
                              "B2"                       : self.B2                       ,
                              "C_float"                  : self.C_float                  ,
                              "C_int"                    : self.C_int                    ,
                              "D2"                       : self.D2                       ,
                              "gauss_frac_scale"         : self.gauss_frac_scale         ,
                              "gauss_mean_scale"         : self.gauss_mean_scale         ,
                              "gauss_sigma_scale"        : self.gauss_sigma_scale        ,
                              "min_gauss_amplitude_frac" : self.min_gauss_amplitude_frac }
        likelihood_models = []
        for obs_idx in range(build_settings["num_observables"]) :
            model_segment_name = build_settings["name"]+f"_observable{obs_idx}"
            if verbose : print("INFO".ljust(8) + "   " + "DensityModel.build".ljust(25) + "   " + f"Building model segment: {model_segment_name} for observable index {obs_idx}")
            observables_limits = build_settings["observables_limits"][:obs_idx] if obs_idx > 0 else None
            if self.types[obs_idx] == float :
                range_min, range_max = build_settings["observables_limits"][obs_idx]
                learning_rate = build_settings["learning_rate"] * (build_settings["learning_rate_evo_factor"] ** obs_idx)
                density_model = ContinuousDensityModel      (model_segment_name,
                                                             num_gaussians            = build_settings["num_gaussians" ]          ,
                                                             num_conditions_in        = build_settings["num_conditions"]          ,
                                                             num_observables_in       = obs_idx                                   ,
                                                             verbose                  = verbose                                   ,
                                                             condition_limits         = build_settings["condition_limits"]        ,
                                                             observables_limits       = observables_limits                        ,
                                                             range_min                = range_min                                 ,
                                                             range_max                = range_max                                 ,
                                                             transform_min            = build_settings["transform_min"]           ,
                                                             transform_max            = build_settings["transform_max"]           ,
                                                             bias_initializer         = build_settings["bias_initializer"]        ,
                                                             optimiser                = build_settings["optimiser"]               ,
                                                             learning_rate            = learning_rate                             ,
                                                             activation               = build_settings["activation"]              ,
                                                             A1                       = build_settings["A1"]                      ,
                                                             A2                       = build_settings["A2"]                      ,
                                                             B1                       = build_settings["B1"]                      ,
                                                             B2                       = build_settings["B2"]                      ,
                                                             C                        = build_settings["C_float"]                 ,
                                                             D2                       = build_settings["D2"]                      ,
                                                             gauss_frac_scale         = build_settings["gauss_frac_scale"]        ,
                                                             gauss_mean_scale         = build_settings["gauss_mean_scale"]        ,
                                                             gauss_sigma_scale        = build_settings["gauss_sigma_scale"]       ,
                                                             min_gauss_amplitude_frac = build_settings["min_gauss_amplitude_frac"])
            elif self.types[obs_idx] == int :
                range_min, range_max = build_settings["observables_limits"][obs_idx]
                density_model = DiscreteDensityModel        (model_segment_name,
                                                             minimum            = range_min                               ,
                                                             maximum            = range_max                               ,
                                                             num_conditions_in  = build_settings["num_conditions"]        ,
                                                             num_observables_in = obs_idx                                 ,
                                                             verbose            = verbose                                 ,
                                                             condition_limits   = build_settings["condition_limits"]      ,
                                                             observables_limits = observables_limits                      ,
                                                             transform_min      = build_settings["transform_min"]         ,
                                                             transform_max      = build_settings["transform_max"]         ,
                                                             bias_initializer   = build_settings["bias_initializer"]      ,
                                                             optimiser          = build_settings["optimiser"]             ,
                                                             learning_rate      = learning_rate                           ,
                                                             activation         = build_settings["activation"]            ,
                                                             A1                 = build_settings["A1"]                    ,
                                                             A2                 = build_settings["A2"]                    ,
                                                             B1                 = build_settings["B1"]                    ,
                                                             B2                 = build_settings["B2"]                    ,
                                                             C_layers           = build_settings["C_int"]                 ,
                                                             D2                 = build_settings["D2"]                    )
            else :
                raise TypeError(f"Observable index {obs_idx} requested an unrecognised type {self.types[obs_idx]}")
            likelihood_models.append(density_model)
        if verbose : print("INFO".ljust(8) + "   " + "DensityModel.build".ljust(25) + "   " + f"{len(likelihood_models)} partial density models constructed")
        self.build_settings    = build_settings
        self.likelihood_models = likelihood_models
        self.fit_record        = {}
        self.create_evaluator_model_for_density_models (build_settings["name"]+"_evaluator")
    #
    #  construct
    #
    def construct (self, **kwargs) :
        """Configure the model, and call the self.build method"""
        #
        #  Parse input arguments
        #
        name                     = kwargs.get("name"              , None       )
        num_gaussians            = kwargs.get("num_gaussians"     , None       )
        num_conditions           = kwargs.get("num_conditions"    , None       )
        num_observables          = kwargs.get("num_observables"   , None       )
        types                    = kwargs.get("types"             , None       )
        condition_limits         = kwargs.get("condition_limits"  , None       ) 
        observables_limits       = kwargs.get("observables_limits", None       )   
        verbose                  = kwargs.get("verbose"           , True       ) 
        do_build                 = kwargs.get("build"             , True       )  
        learning_rate            = kwargs.get("learning_rate"     , 0.001      ) 
        learning_rate_evo_factor = kwargs.get("learning_rate_evo_factor", 1.   )   
        optimiser                = kwargs.get("optimiser"         , "adam"     )   
        bias_initializer         = kwargs.get("bias_initializer"  , "zeros"    )
        activation               = kwargs.get("activation"        , "leakyrelu")
        transform_min            = float(kwargs.get("transform_min", -2.))
        transform_max            = float(kwargs.get("transform_max",  2.))
        A1                       = int(kwargs.get("A1"            , 10))
        A2                       = int(kwargs.get("A2"            , 10))
        B1                       = int(kwargs.get("B1"            , 10))
        B2                       = int(kwargs.get("B2"            , 10))
        C_float                  = int(kwargs.get("C_float"       , 0))
        C_int                    = kwargs.get("C_int"             , [32])
        D2                       = int(kwargs.get("D2"            , 2))
        gauss_frac_scale         = float(kwargs.get("gauss_frac_scale" , 1./8.))
        gauss_mean_scale         = float(kwargs.get("gauss_mean_scale" , 1./8.))
        gauss_sigma_scale        = float(kwargs.get("gauss_sigma_scale", 1./8.))
        min_gauss_amplitude_frac = float(kwargs.get("min_gauss_amplitude_frac", 0.))
        #
        #  Check inputs are sensible
        #
        if type(name)            == type(None) : raise ArgumentError("Missing argument: name")
        if type(num_gaussians)   == type(None) : raise ArgumentError("Missing argument: num_gaussians")
        if type(num_conditions)  == type(None) : raise ArgumentError("Missing argument: num_conditions")
        if type(num_observables) == type(None) : raise ArgumentError("Missing argument: num_observables")
        if type(types) == type(None) :
            if verbose : print(f"No observable types argument provided - assuming all are floats")
            types = [float for i in range(num_observables)]
        if type(observables_limits) == type(None) :
            if verbose : print(f"No observables_limits argument provided - assuming they are [-5, 5] for all observables")
            observables_limits = [[-5., 5.] for i in range(num_observables)]
        if len(observables_limits) != num_observables : raise ValueError(f"len(observables_limits) != num_observables  ({len(observables_limits)} != {num_observables})")
        if type(name)            != str : raise TypeError(f"name argument {name} of type {type(name)} where {type(int)} expected")
        if type(num_gaussians)   != int : raise TypeError(f"num_gaussians argument {num_gaussians} of type {type(num_gaussians)} where {type(int)} expected")
        if type(num_conditions)  != int : raise TypeError(f"num_conditions argument {num_conditions} of type {type(num_conditions)} where {type(int)} expected")
        if type(num_observables) != int : raise TypeError(f"num_observables argument {num_observables} of type {type(num_observables)} where {type(int)} expected")
        if len(types) != num_observables : raise TypeError(f"length of types argument {types} ({len(types)}) != num_observables ({num_observables})")
        if num_gaussians   < 1 : raise ValueError(f"num_gaussians must be > 0, but {num_gaussians} provided")
        if num_conditions  < 1 : raise ValueError(f"num_conditions must be > 0, but {num_conditions} provided")
        if num_observables < 1 : raise ValueError(f"num_observables must be > 0, but {num_observables} provided")
        if min_gauss_amplitude_frac < 0. : raise ValueError(f"min_gauss_amplitude_frac must be >= 0, but {min_gauss_amplitude_frac} provided")
        if min_gauss_amplitude_frac > 1. : raise ValueError(f"min_gauss_amplitude_frac must be <= 1, but {min_gauss_amplitude_frac} provided")
        #
        #  Print configuration
        #
        if verbose :
            print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set model name              : {name}"           )
            print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set model num_gaussians     : {num_gaussians}"  )
            print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set model num_conditions    : {num_conditions}" )
            print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set model num_observables   : {num_observables}")
            print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set observable types        : {types}")
            print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set bias_initializer        : {bias_initializer}")
            print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set learning_rate           : {learning_rate}")
            print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set learning_rate_evo_factor: {learning_rate_evo_factor}")
            print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set optimiser               : {optimiser}")
            print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set activation              : {activation}")
            print("INFO".ljust(8) + "   " + "DensityModel.construct".ljust(25) + "   " + f"Set min_gauss_amplitude_frac: {min_gauss_amplitude_frac}")
        #
        #  Save configuration
        #
        self.name                     = name
        self.num_gaussians            = num_gaussians
        self.num_conditions           = num_conditions
        self.num_observables          = num_observables
        self.condition_limits         = condition_limits
        self.observables_limits       = observables_limits
        self.types                    = types
        self.bias_initializer         = bias_initializer
        self.learning_rate            = learning_rate
        self.learning_rate_evo_factor = learning_rate_evo_factor
        self.optimiser                = optimiser
        self.activation               = activation
        self.A1                       = A1
        self.A2                       = A2
        self.B1                       = B1
        self.B2                       = B2
        self.C_float                  = C_float
        self.C_int                    = C_int
        self.D2                       = D2
        self.gauss_frac_scale         = gauss_frac_scale
        self.gauss_mean_scale         = gauss_mean_scale
        self.gauss_sigma_scale        = gauss_sigma_scale
        self.transform_min            = transform_min
        self.transform_max            = transform_max
        self.min_gauss_amplitude_frac = min_gauss_amplitude_frac
        self.build_settings           = None
        #
        #  Build the Keras models associated with these settings
        #
        if do_build is False : return
        self.build(verbose=verbose)
    #
    #  create_evaluator_model_for_density_models
    #
    def create_evaluator_model_for_density_models (self, name) :
        """return named model which returns the log-likelihood for set of datapoints using the DensityModel model"""
        density_models  = [x for x in self.likelihood_models]
        if len(density_models) != self.num_observables :
            raise RuntimeError(f"Cannot create evaluator model. I am configured for {num_observables} observables but I have {len(density_models)} models stored") 
        #
        #  Create input layers for conditions and observables
        #
        condition_input_layer = Input((self.num_conditions ,), name=f"{name}_condition_input" )
        obs_input_layer       = Input((self.num_observables,), name=f"{name}_observable_input")
        #
        #  Create a list of layers which calculate the logL for each observable
        #
        logL_layers = []
        for obs_idx, (obs_type, density_model) in enumerate(zip(self.types, density_models)) :
            obs_model    = density_model.model
            model_inputs = [condition_input_layer]
            if obs_idx > 0 :
                cond_obs_split_layer = Lambda(lambda x : x[:, :obs_idx], name=f"{name}_split_observable_below{obs_idx}")(obs_input_layer)
                model_inputs.append(cond_obs_split_layer)
            params = obs_model(model_inputs)
            eval_p = Lambda(lambda x : x[:, obs_idx], name=f"{name}_split_observable_eq{obs_idx}")(obs_input_layer)
            if   obs_type is int : logL_layer = Lambda(lambda x : K_get_log_likelihood_for_int_datapoints  (x, density_model.minimum, density_model.maximum), name=f"{name}_logL_observable{obs_idx}")([eval_p, params])
            else                 : logL_layer = Lambda(lambda x : K_get_log_likelihood_for_float_datapoints(x, self.num_gaussians)                          , name=f"{name}_logL_observable{obs_idx}")([eval_p, params])
            logL_layer = tf.reshape(tensor=logL_layer, shape=(-1, 1))
            logL_layers.append(logL_layer)
        #
        #  Create a trainable model which outputs the logL
        #
        logL_final = Lambda(lambda x : tf.add_n(x), name=f"{name}_logL_total")(logL_layers)
        logL_model = Model([condition_input_layer, obs_input_layer], logL_final, name=name)
        logL_model.compile(loss=lambda y_true, y_pred : y_pred, optimizer="adam")
        self.evaluator = logL_model
        #
        #  Create a second model which outputs a list of the logL in all layers, as well as the total
        #
        all_logL = Concatenate()(logL_layers + [logL_final])
        all_logL = Model([condition_input_layer, obs_input_layer], all_logL, name=f"{name}_logL_split")
        all_logL.compile(loss=lambda y_true, y_pred : y_pred, optimizer="adam")
        self.split_evaluator = all_logL
    #
    #  ensure_valid_over_dataset
    #
    def ensure_valid_over_dataset (self, data, weights, max_attempts=100, verbose=True) :
        attempt = 0
        while attempt < max_attempts :
            if verbose : print(f"Evaluating losses  (attempt {attempt+1} / {max_attempts})")
            start_time = time.time()
            losses     = self.evaluate_over_dataset(data, weights)
            if verbose : 
                print(f"-  observable logL are {losses[:-1]}, combined is {losses[-1]}")
                print(f"-  eval completed in {int(time.time() - start_time):.0f}s")
            indices_to_retry = [idx for idx, loss in zip(np.arange(self.num_observables), losses) if not np.isfinite(loss)]
            if len(indices_to_retry) > 0 :
                attempt += 1
                if verbose :
                    if attempt == max_attempts : print(f"-  max_attempts reached, remaining models will not be rebuilt")
                    else : print(f"-  rebuilding models for indices {', '.join([f'{idx}' for idx in indices_to_retry])}")
            else :
                attempt = max_attempts
            for idx in indices_to_retry :
                self.likelihood_models[idx].reset_weights()
    #
    #  evaluate
    #
    def evaluate (self, conditions, observables) :
        """return the likelihood for the datapoints provided
           - [arg 1] conditions is a list of external parameter dependence
           - [arg 2] observables is a list of datapoints of shape [num_datapoints, num_observables]
           - returns the logL for every datapoint, with shape [num_datapoints]"""
        num_datapoints  = observables.shape[0]
        num_observables = observables.shape[1]
        if type(conditions) not in [np.ndarray, list] : conditions = [conditions]
        if type(conditions) is list : conditions = np.array(conditions)
        conditions = np.array([conditions for i in range(num_datapoints)])
        if len(conditions.shape) == 1 :
            conditions.reshape(len(conditions), 1)
        if len(observables.shape) == 1 :
            observables = observables.copy()
            observables.reshape(len(observables), 1)
        logL = self.evaluator.predict([conditions, observables])
        return np.exp(logL)
    #
    #  evaluate
    #
    def evaluate_over_dataset (self, data, weights) :
        fit_X = np.concatenate([np.full(fill_value=c, shape=(len(d),self.num_conditions)) for c,d in data.items()])
        fit_Y = np.concatenate([d for c,d in data   .items()])
        fit_W = np.concatenate([d for c,d in weights.items()])
        losses = self.split_evaluator.predict([fit_X, fit_Y])
        losses = np.multiply(fit_W[:, np.newaxis], losses)
        losses = np.sum(losses, axis=0)
        return losses
    #
    #  fit
    #
    def fit (self, dataset, weights=None, **kwargs) :
        """fit density models to the dataset provided"""                                  
        #  Parse settings
        #
        observable_idx             = kwargs.get("observable"                , None )
        max_epochs_per_observable  = kwargs.get("max_epochs_per_observable" , 2000 )
        early_stopping_patience    = kwargs.get("early_stopping_patience"   , 100  )
        early_stopping_min_delta   = kwargs.get("early_stopping_min_delta"  , 0    )
        batch_size_per_observable  = kwargs.get("batch_size_per_observable" , -1   )
        validation_split           = kwargs.get("validation_split"          , 0.3  )
        do_build                   = kwargs.get("build"                     , False)
        verbose                    = kwargs.get("verbose"                   , True )
        tf_verbose                 = kwargs.get("tf_verbose"                , 1    )
        learning_rate_evo_factor   = kwargs.get("learning_rate_evo_factor"  , 1    )
        learning_rate_evo_patience = kwargs.get("learning_rate_evo_patience", 0    )
        monitor = "val_loss"
        if validation_split <= 0 : monitor = "loss"
        #                                                        
        #  (Re-)build model if requested
        #
        if do_build : self.build(**kwargs)
        #                 
        #  Make sure model has been built, and with the same settings as currently set
        #                                     
        if hasattr(self, "build_settings") is False : raise RuntimeError(f"self.build_settings does not exist - you must call self.build() before self.fit(), or specify self.fit(build=True)")
        build_settings = {"name":self.name, "num_gaussians":self.num_gaussians, "num_conditions":self.num_conditions, "num_observables":self.num_observables, "types":self.types, 
                          "condition_limits":self.condition_limits, "observables_limits":self.observables_limits, 
                          "transform_min":self.transform_min, "transform_max":self.transform_max, "bias_initializer":self.bias_initializer, "optimiser":self.optimiser, "learning_rate":self.learning_rate,
                          "activation":self.activation, "A1":self.A1, "A2":self.A2, "B1":self.B1, "B2":self.B2, "C_float":self.C_float, "C_int":self.C_int, "D2":self.D2, 
                          "gauss_frac_scale":self.gauss_frac_scale, "gauss_mean_scale":self.gauss_mean_scale, "gauss_sigma_scale":self.gauss_sigma_scale, "min_gauss_amplitude_frac":self.min_gauss_amplitude_frac}
        for setting, value in build_settings.items() :
            built_value = self.build_settings [setting]
            if built_value == value : continue
            raise ValueError(f"Setting {setting}={value} has changed since last build (with {setting}={built_value}. You must specify self.fit(build=True) if you want to re-build the model.")                        
        #                         
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
        #                                           
        #  Create weights if None provided, and make sure all samples normalised equally
        # 
        if type(weights) == type(None) :
            if verbose : print("INFO".ljust(8) + "   " + "DensityModel.fit".ljust(25) + "   " + "Creating weights, as None provided")
            weights = {}
            for condition, datapoints in dataset.items() :
                weights [condition] = np.full(fill_value=1./len(datapoints), shape=(len(datapoints),))
        #
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
        #
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
            #                                          
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
            callbacks = [EarlyStopping(patience=early_stopping_patience, restore_best_weights=True, monitor=monitor, min_delta=early_stopping_min_delta)]
            if learning_rate_evo_factor != 1 :
                callbacks.append(EvolvingLearningRate(factor=learning_rate_evo_factor, monitor=monitor, patience=learning_rate_evo_patience))
            start_time = time.time()
            fit_record = self.likelihood_models[observable_idx].fit(
                                                       train_data_X,
                                                       train_data_Y,
                                                       sample_weight    = train_data_weights,
                                                       validation_split = validation_split,
                                                       epochs           = max_epochs_per_observable,
                                                       shuffle          = True,
                                                       batch_size       = batch_size,
                                                       callbacks        = callbacks)
            self.fit_record[observable_idx] = fit_record
            print(f"Fit completed in {int(time.time() - start_time):.0f}s")
    #  
    #  from_dir
    #  
    @classmethod
    def from_dir (cls, dirname, verbose=False) :
        """Create a new instance from the data saved in the directory provided"""
        ret = cls(name="dflt", num_gaussians=2, num_observables=1, num_conditions=1, do_build=False, verbose=verbose)
        ret.load_from_dir(dirname)
        return ret
    #  
    #  load_from_dir
    #  
    def load_from_dir (self, dirname) :
        #  Open the input file
        pfile_name = dirname + "/density_model.pickle"
        to_load = pickle.load(open(pfile_name, "rb"))
        #  Load all settings
        def get_from_dictionary (dictionary, label) :
            if label not in dictionary :
                raise RuntimeError(f"Required entry '{label}' not found in file '{pfile_name}'")
            return dictionary [label]
        model_files                   = get_from_dictionary (to_load, "model_files")
        build_settings                = get_from_dictionary (to_load, "build_settings")
        self.name                     = get_from_dictionary (to_load, "name")
        self.num_gaussians            = get_from_dictionary (to_load, "num_gaussians")
        self.num_conditions           = get_from_dictionary (to_load, "num_conditions")
        self.num_observables          = get_from_dictionary (to_load, "num_observables")
        self.condition_limits         = get_from_dictionary (to_load, "condition_limits")
        self.observables_limits       = get_from_dictionary (to_load, "observables_limits")
        self.types                    = get_from_dictionary (to_load, "types")
        self.bias_initializer         = get_from_dictionary (to_load, "bias_initializer")
        self.learning_rate            = get_from_dictionary (to_load, "learning_rate")
        self.learning_rate_evo_factor = get_from_dictionary (to_load, "learning_rate_evo_factor")
        self.optimiser                = get_from_dictionary (to_load, "optimiser")
        self.activation               = get_from_dictionary (to_load, "activation")
        self.A1                       = get_from_dictionary (to_load, "A1")
        self.A2                       = get_from_dictionary (to_load, "A2")
        self.B1                       = get_from_dictionary (to_load, "B1")
        self.B2                       = get_from_dictionary (to_load, "B2")
        self.C_float                  = get_from_dictionary (to_load, "C_float")
        self.C_int                    = get_from_dictionary (to_load, "C_int")
        self.D2                       = get_from_dictionary (to_load, "D2")
        self.gauss_frac_scale         = get_from_dictionary (to_load, "gauss_frac_scale")
        self.gauss_mean_scale         = get_from_dictionary (to_load, "gauss_mean_scale")
        self.gauss_sigma_scale        = get_from_dictionary (to_load, "gauss_sigma_scale")
        self.transform_min            = get_from_dictionary (to_load, "transform_min")
        self.transform_max            = get_from_dictionary (to_load, "transform_max")
        self.min_gauss_amplitude_frac = get_from_dictionary (to_load, "min_gauss_amplitude_frac")
        if "fit_record" in to_load :
            self.fit_record           = get_from_dictionary (to_load, "fit_record")
        #  Build a set of likelihood models using these settings
        self.build (build_settings=build_settings, verbose=False)
        #  Load the model weights
        for idx, (likelihood_model, model_fname) in enumerate(zip(self.likelihood_models, to_load["model_files"])) :
            likelihood_model.model.load_weights(model_fname)
    #
    #  sample
    #
    def sample (self, n_points, conditions, num_observables=None, n_processes=1, verbose=True) :
        """draw n_points samples from the density model evaluated at the conditional parameters provided
           - if num_observables is provided, sample the first num_observables observables only"""
        if type(num_observables) == type(None) :
            num_observables = self.num_observables
        if verbose : print("INFO".ljust(8) + "   " + "DensityModel.sample".ljust(25) + "   " + f"Sampling {n_points} datapoints, observable index is 0")  
        X1_to_XN = [self.likelihood_models[0].sample (n_points, [conditions], n_processes=n_processes).reshape(n_points)]
        conditions = [conditions for i in range(n_points)]
        for obs_idx in range(1, num_observables) :
            if verbose : print("INFO".ljust(8) + "   " + "DensityModel.sample".ljust(25) + "   " + f"Sampling {n_points} datapoints, observable index is {obs_idx}")  
            XN = self.likelihood_models[obs_idx].sample(1, conditions, np.array(X1_to_XN).transpose(), n_processes=n_processes).reshape(n_points)
            X1_to_XN.append(XN)
        return np.array(X1_to_XN).transpose()
    #
    #  save_to_dir
    #
    def save_to_dir (self, dirname) :
        """Create a directory storing all information for this model, including all of the model parameters"""
        #  First make sure the target directory exists
        pfile_name = dirname + "/density_model.pickle"
        make_sure_dir_exists_for_filename(pfile_name)
        #  Save the keras model parameters to files in this directory, and keep a record of the filenames
        model_files = []
        for idx, likelihood_model in enumerate(self.likelihood_models) :
            model_fname = dirname + f"/tf_model_weights_obs{idx}"
            likelihood_model.model.save_weights(model_fname)
            model_files.append(model_fname)
        # Create a pickled dictionary storing all class settings, and the filenames which store the parameter values
        to_pickle = {}
        to_pickle ["model_files"]              = model_files
        to_pickle ["build_settings"]           = self.build_settings
        to_pickle ["name"]                     = self.name
        to_pickle ["num_gaussians"]            = self.num_gaussians
        to_pickle ["num_conditions"]           = self.num_conditions
        to_pickle ["num_observables"]          = self.num_observables
        to_pickle ["condition_limits"]         = self.condition_limits
        to_pickle ["observables_limits"]       = self.observables_limits
        to_pickle ["types"]                    = self.types
        to_pickle ["bias_initializer"]         = self.bias_initializer
        to_pickle ["learning_rate"]            = self.learning_rate
        to_pickle ["learning_rate_evo_factor"] = self.learning_rate_evo_factor
        to_pickle ["optimiser"]                = self.optimiser
        to_pickle ["activation"]               = self.activation
        to_pickle ["A1"]                       = self.A1
        to_pickle ["A2"]                       = self.A2
        to_pickle ["B1"]                       = self.B1
        to_pickle ["B2"]                       = self.B2
        to_pickle ["C_float"]                  = self.C_float
        to_pickle ["C_int"]                    = self.C_int
        to_pickle ["D2"]                       = self.D2
        to_pickle ["gauss_frac_scale"]         = self.gauss_frac_scale
        to_pickle ["gauss_mean_scale"]         = self.gauss_mean_scale
        to_pickle ["gauss_sigma_scale"]        = self.gauss_sigma_scale
        to_pickle ["transform_min"]            = self.transform_min
        to_pickle ["transform_max"]            = self.transform_max
        to_pickle ["min_gauss_amplitude_frac"] = self.min_gauss_amplitude_frac
        to_pickle ["fit_record"]               = self.fit_record
        pickle.dump(to_pickle, open(pfile_name, "wb"))

        
