#   Implementation of WGAN and associated methods
#   Author:  Stephen Menary  (stmenary@cern.ch)


from keras.layers     import BatchNormalization, Dense, Dropout, Input, LeakyReLU, Concatenate
from keras.models     import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop

import keras.backend as K


#  Brief:  keras implementation of Wasserstein loss
#
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


#  Brief:  keras implementation of Wasserstein GAN
#
def create_critic_generator_wgan (**kwargs) :
    #
    #  Parse settings
    #
    dropout            = kwargs.get("dropout"        , -1.)
    GAN_noise_size     = kwargs.get("GAN_noise_size" , 3  )
    leaky_relu         = kwargs.get("leaky_relu"     , 0.2)
    num_observables    = kwargs.get("num_observables")
    num_conditions     = kwargs.get("num_conditions" )
    verbose            = kwargs.get("verbose"        , True)
    use_batch_norm     = kwargs.get("batch_norm"     , False)

    use_dropout = False
    if dropout > 0 : use_dropout = True

    critic_data_layers      = kwargs.get("critic_data_layers"     , (10,))
    critic_condition_layers = kwargs.get("critic_condition_layers", (10,))
    critic_combined_layers  = kwargs.get("critic_combined_layers" , (20, 20,))

    generator_noise_layers     = kwargs.get("generator_noise_layers"    , (20,))
    generator_condition_layers = kwargs.get("generator_condition_layers", (10,))
    generator_combined_layers  = kwargs.get("generator_combined_layers" , (20, 20,))
    #
    #  Print stage
    #
    if verbose : print(f"Creating WGAN with {num_observables} observables and {num_conditions} conditions")
    #
    #  Create input layers
    #
    data_input      = Input((num_observables,))
    condition_input = Input((num_conditions ,))
    noise_input     = Input((GAN_noise_size ,))
    #
    #  Create initially separate layers for the condition and data (critic)
    #
    critic_data = data_input
    for layer_size in critic_data_layers :
        critic_data = Dense(layer_size)(critic_data)
        critic_data = LeakyReLU(leaky_relu)(critic_data)
        if use_dropout : critic_data = Dropout(dropout)(critic_data)

    critic_condition = condition_input
    for layer_size in critic_condition_layers :
        critic_condition = Dense(layer_size)(critic_condition)
        critic_condition = LeakyReLU(leaky_relu)(critic_condition)
        if use_dropout : critic_condition = Dropout(dropout)(critic_condition)
    #
    #  Concatenate the condition and data latent states (critic)
    #
    critic = Concatenate()([critic_data, critic_condition])
    #
    #  Create final critic layers
    #
    for layer_size in critic_combined_layers :
        critic = Dense(layer_size)(critic)
        critic = LeakyReLU(leaky_relu)(critic)
        if use_dropout : critic = Dropout(dropout)(critic)
    #
    #  Compile critic model
    #
    critic = Dense(1, activation="linear")(critic)
    critic = Model(name="Critic", inputs=[data_input, condition_input], outputs=[critic])
    critic.compile(loss=wasserstein_loss, optimizer=RMSprop(learning_rate=5e-5, rho=0))
    if verbose : critic.summary()
    #
    #  Create initially separate layers for the noise and data (generator)
    #
    generator_noise = noise_input
    for layer_size in generator_noise_layers :
        generator_noise = Dense(layer_size)(generator_noise)
        generator_noise = LeakyReLU(leaky_relu)(generator_noise)
        if use_batch_norm : generator_noise = BatchNormalization()(generator_noise)

    generator_condition = condition_input
    for layer_size in generator_condition_layers :
        generator_condition = Dense(layer_size)(generator_condition)
        generator_condition = LeakyReLU(leaky_relu)(generator_condition)
        if use_batch_norm : generator_condition = BatchNormalization()(generator_condition)
    #
    #  Concatenate the condition and noise latent states (generator)
    #
    generator = Concatenate()([generator_noise, generator_condition])
    #
    #  Create final generator layers
    #
    for layer_size in generator_combined_layers :
        generator = Dense(layer_size)(generator)
        generator = LeakyReLU(leaky_relu)(generator)
        if use_batch_norm : generator = BatchNormalization()(generator)
    #
    #  Compile generator model
    #
    generator = Dense(num_observables, activation="linear")(generator)
    generator = Model(name="Generator", inputs=[noise_input, condition_input], outputs=[generator])
    if verbose : generator.summary()
    #
    #  Create and compile GAN
    #
    GAN = critic([generator([noise_input, condition_input]), condition_input])
    GAN = Model([noise_input, condition_input], GAN, name="GAN")
    critic.trainable = False
    GAN.compile(loss=wasserstein_loss, optimizer=RMSprop(learning_rate=5e-5, rho=0))
    if verbose : GAN.summary()
    #
    #  return critic, generator, GAN
    #
    return critic, generator, GAN


#  Brief:  keras implementation of conditional discriminator
#
def create_conditional_discriminator (**kwargs) :
    #
    #  Parse settings
    #
    dropout         = kwargs.get("dropout"        , -1.)
    leaky_relu      = kwargs.get("leaky_relu"  , 0.2)
    num_categories  = kwargs.get("num_categories" )
    num_conditions  = kwargs.get("num_conditions" )
    num_observables = kwargs.get("num_observables")
    use_batch_norm  = kwargs.get("batch_norm"     , False)
    verbose         = kwargs.get("verbose"        , True)

    use_dropout = False
    if dropout > 0 : use_dropout = True

    data_layers      = kwargs.get("data_layers"     , (10,))
    condition_layers = kwargs.get("condition_layers", (10,))
    combined_layers  = kwargs.get("combined_layers" , (20, 20,))
    #
    #  Print stage
    #
    if verbose : print(f"Creating discriminator with {num_observables} observables and {num_conditions} conditions")
    #
    #  Create input layers
    #
    data_input      = Input((num_observables,))
    condition_input = Input((num_conditions ,))
    #
    #  Create initially separate layers for the condition and data
    #
    discriminator_data = data_input
    for layer_size in data_layers :
        discriminator_data = Dense(layer_size)(discriminator_data)
        discriminator_data = LeakyReLU(leaky_relu)(discriminator_data)
        if use_batch_norm : discriminator_data = BatchNormalization()(discriminator_data)
        if use_dropout    : discriminator_data = Dropout(dropout)(discriminator_data)

    discriminator_condition = condition_input
    for layer_size in condition_layers :
        discriminator_condition = Dense(layer_size)(discriminator_condition)
        discriminator_condition = LeakyReLU(leaky_relu)(discriminator_condition)
        if use_batch_norm : discriminator_condition = BatchNormalization()(discriminator_condition)
        if use_dropout    : discriminator_condition = Dropout(dropout)(discriminator_condition)
    #
    #  Concatenate the condition and data latent states
    #
    discriminator = Concatenate()([discriminator_data, discriminator_condition])
    #
    #  Create final discriminator layers
    #
    for layer_size in combined_layers :
        discriminator = Dense(layer_size)(discriminator)
        discriminator = LeakyReLU(leaky_relu)(discriminator)
        if use_batch_norm : discriminator = BatchNormalization()(discriminator)
        if use_dropout    : discriminator = Dropout(dropout)(discriminator)
    #
    #  Compile discriminator model
    #
    discriminator = Dense(num_categories, activation="sigmoid")(discriminator)
    discriminator = Model(name="Discriminator", inputs=[data_input, condition_input], outputs=[discriminator])
    if num_categories == 1 :
        discriminator.compile(loss="binary_crossentropy", optimizer=Adam())
    else :
        discriminator.compile(loss="categorical_crossentropy", optimizer=Adam())
    if verbose : discriminator.summary()
    #
    #  return discriminator
    #
    return discriminator


#  Brief:  keras implementation of a network with two discrete inputs
#
def create_conditional_model (**kwargs) :
    #
    #  Parse settings
    #
    dropout         = kwargs.get("dropout"        , -1.)
    leaky_relu         = kwargs.get("leaky_relu"  , 0.2)
    name            = kwargs.get("name"           , "model")
    num_outputs     = kwargs.get("num_outputs"    )
    num_conditions  = kwargs.get("num_conditions" )
    num_observables = kwargs.get("num_observables")
    use_batch_norm  = kwargs.get("batch_norm"     , False)
    verbose         = kwargs.get("verbose"        , True)

    use_dropout = False
    if dropout > 0 : use_dropout = True

    data_layers      = kwargs.get("data_layers"     , (10,))
    condition_layers = kwargs.get("condition_layers", (10,))
    combined_layers  = kwargs.get("combined_layers" , (20, 20,))
    #
    #  Print stage
    #
    if verbose : print(f"Creating model with {num_observables} observables and {num_conditions} conditions")
    #
    #  Create input layers
    #
    data_input      = Input((num_observables,))
    condition_input = Input((num_conditions ,))
    #
    #  Create initially separate layers for the condition and data
    #
    model_data = data_input
    for layer_size in data_layers :
        model_data = Dense(layer_size)(model_data)
        model_data = LeakyReLU(leaky_relu)(model_data)
        if use_batch_norm : model_data = BatchNormalization()(model_data)
        if use_dropout    : model_data = Dropout(dropout)(model_data)

    model_condition = condition_input
    for layer_size in condition_layers :
        model_condition = Dense(layer_size)(model_condition)
        model_condition = LeakyReLU(leaky_relu)(model_condition)
        if use_batch_norm : model_condition = BatchNormalization()(model_condition)
        if use_dropout    : model_condition = Dropout(dropout)(model_condition)
    #
    #  Concatenate the condition and data latent states
    #
    model = Concatenate()([discriminator_data, discriminator_condition])
    #
    #  Create final model layers
    #
    for layer_size in combined_layers :
        model = Dense(layer_size)(model)
        model = LeakyReLU(leaky_relu)(model)
        if use_batch_norm : model = BatchNormalization()(model)
        if use_dropout    : model = Dropout(dropout)(model)
    #
    #  Compile model
    #
    model = Dense(num_outputs, activation="linear")(model)
    model = Model(name=name, inputs=[data_input, condition_input], outputs=[model])
    model.compile(loss="mse", optimizer=Adam())
    if verbose : model.summary()
    #
    #  return model
    #
    return model


#  Brief:  keras implementation of a network with one input
#
def create_simple_model (**kwargs) :
    #
    #  Parse settings
    #
    dropout         = kwargs.get("dropout"        , -1.)
    leaky_relu      = kwargs.get("leaky_relu"     , 0.2)
    sigmoid         = kwargs.get("sigmoid"        , False)
    name            = kwargs.get("name"           , "model")
    num_outputs     = kwargs.get("num_outputs"    )
    num_observables = kwargs.get("num_observables")
    use_batch_norm  = kwargs.get("batch_norm"     , False)
    verbose         = kwargs.get("verbose"        , True)

    use_dropout = False
    if dropout > 0 : use_dropout = True

    layers = kwargs.get("data_layers", (10,))
    #
    #  Print stage
    #
    if verbose : print(f"Creating model with {num_observables} observables")
    #
    #  Create input layers
    #
    data_input = Input((num_observables,))
    #
    #  Create initially separate layers for the condition and data
    #
    model = data_input
    for layer_size in layers :
        if sigmoid :
            model = Dense(layer_size, activation="sigmoid")(model)
        else :
            model = Dense(layer_size)(model)
            model = LeakyReLU(leaky_relu)(model)
        if use_batch_norm : model = BatchNormalization()(model)
        if use_dropout    : model = Dropout(dropout)(model)

    #
    #  Compile model
    #
    model = Dense(num_outputs, activation="linear")(model)
    model = Model(name=name, inputs=[data_input], outputs=[model])
    model.compile(loss="mse", optimizer=Adam())
    if verbose : model.summary()
    #
    #  return model
    #
    return model


#  Brief:  keras implementation of unconditional discriminator
#
def create_unconditional_discriminator (**kwargs) :
    #
    #  Parse settings
    #
    dropout         = kwargs.get("dropout"        , -1.)
    leaky_relu      = kwargs.get("leaky_relu"  , 0.2)
    num_categories  = kwargs.get("num_categories" )
    num_conditions  = kwargs.get("num_conditions" )
    num_observables = kwargs.get("num_observables")
    use_batch_norm  = kwargs.get("batch_norm"     , False)
    verbose         = kwargs.get("verbose"        , True)
    mid_layers      = kwargs.get("mid_layers"     , (10,))

    use_dropout = False
    if dropout > 0 : use_dropout = True
    #
    #  Print stage
    #
    if verbose : print(f"Creating discriminator with {num_observables} observables and {num_conditions} conditions")
    #
    #  Create input layers
    #
    data_input      = Input((num_observables,))
    #
    #  Create initially separate layers for the condition and data
    #
    discriminator = data_input
    for layer_size in mid_layers :
        discriminator = Dense(layer_size)(discriminator)
        discriminator = LeakyReLU(leaky_relu)(discriminator)
        if use_batch_norm : discriminator = BatchNormalization()(discriminator)
        if use_dropout    : discriminator = Dropout(dropout)(discriminator)
    #
    #  Compile discriminator model
    #
    discriminator = Dense(num_categories, activation="sigmoid")(discriminator)
    discriminator = Model(name="Discriminator", inputs=[data_input], outputs=[discriminator])
    if num_categories == 1 :
        discriminator.compile(loss="binary_crossentropy", optimizer=Adam())
    else :
        discriminator.compile(loss="categorical_crossentropy", optimizer=Adam())
    if verbose : discriminator.summary()
    #
    #  return discriminator
    #
    return discriminator


