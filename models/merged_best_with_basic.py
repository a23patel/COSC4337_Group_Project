from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from load_tensors import load_data

from models.basic_cnn import create_model as create_basic_cnn_model

TYPE='keras'
SAVED_MODEL='merged_best'
EPOCHS=30
BATCH_SIZE=16

def create_model():
    # conv_cycles = 2
    # conv_kernel_size = (3,2)
    # conv_filters = 8
    # poolsize = (2,2)
    # num_dense_neurons_1 = 64
    # num_dense_neurons_2 = 64
    # learning_rate = 0.01
    # image_shape= (200, 300, 1)
    # l2_scaling = 0.01
    # dropout_rate = 0.2
    # if binary:
    #     output_neurons = 2
    #     loss = BinaryCrossentropy
    conv_cycles = 3
    conv_kernel_size = (6,4)
    conv_filters = 16
    poolsize = (2,2)
    dense_layer_number = 2
    dense_layer_size = 32
    l2_scaling = 0.001
    dropout_rate = 0.5

    learning_rate = 0.01
    input_shape = (200, 300, 1)
    output_neurons = 4
    loss = CategoricalCrossentropy

    inputs = tf.keras.Input(input_shape)

    nadam_best = tf.keras.models.load_model('saved_models/models.tuner_nadam_746646',
                                             custom_objects={"metrics":F1Score})

    dosbol_best = tf.keras.models.load_model('saved_models/models.dosbol_model_tuner_749083/',
                                             custom_objects={"metrics":F1Score})

    basic_cnn = create_basic_cnn_model()

    output_basic = basic_cnn(inputs)

    nadam_best._name = "models.tuner_nadam_746646"
    dosbol_best._name = "models.dosbol_model_tuner_749083"

    for layer in nadam_best.layers:
        layer._name = layer.name + str("_nadam")
        layer.trainable = False

    for layer in dosbol_best.layers:
        layer._name = layer.name + str("_dosbol")
        layer.trainable = False
    
    output_nadam = nadam_best(inputs)
    output_dosbol = dosbol_best(inputs)
    
    print(nadam_best.summary())
    print(dosbol_best.summary())

    flat = Flatten()(tf.concat([output_basic, output_nadam, output_dosbol], axis=1))
    dense_1 = Dense(dense_layer_size)(flat)
    dense_2 = Dense(dense_layer_size)(dense_1)
    output = Dense(output_neurons, activation="softmax")(dense_2)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(optimizer=Nadam(learning_rate=learning_rate),
                loss=loss(),
                metrics=['accuracy', F1Score(num_classes=output_neurons)]
                )

    print(model.summary())
    return model