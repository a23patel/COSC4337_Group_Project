from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import AUC
import tensorflow as tf


from tensorflow.keras.layers import Dropout #----->>>>> DOSBOL ADDED
import warnings #----->>>>> DOSBOL ADDED
warnings.filterwarnings("ignore") #----->>>>> DOSBOL ADDED

from load_tensors import load_data


# SAVED_MODEL_DIR='./basic_cnn_with_validation/'
SAVED_MODEL='dosbol_model_tuner'
EPOCHS=10
BATCH_SIZE=16

def create_model(hp):
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
    conv_cycles = hp.Choice('conv_cycles', [1,2,3,4])
    conv_kernel_size = hp.Choice('conv_kernel_size', [1, 2, 3])
    if conv_kernel_size > 1:
        conv_kernel_shape = (conv_kernel_size*3,conv_kernel_size*2)
        conv_cycles = max(2, conv_cycles)
    else:
        conv_kernel_shape = (3,2)
    conv_filters = hp.Choice('conv_filters', [4, 8, 16])
    poolsize_hp = hp.Choice('poolsize', [2])
    poolsize = (poolsize_hp, poolsize_hp)
    num_dense_neurons = hp.Choice('dense_layer_size', [64, 128, 256])
    #l2_scaling = hp.Choice('l2_scaling', [0.001, 0.01, 0.1])
    dropout_rate = hp.Choice('dropout_rate', [0.001, 0.2, 0.3, 0.5, 0.7])
    dense_layer_activation = hp.Choice('dense_layer_activation', ['relu', 'tanh','sigmoid'])

    image_shape = (200, 300, 1)
    learning_rate = 0.001
    image_shape= (200, 300, 1)

    model = Sequential()
    print(image_shape[1:], flush=True)
    model.add(Conv2D(conv_filters, conv_kernel_shape,
                    input_shape=image_shape,
                    activation="relu",
                    data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=poolsize))
    for i in range(conv_cycles-1):
        model.add(Conv2D(conv_filters, conv_kernel_shape,
                            activation="relu"))
        model.add(MaxPooling2D(pool_size=poolsize))
    model.add(Flatten())
    model.add(Dropout(dropout_rate)) #----->>>>> DOSBOL ADDED
    model.add(Dense(num_dense_neurons, activation=dense_layer_activation))
    model.add(Dense(4, activation="softmax"))


    model.compile(optimizer=Adam(learning_rate=learning_rate),
                loss=CategoricalCrossentropy(),
                metrics=['accuracy', AUC()])

    print(model.summary())
    return model

def train(model, epochs=EPOCHS, batch_size=BATCH_SIZE):
    (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_data()
    #X_train = tf.reshape(X_train, (613, 200, 300, 1))
    train_history = model.fit(X_train, y_train, 
        batch_size=batch_size, epochs=epochs, 
        verbose=2, validation_data=(X_valid, y_valid))
    #model.fit(X_train, y_train, epochs=EPOCHS, verbose=2)
    (test_loss, test_accuracy, test_auc) = model.evaluate(X_test, y_test, batch_size=16, verbose=0)
    print(f"EVALUATION on testing data")
    print(f"Loss is {test_loss} with accuracy {test_accuracy} and AUC {test_auc}")
    return (train_history, test_loss, test_accuracy, test_auc)
