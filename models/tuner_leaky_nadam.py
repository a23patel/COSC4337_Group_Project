from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from load_tensors import load_data

EPOCHS=30
BATCH_SIZE=16

def create_model(hp):

    conv_cycles = hp.Choice('conv_cycles', [1,2])
    if hp.Boolean('conv_kernel_zoom'):
        conv_kernel_size = (6,4)
    else:
        conv_kernel_size = (3,2)
    conv_filters = hp.Choice('conv_filters', [4, 8, 16])
    poolsize_hp = hp.Choice('poolsize', [2])
    poolsize = (poolsize_hp, poolsize_hp)
    dense_layer_number = hp.Choice('dense_layer_number', [1,2])
    dense_layer_size = hp.Choice('dense_layer_size', [64, 128, 256])
    l2_scaling = hp.Choice('l2_scaling', [0.001, 0.01, 0.1])
    dropout_rate = hp.Choice('dropout_rate', [0.2, 0.3, 0.5, 0.7])
    optimizer_choice = hp.Choice('optimizer', ['adam', 'nadam'])

    learning_rate = 0.01
    image_shape = (200, 300, 1)
    output_neurons = 4
    loss = CategoricalCrossentropy

    model = Sequential()
    #print(image_shape[1:], flush=True)
    model.add(Conv2D(conv_filters, conv_kernel_size,
                    input_shape=image_shape,
                    activation="relu",
                    data_format="channels_last",
                    kernel_regularizer=l2(l2_scaling),
                    bias_regularizer=l2(l2_scaling)))
    model.add(MaxPooling2D(pool_size=poolsize))
    model.add(Dropout(dropout_rate))
    for i in range(conv_cycles-1):
        model.add(Conv2D(conv_filters, conv_kernel_size,
                            activation="relu",
                            kernel_regularizer=l2(l2_scaling),
                            bias_regularizer=l2(l2_scaling)))
        model.add(MaxPooling2D(pool_size=poolsize))
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    for i in range(dense_layer_number):
        if (hp.Boolean('leaky_dense')):
            model.add(Dense(dense_layer_size, 
                activation=None,
                kernel_regularizer=l2(l2_scaling),
                bias_regularizer=l2(l2_scaling)))
            model.add(LeakyReLU(0.3))
        else:
            model.add(Dense(dense_layer_size, 
                activation="relu",
                kernel_regularizer=l2(l2_scaling),
                bias_regularizer=l2(l2_scaling)))
    model.add(Dense(output_neurons, activation="softmax"))

    if optimizer_choice == 'nadam':
        optimizer = Nadam(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                loss=loss(),
                metrics=['accuracy', F1Score(num_classes=output_neurons)]
                )

    print(model.summary())
    return model

def train(model, epochs=EPOCHS, batch_size=BATCH_SIZE, binary=False, valid=True):
    if valid:
        (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_data(binary=binary,valid=valid, flat=False)
        print(X_train.shape)
        train_history = model.fit(X_train, y_train, 
            batch_size=batch_size, epochs=epochs, 
            verbose=2, validation_data=(X_valid, y_valid),
            callbacks=[EarlyStopping(monitor='loss', patience=3)])
    else:
        # TODO implement
        return
    y_pred = model.predict(X_test)
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, batch_size=16, verbose=0)
    print(f"EVALUATION on testing data")
    print(f"Loss is {test_loss} with accuracy {test_accuracy} and AUC {test_auc}")
    print(f"Saving model to ./f{SAVED_MODEL}/")
    model.save(SAVED_MODEL)
    return (train_history, test_loss, test_accuracy, test_auc, y_pred)
