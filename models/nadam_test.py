from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import tensorflow as tf
from load_tensors import load_data

TYPE='keras'
SAVED_MODEL='nadam_test'
EPOCHS=30
BATCH_SIZE=16

def create_model(binary=False):
    conv_cycles = 2
    conv_kernel_size = (3,2)
    conv_filters = 8
    poolsize = (2,2)
    num_dense_neurons_1 = 64
    num_dense_neurons_2 = 64
    learning_rate = 0.01
    image_shape= (200, 300, 1)
    l2_scaling = 0.01
    dropout_rate = 0.2
    if binary:
        output_neurons = 2
        loss = BinaryCrossentropy
    else:
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
    model.add(Dense(num_dense_neurons_1, 
        activation="relu",
        kernel_regularizer=l2(l2_scaling),
        bias_regularizer=l2(l2_scaling)))
    model.add(Dense(num_dense_neurons_2, 
        activation="relu",
        kernel_regularizer=l2(l2_scaling),
        bias_regularizer=l2(l2_scaling)))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(output_neurons, activation="softmax"))

    model.compile(optimizer=Nadam(learning_rate=learning_rate),
                loss=loss(),
                metrics=['accuracy', AUC()]
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
