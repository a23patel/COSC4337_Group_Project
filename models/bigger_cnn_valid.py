from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import AUC
import tensorflow as tf
from load_tensors import load_data

TYPE='keras'
SAVED_MODEL_DIR='./bigger_cnn_valid/'
EPOCHS=10
BATCH_SIZE=16

def create_model():
    conv_cycles = 3
    conv_kernel_size = (3,2)
    conv_filters = 16
    poolsize = (2,2)
    num_dense_neurons = 128
    learning_rate = 0.001
    image_shape= (200, 300, 1)

    model = Sequential()
    print(image_shape[1:], flush=True)
    model.add(Conv2D(conv_filters, conv_kernel_size,
                    input_shape=image_shape,
                    activation="relu",
                    data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=poolsize))
    for i in range(conv_cycles-1):
        model.add(Conv2D(conv_filters, conv_kernel_size,
                            activation="relu"))
        model.add(MaxPooling2D(pool_size=poolsize))
    model.add(Flatten())
    model.add(Dense(num_dense_neurons, activation="relu"))

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
