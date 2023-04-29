import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import InceptionV3

TYPE='inception_v3'
BATCH_SIZE=16
EPOCHS=16

def create_model():
    InceptionV3_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
    print(InceptionV3_model.summary())
    for layer in InceptionV3_model.layers:
        layer.trainable = False
    
    model = Sequential()
    model.add(InceptionV3_model)
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation = 'softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy', F1Score(num_classes=4)]
            )

    return model