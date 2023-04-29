import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB2

TYPE='efficientnet'
BATCH_SIZE=16
EPOCHS=10
OPTIMIZER=Adam

def create_model():
    EfficientNetB2_model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    print(EfficientNetB2_model.summary())
    for layer in EfficientNetB2_model.layers:
        layer.trainable = False
    
    model = Sequential()
    model.add(EfficientNetB2_model)
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

    return EfficientNetB2_model, model