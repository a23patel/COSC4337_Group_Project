import tensorflow as tf
import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16

TYPE='vgg16'
BATCH_SIZE=16
EPOCHS=10
OPTIMIZER=Adam

def create_model():
    #InceptionV3_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
    VGG16_model = VGG16(weights='imagenet', include_top=False)
    print(VGG16_model.summary())
    VGG16_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = (VGG16_model(inputs, training=False))
    x = GlobalAveragePooling2D()(x)
    #x = Flatten()(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(4, activation = 'softmax')(x)
    model = keras.Model(inputs, output)
    
    model.compile(optimizer=OPTIMIZER(learning_rate=0.001),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy', F1Score(num_classes=4)]
            )

    return VGG16_model, model
