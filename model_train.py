import importlib
import os
import sys
import load_tensors
import time
import pickle
import re
from load_tensors import load_data
import tensorflow as tf
import keras_tuner as kt
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D,LeakyReLU
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import EfficientNetB2, InceptionV3, VGG16, ResNet50
from tensorflow.keras.applications.xception import Xception

CONTENTPATH='./Data/'
TRAINPATH=CONTENTPATH+'train'
VALIDPATH=CONTENTPATH+'valid'
TESTPATH=CONTENTPATH+'train'
SAVED_MODEL='./new_saved_model/'

def get_nonce():
    return str(int(time.time()) % 1000000)

class BasicModel:
    def __init__(self, batch_size=16, epochs=8, imagenet=False, **kwargs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.imagenet = imagenet

    def train_static(self, prefix=None):
        assert(self.imagenet == False)
        if prefix:
            (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_data(prefix)
        else:
            (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_data()
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=2, validation_data=(X_valid, y_valid))
        return self.model.evaluate(X_test, y_test, batch_size=16, verbose=2)

    def train_flow(self):
        if self.imagenet:
            target_size=(224,224)
        else:
            target_size=(200,300)
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(TRAINPATH, target_size=target_size, batch_size=32, class_mode='categorical')
        valid_generator = train_datagen.flow_from_directory(VALIDPATH, target_size=target_size, batch_size=32, class_mode='categorical')
        test_generator = test_datagen.flow_from_directory(TESTPATH, target_size=target_size, batch_size=32, class_mode='categorical')
        history = self.model.fit(train_generator,
                    validation_data=valid_generator,
                    epochs=self.epochs,
                    steps_per_epoch=len(train_generator),
                    validation_steps=len(valid_generator))
        return self.model.evaluate(test_generator, verbose=2)
    
    def save(self):
        nonce = get_nonce()
        name = self.__class__.__name__
        save_dir = SAVED_MODEL+name+'_'+nonce
        print(f"Writing best model for model {name} to {save_dir}...")
        self.model.save(save_dir)
    
    def predictions(self):
        if (self.imagenet):
            prefix='VGG_'
        else:
            prefix=None
        (_,_,X_test), (_,_,y_test) = load_data(prefix=prefix)
        y_test_no_onehot = tf.argmax(y_test, axis=1)
        y_pred = tf.argmax(self.model.predict(X_test), axis=1)
        name = self.__class__.__name__
        nonce = get_nonce()       
        save_dir = SAVED_MODEL+name+'_'+nonce       
        pickle_path = save_dir+'.predictions.pickle'
        print(f"Writing {name} to {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump(y_pred, f)
        print(f"Model {name} classification report: ")
        print(classification_report(y_test_no_onehot.numpy(), y_pred.numpy()))
        print(f"Model {name} confusion matrix: ")
        print(confusion_matrix(y_test_no_onehot.numpy(), y_pred.numpy()))

class TunedModel(BasicModel):
    def __init__(self, max_trials=10, **kwargs):
        super().__init__(**kwargs)
        self.max_trials = max_trials

    def generate_model(hp):
        pass

    def train_static(self, prefix=None, max_trials=10, directory='tuner_run', project_name=None, overwrite=True):
        assert(self.imagenet == False)
        if prefix:
            (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_data(prefix)
        else:
            (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_data()
        self.tuner = kt.RandomSearch(
            self.generate_model,
        objective='val_accuracy',
        max_trials=max_trials,
        directory=directory,
        project_name=project_name,
        overwrite=True)
        self.tuner.search(X_train, y_train, epochs=self.epochs, validation_data=(X_valid, y_valid))
        self.model = self.tuner.get_best_models()[0]
        return self.model.evaluate(X_test, y_test, verbose=2)    
    
    def train_flow(self, directory='tuner_run', project_name=None, overwrite=True):
        if self.imagenet:
            target_size=(224,224)
        else:
            target_size=(200,300)
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(TRAINPATH, target_size=target_size, batch_size=32, class_mode='categorical')
        valid_generator = train_datagen.flow_from_directory(VALIDPATH, target_size=target_size, batch_size=32, class_mode='categorical')
        test_generator = test_datagen.flow_from_directory(TESTPATH, target_size=target_size, batch_size=32, class_mode='categorical')
        self.tuner = kt.RandomSearch(
            self.generate_model,
            objective='val_accuracy',
            max_trials=self.max_trials,
            directory=directory,
            project_name=project_name,
            overwrite=True)
        self.tuner.search(train_generator,
                    validation_data=valid_generator,
                    epochs=self.epochs,
                    steps_per_epoch=len(train_generator),
                    validation_steps=len(valid_generator))
        self.model = self.tuner.get_best_models()[0]
        return self.model.evaluate(test_generator, verbose=2)
        
class BasicCNN(BasicModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        conv_cycles = 2
        conv_kernel_size = (3,2)
        conv_filters = 8
        poolsize = (2,2)
        num_dense_neurons = 64
        learning_rate = 0.001
        if self.imagenet:
            image_shape= (224, 224, 3)
        else:
            image_shape= (200, 300, 1)

        model = Sequential()
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
                    metrics=['accuracy'])

        print(model.summary())
        self.model = model

class Dosbol(BasicModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        conv_cycles = 2
        conv_kernel_size = (3,2)
        conv_filters = 8
        poolsize = (2,2)
        num_dense_neurons = 64
        learning_rate = 0.001
        if self.imagenet:
            image_shape= (224, 224, 3)
        else:
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
        model.add(Dropout(0.5)) #----->>>>> DOSBOL ADDED
        model.add(Flatten())
        model.add(Dense(num_dense_neurons, activation="relu"))
        
        model.add(Dense(4, activation="softmax"))
        

        model.compile(optimizer=Adam(learning_rate=learning_rate),
                    loss=CategoricalCrossentropy(),
                    metrics=['accuracy'])

        print(model.summary())
        self.model = model

class KerasTuner(TunedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_model(self, hp):
        conv_cycles = hp.Choice('conv_cycles', [1,2,3])
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
        if self.imagenet:
            image_shape= (224, 224, 3)
        else:
            image_shape= (200, 300, 1)
        learning_rate = 0.01

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
            model.add(Dense(dense_layer_size, 
                activation="relu",
                kernel_regularizer=l2(l2_scaling),
                bias_regularizer=l2(l2_scaling)))
        model.add(Dense(output_neurons, activation="softmax"))

        model.compile(optimizer=Nadam(learning_rate=learning_rate),
                    loss=loss(),
                    metrics=['accuracy', F1Score(num_classes=output_neurons)]
                    )

        print(model.summary())
        return model

class KerasLeakyNadam(TunedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_model(self, hp):
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
        if self.imagenet:
            image_shape= (224, 224, 3)
        else:
            image_shape= (200, 300, 1)
        learning_rate = 0.01
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

class TransferModel(BasicModel):
    def __init__(self, trainable_layers=0, optimizer=Adam, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = optimizer
        self.transfer_model = None
        self.model = None
        self.trainable_layers = trainable_layers
        self.imagenet = True

    def train_static(self):
        (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_data(prefix='VGG_')
        # Initial train, fully frozen
        self.model.fit(X_train, y_train, 
            batch_size=self.batch_size, epochs=self.epochs, 
            verbose=2, validation_data=(X_valid, y_valid))
        # Unfreeze
        for layer in self.transfer_model.layers[self.trainable_layers]:
            layer.trainable = True
        self.model.compile(optimizer=self.optimizer(learning_rate=self.learning_rate/2.),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy', F1Score(num_classes=4)]
        )
        self.model.fit(X_train, y_train, 
            batch_size=self.batch_size, epochs=self.epochs+10, 
            verbose=2, validation_data=(X_valid, y_valid))
        return self.model.evaluate(X_test, y_test, verbose=2)

    def train_flow(self):
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(TRAINPATH, target_size=(224,224), batch_size=32, class_mode='categorical')
        valid_generator = train_datagen.flow_from_directory(VALIDPATH, target_size=(224,224), batch_size=32, class_mode='categorical')
        test_generator = test_datagen.flow_from_directory(TESTPATH, target_size=(224,224), batch_size=32, class_mode='categorical')
        # Initial train, fully frozen
        self.model.fit(train_generator, 
            batch_size=self.batch_size, epochs=self.epochs, 
            verbose=2, validation_data=valid_generator)
        # Unfreeze
        for layer in self.transfer_model.layers[-self.trainable_layers:]:
            layer.trainable = True
        self.model.compile(optimizer=self.optimizer(learning_rate=self.learning_rate/2.),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy', F1Score(num_classes=4)]
        )
        self.model.fit(train_generator, 
            batch_size=self.batch_size, epochs=self.epochs+10, 
            verbose=2, validation_data=valid_generator)
        return self.model.evaluate(test_generator, verbose=2)

class InceptionV3Model(TransferModel):
    def __init__(self):
        kwargs = {}
        kwargs['trainable_layers']=15
        kwargs['optimizer']=Adam
        super().__init__(**kwargs)
        self.learning_rate = 0.001

        self.transfer_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
        print(self.transfer_model.summary())
        for layer in self.transfer_model.layers:
            layer.trainable = False
        
        self.model = Sequential()
        self.model.add(self.transfer_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(512, activation = 'relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(512, activation = 'relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(4, activation = 'softmax'))

        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy', F1Score(num_classes=4)]
            )

class EfficientNetB2Model(TransferModel):
    def __init__(self):
        kwargs = {}
        kwargs['trainable_layers']=15
        kwargs['optimizer']=Adam
        super().__init__(**kwargs)    
        self.learning_rate = 0.001  
        self.transfer_model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(224,224,3))
        print(self.transfer_model.summary())
        for layer in self.transfer_model.layers:
            layer.trainable = False
        
        model = Sequential()
        model.add(self.transfer_model)
        model.add(GlobalAveragePooling2D())
        model.add(Flatten())
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(0.3))
        model.add(Dense(4, activation = 'softmax'))

        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                loss=CategoricalCrossentropy(),
                metrics=['accuracy', F1Score(num_classes=4)]
                )

        self.model = model

class VGG16Model(TransferModel):
    def __init__(self):
        kwargs = {}
        kwargs['trainable_layers']=8
        kwargs['optimizer']=Adam
        super().__init__(**kwargs)   
        self.transfer_model = VGG16(weights='imagenet', include_top=False)
        print(self.transfer_model.summary())
        self.transfer_model.trainable = False
        self.learning_rate=0.001

        inputs = keras.Input(shape=(224, 224, 3))
        x = (self.transfer_model(inputs, training=False))
        x = GlobalAveragePooling2D()(x)
        #x = Flatten()(x)
        x = Dense(512, activation = 'relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation = 'relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(4, activation = 'softmax')(x)
        self.model = keras.Model(inputs, output)
    
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy', F1Score(num_classes=4)]
            )

class ResNet50Model(TransferModel):
    def __init__(self):
        kwargs = {}
        kwargs['trainable_layers']=16
        kwargs['optimizer']=Adam
        super().__init__(**kwargs)   
        self.transfer_model = ResNet50(weights='imagenet', include_top=False)
        print(self.transfer_model.summary())
        self.transfer_model.trainable = False
        self.learning_rate=0.001

        inputs = keras.Input(shape=(224, 224, 3))
        x = (self.transfer_model(inputs, training=False))
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(512, activation = 'relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation = 'relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(4, activation = 'softmax')(x)
        self.model = keras.Model(inputs, output)
    
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy', F1Score(num_classes=4)]
            )

class XceptionModel(TransferModel):
    def __init__(self):
        kwargs = {}
        kwargs['trainable_layers']=15
        kwargs['optimizer']=Adam
        super().__init__(**kwargs)   
        self.transfer_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224,3))
        print(self.transfer_model.summary())
        self.transfer_model.trainable = False
        self.learning_rate=0.001

        inputs = keras.Input(shape=(224, 224, 3))
        x = (self.transfer_model(inputs, training=False))
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(512, activation = 'relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation = 'relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(4, activation = 'softmax')(x)
        self.model = keras.Model(inputs, output)
    
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy', F1Score(num_classes=4)]
            )    

def main():
    if sys.argv[1] == 'all':
        models = [
#            BasicCNN(imagenet=True),
#            Dosbol(imagenet=True),
#            KerasTuner(imagenet=True, max_trials=10),
#            KerasLeakyNadam(imagenet=True, max_trials=10),
#            InceptionV3Model(),
#            EfficientNetB2Model(),
#            VGG16Model(),
            ResNet50Model(),
#            XceptionModel()
        ]
        for model in models:
            res = model.train_flow()
            loss = res[0]
            accuracy = res[1]
            print(f"Model {model.__class__.__name__}: loss {loss}, accuracy: {accuracy}")
            model.save()
            model.predictions()

if __name__=='__main__':
    main()