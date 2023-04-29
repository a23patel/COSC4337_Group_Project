from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_shape = (460,460,3)
N_CLASSES = 4
BATCH_SIZE = 32

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
checkpointer = ModelCheckpoint(filepath='./chest_CT_SCAN-ResNet50.hdf5',
                            monitor='val_loss', verbose = 1,
                            save_best_only=True)
early_stopping = EarlyStopping(verbose=1, patience=100)

train_path1 = '/home/michael/Projects/cosc4337/Project/Data/train'
valid_path1 = '/home/michael/Projects/cosc4337/Project/Data/valid'
test_path1 = '/home/michael/Projects/cosc4337/Project/Data/test'
train_datagen = ImageDataGenerator(dtype='float32')#, preprocessing_function=resnet.preprocess_input)
train_generator = train_datagen.flow_from_directory(train_path1,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (460,460),
                                                   class_mode = 'categorical')

valid_datagen = ImageDataGenerator(dtype='float32')#, preprocessing_function=resnet.preprocess_input)
valid_generator = valid_datagen.flow_from_directory(valid_path1,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (460,460),
                                                   class_mode = 'categorical')

test_datagen = ImageDataGenerator(dtype='float32')#, preprocessing_function=resnet.preprocess_input)
test_generator = test_datagen.flow_from_directory(test_path1,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (460,460),
                                                   class_mode = 'categorical')

from tensorflow.keras.applications import ResNet50,VGG16,ResNet101, VGG19, DenseNet201, EfficientNetB4, MobileNetV2
res_model = ResNet50(include_top=False, pooling='avg', weights='imagenet', input_shape = (image_shape))
for layer in res_model.layers:
    if 'conv5' not in layer.name:
        layer.trainable = False

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
model = Sequential()
model.add(res_model)
model.add(Dropout(0.4))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(N_CLASSES, activation='softmax'))
model.summary()

from tensorflow.keras import regularizers, optimizers
optimizer = optimizers.Adam(learning_rate= 0.001, decay= 1e-5)
model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['acc'])
TYPE='keras'
BATCH_SIZE=16
SAVED_MODEL_DIR='./basic_cnn_model/'
EPOCHS=10

history_res = model.fit(train_generator,
                    steps_per_epoch = 5,
                    epochs = 5,
                    verbose = 1,
                    validation_data = valid_generator,
                    callbacks = [checkpointer, early_stopping]
                    )

#model.save('saved_ models/kuma_model_1')

model.evaluate(test_generator)
