import importlib
import os
import sys
import load_tensors
import time
import pickle
import re
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import classification_report, confusion_matrix


#ignore_models = ['svm_model', 'rfc_model', 'basic_cnn', 'bigger_cnn_valid', 'basic_cnn_with_validation', 'dosbol_model', 'nadam_test', 'bigger_cnn_l2']
#ignore_models = ['rfc_model', 'basic_cnn', 'bigger_cnn_valid', 'basic_cnn_with_validation', 'dosbol_model', 'nadam_test', 'bigger_cnn_l2']
ignored_models = ['kuma_model_1', 'svm.best.pickle', 'models.svm_model_833178.pickle', 'models.rfc_model_786902.pickle']
saved_model_prefix ='./saved_models/'

def get_nonce():
    return str(int(time.time()) % 1000000)

def get_all_models():
    python_files = filter(lambda file: len(file.split('.')) > 1 and file.split('.')[1] == 'py', os.listdir('./models'))
    return ['models.' + python_file.split('.')[0] for python_file in python_files]

def train_all_models(model_list = None, **kwargs):
    nonce = get_nonce()
    if not model_list:
        model_list = get_all_models()
    else:
        model_list = ['models.' + model for model in model_list]
    if model_list == []:
        print(f"No models to train, exiting.", flush=True)
        sys.exit(0)
    try:
        models = [importlib.import_module(model) for model in model_list]
    except (ImportError):
        print(f"Error: tried to import nonexistent module!")
        sys.exit(1)
    print(f"Training models...", flush=True)
    results = {}
    print(models[0])
    for model in models:
        if model.__name__.split('.')[1] not in ignored_models:
            if model.TYPE == 'keras_tuner':
                results[model.__name__] = train_tuned_model(model, nonce=nonce, **kwargs)
            elif model.TYPE == 'keras':
                results[model.__name__] = train_untuned_model(model, nonce=nonce, **kwargs)
            elif model.TYPE == 'randomcv':
                results[model.__name__] = train_randomcv_model(model, nonce=nonce, **kwargs)
            elif model.TYPE == 'inception_v3':
                results[model.__name__] = train_inception_v3_model(model, nonce=nonce, **kwargs)
            elif model.TYPE == 'vgg16':
                results[model.__name__] = train_vgg16_model(model, nonce=nonce, **kwargs)
            elif model.TYPE == 'efficientnet':
                results[model.__name__] = train_efficientnet_model(model, nonce=nonce, **kwargs)
            else:
                print(f"Error: model type {model.TYPE} unsupported", flush=True)
        else:
            print(f"Ignoring model {model.__name__} that is in ignore list", flush=True)
    return results

def train_randomcv_model(model_module, nonce=None, binary=False, **kwargs):
    print("CHECKPOINT 4")    
    name = model_module.__name__
    print(name)
    if not nonce:
        nonce = get_nonce()
    (X_train, X_test), (y_train, y_test) = load_tensors.load_data(flat=True, valid=False, binary=binary)
    X_train = X_train.numpy()
    X_test = X_test.numpy()
    y_train = tf.argmax(y_train, axis=1).numpy()
    y_test = tf.argmax(y_test, axis=1).numpy()
    model = model_module.create_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    best_params = model.best_params_
    save_dir = saved_model_prefix+name+'_'+nonce+'.pickle'
    print(f"Writing best model for model {name} to {save_dir}...")
    with open(save_dir, 'wb') as f:
        pickle.dump(model, f)
    return (tf.convert_to_tensor(y_pred), best_params)

def train_untuned_model(model_module, nonce=None, binary=False, **kwargs):
    name = model_module.__name__
    print(name)
    if not nonce:
        nonce = get_nonce()
    (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_tensors.load_data(binary=binary)
    model = model_module.create_model()
    batch_size = model_module.BATCH_SIZE
    epochs = model_module.EPOCHS
    model.fit(X_train, y_train, 
            batch_size=batch_size, epochs=epochs, 
            verbose=2, validation_data=(X_valid, y_valid))
    save_dir = saved_model_prefix+name+'_'+nonce
    print(f"Writing best model for model {name} to {save_dir}...")
    model.save(save_dir)
    y_pred = tf.argmax(model.predict(X_test), axis=1)
    return (y_pred, None)
    
def train_tuned_model(model_module, max_trials=10, nonce=None, objective='val_accuracy', binary=False, **kwargs):
    name = model_module.__name__
    print(name)
    if not nonce:
        nonce = get_nonce()
    (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_tensors.load_data(binary=binary)
    tuner = kt.RandomSearch(
        model_module.create_model,
        objective=objective,
        max_trials=max_trials,
        directory=('tuner_run_'+nonce),
        project_name=name,
        overwrite=True)
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
    best_model = tuner.get_best_models()[0]
    save_dir = saved_model_prefix+name+'_'+nonce
    print(f"Writing best model for model {name} to {save_dir}...")
    best_model.save(save_dir)
    best_params = tuner.get_best_hyperparameters()[0]
    y_pred = tf.argmax(best_model.predict(X_test), axis=1)
    return (y_pred, best_params)

def train_inception_v3_model(model_module, nonce=None, **kwargs):
    name = model_module.__name__
    print(name)
    if not nonce:
        nonce = get_nonce()
    (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_tensors.load_data(prefix='Inception_')
    model = model_module.create_model()
    batch_size = model_module.BATCH_SIZE
    epochs = model_module.EPOCHS
    model.fit(X_train, y_train, 
            batch_size=batch_size, epochs=epochs, 
            verbose=2, validation_data=(X_valid, y_valid))
    save_dir = saved_model_prefix+name+'_'+nonce
    print(f"Writing best model for model {name} to {save_dir}...")
    model.save(save_dir)
    y_pred = tf.argmax(model.predict(X_test), axis=1)
    return (y_pred, None)

def train_vgg16_model(model_module, nonce=None, **kwargs):
    name = model_module.__name__
    print(name)
    if not nonce:
        nonce = get_nonce()
    (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_tensors.load_data(prefix='VGG_')
    base_model, model = model_module.create_model()
    batch_size = model_module.BATCH_SIZE
    epochs = model_module.EPOCHS
    optimizer = model_module.OPTIMIZER
    # Initial train: VGG-16 set to fully fixed
    model.fit(X_train, y_train, 
            batch_size=batch_size, epochs=epochs, 
            verbose=2, validation_data=(X_valid, y_valid))
    for layer in base_model.layers[:8]:
        layer.trainable = True
    model.compile(optimizer=optimizer(learning_rate=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy', F1Score(num_classes=4)]
        )
    model.fit(X_train, y_train, 
            batch_size=batch_size, epochs=12, 
            verbose=2, validation_data=(X_valid, y_valid))
    save_dir = saved_model_prefix+name+'_'+nonce
    print(f"Writing best model for model {name} to {save_dir}...")
    model.save(save_dir)
    y_pred = tf.argmax(model.predict(X_test), axis=1)
    return (y_pred, None)

def train_efficientnet_model(model_module, nonce=None, **kwargs):
    name = model_module.__name__
    print(name)
    if not nonce:
        nonce = get_nonce()
    (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_tensors.load_data(prefix='VGG_')
    base_model, model = model_module.create_model()
    batch_size = model_module.BATCH_SIZE
    epochs = model_module.EPOCHS
    optimizer = model_module.OPTIMIZER
    model.fit(X_train, y_train, 
            batch_size=batch_size, epochs=epochs, 
            verbose=2, validation_data=(X_valid, y_valid))
    for layer in base_model.layers[-6:]:
        layer.trainable = True
    model.compile(optimizer=optimizer(learning_rate=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy', F1Score(num_classes=4)]
        )
    model.fit(X_train, y_train, 
            batch_size=batch_size, epochs=30, 
            verbose=2, validation_data=(X_valid, y_valid))
    save_dir = saved_model_prefix+name+'_'+nonce
    print(f"Writing best model for model {name} to {save_dir}...")
    model.save(save_dir)
    y_pred = tf.argmax(model.predict(X_test), axis=1)
    return (y_pred, None)

def evaluate_all_models(model_list=None, **kwargs):
    if not model_list:
        saved_models = os.listdir(saved_model_prefix)
    else:
        saved_models = model_list
    for directory in saved_models:
        if re.search(r"inception", directory):
            prefix="Inception_"
        elif re.search(r"vgg", directory) or re.search(r"efficient", directory):
            prefix="VGG_"
        else:
            prefix=""
        (_,_,X_test), (_,_,y_test) = load_tensors.load_data(prefix=prefix)
        best_model = tf.keras.models.load_model('saved_models/' + directory,
                                             custom_objects={"metrics":F1Score})
        y_test_no_onehot = tf.argmax(y_test, axis=1)
        y_pred = tf.argmax(best_model.predict(X_test), axis=1)
        pickle_path = saved_model_prefix+'/'+directory+'.predictions.pickle'
        print(f"Writing {directory} to {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump(y_pred, f)
        print(f"Model {directory} classification report: ")
        print(classification_report(y_test_no_onehot.numpy(), y_pred.numpy()))
        print(f"Model {directory} confusion matrix: ")
        print(confusion_matrix(y_test_no_onehot.numpy(), y_pred.numpy()))

def main():
    model_names = None
    max_trials = 50
    if sys.argv[1] == 'train':
        if len(sys.argv) > 2:
            max_trials = int(sys.argv[2])
        if len(sys.argv) > 3:
            model_names = sys.argv[3:]
            print(f"Training on models {model_names}, with {max_trials} attempts on each")
        model_results = train_all_models(model_list=model_names, max_trials=max_trials)
        for model_name in model_results.keys():
            print(f"{model_name}: ")
            print(model_results[model_name][1])
    elif sys.argv[1] == 'evaluate':
        if len(sys.argv) > 2:
            model_names = sys.argv[2:]
        evaluate_all_models(model_list=model_names)
    else:
        print("Error: use 'train' or 'evaluate'")

if __name__=='__main__':
    main()
