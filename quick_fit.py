import models.svm_model as model1
import models.nadam_test as model2
import models.tuner_nadam as model3
import models.dosbol_model_tuner as modelD
#import tensorflow as tf
from load_tensors import load_data
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import keras_tuner as kt
import pickle
import sys

def main():
    model_no = int(sys.argv[1])
    if model_no == 1:
        model = model1.create_model()
        report, cm = model1.train(model)
        with open('svm.best.pickle', 'wb') as f:
            pickle.dump(model, f)
    elif model_no == 2:
        model = model2.create_model()
        train_history, test_loss, test_accuracy, test_auc, y_pred = model2.train(model)
        _, (_, _, y_test) = load_data()
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.argmax(y_test, axis=1)
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
    elif model_no == 3:
        (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_data()
        tuner = kt.RandomSearch(
            model3.create_model,
            objective='val_accuracy',
            max_trials=10
        )
        tuner.search(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
        best_model = tuner.get_best_models()[0]
        best_model.save(model3.SAVED_MODEL)
        y_pred = tf.argmax(best_model.predict(X_test), axis=1)
        y_true = tf.argmax(y_test, axis=1)
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
    elif model_no == 4:
        (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_data()
        tuner = kt.RandomSearch(
            modelD.create_model,
            objective='val_accuracy',
            max_trials=10
        )
        tuner.search(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
        best_model = tuner.get_best_models()[0]
        best_model.save(modelD.SAVED_MODEL)
        y_pred = tf.argmax(best_model.predict(X_test), axis=1)
        y_true = tf.argmax(y_test, axis=1)
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
    print(report)
    print(cm)
    return 0

if __name__=='__main__':
    main()