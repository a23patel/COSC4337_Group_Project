import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from load_tensors import load_data

TYPE='randomcv'

def create_model():
    rfc_params = {'n_estimators': [int(x) for x in np.linspace(start = 1, stop = 100, num = 100)], 'max_features': ['auto', 'sqrt'],
          'max_depth': [4,5,6, 7, 8], 'max_samples': [0.5,0.75,1.0] }

    rfc = RandomForestClassifier()

    rfc_model = RandomizedSearchCV(rfc, param_distributions=rfc_params, cv=3, refit=True, verbose=4)
    
    return rfc_model

def train(model_rf):
    (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_data(flat=True)
    X_train_and_valid = tf.concat([X_train,X_valid],axis=0).numpy()
    y_train_and_valid = tf.argmax(tf.concat([y_train,y_valid],axis=0),axis=1).numpy()
    X_test = X_test.numpy()
    y_test = tf.argmax(y_test, axis=1).numpy()
    model_rf.fit(X_train_and_valid,y_train_and_valid)
    print("\nThe best hyperparameters obtained are:", model_rf.best_params_)
    rfc_predictions = model_rf.predict(X_test)
    print("\n\n\t\t\tClassification Report")
    print(classification_report(y_test, rfc_predictions))
    confusion_matrix = tf.math.confusion_matrix(y_test, rfc_predictions)
    sns.heatmap(confusion_matrix, cmap='inferno', annot=True)  # heatmap of confusion matrix
    return confusion_matrix