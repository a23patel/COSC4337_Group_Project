import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from load_tensors import load_data

TYPE='randomcv'
EPOCHS=10
BATCH_SIZE=16

def create_model():
    svc_params = {'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}

    svc = svm.SVC(probability=True)

    svm_model = RandomizedSearchCV(svc, param_distributions=svc_params, cv=3, refit=True, verbose=4, n_iter=10)
    
    return svm_model

def train(model):
    #(X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_data(flat=True, valid=False)
    (X_train, X_test), (y_train, y_test) = load_data(flat=True, valid=False)
    X_train = X_train.numpy()
    X_test = X_test.numpy()
    y_train = tf.argmax(y_train, axis=1).numpy()
    y_test = tf.argmax(y_test, axis=1).numpy()
    model.fit(X_train,y_train)
    print("The best hyperparameters obtained are:", model.best_params_)
    svm_predictions = model.predict(X_test)
    print('\t\tClassification Report')
    report =  classification_report(y_test, svm_predictions)
    print(report)
    confusion_matrix = tf.math.confusion_matrix(y_test, svm_predictions)
    sns.heatmap(confusion_matrix, cmap='inferno', annot=True)  # heatmap of confusion matrix
    return (report, confusion_matrix)
    

