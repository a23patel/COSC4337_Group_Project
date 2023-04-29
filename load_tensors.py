import os
import pickle
#import tensorflow as tf
import numpy as np
#import cv2 as cv
#import matplotlib.pyplot as plt


def load_catalog():
    with open('catalog.pickle', 'rb') as f:
        catalog = pickle.load(f)
    return catalog

def load_data(binary=False, flat=False, valid=True, prefix='', old=False):
    if flat and valid:
        x_file = prefix + 'x_flat.pickle'
    elif flat and not valid:
        x_file = prefix + 'x_flat_no_valid.pickle'
    elif not flat and not valid:
        x_file = prefix + 'x_no_valid.pickle'
    else:
        x_file = prefix + 'x.pickle'
    
    if binary and valid:
        y_file = 'y_binary.pickle'
    elif binary and not valid:
        y_file = 'y_binary_no_valid.pickle'
    elif not binary and not valid:
        y_file = 'y_no_valid.pickle'
    else:
        y_file = 'y.pickle'
    if old:
        y_file = 'old_' + y_file

    print(f"We are loading X from {x_file} and y from {y_file}")

    with open(x_file, 'rb') as f:
        X = pickle.load(f)

    with open(y_file, 'rb') as f:
        y = pickle.load(f)


    return (X,y)

def test_data():
    (X_train, X_valid, X_test), (y_train, y_valid, y_test) = load_data()
    print(f"The shape of X_train is {X_train.shape}")
    print(f"The shape of X_valid is {X_valid.shape}")
    print(f"The shape of X_test is {X_test.shape}")

    print(f"The shape of y_train is {y_train.shape}")
    print(f"The shape of y_valid is {y_valid.shape}")
    print(f"The shape of y_test is {y_test.shape}")
