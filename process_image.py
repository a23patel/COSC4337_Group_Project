import os
import sys
import numpy as np
import pandas as pd
import cv2 as cv
import tensorflow as tf
from PIL import Image
import pickle

TRAIN_DIR = './Data/train'
TEST_DIR = './Data/test'
VALID_DIR = './Data/valid'

IMAGE_X = 200
IMAGE_Y = 300

valid_classes = ['normal', 'squamous', 'adeno', 'large']

def parse_classification(classification_string):
    for i in range(len(valid_classes)):
        if (classification_string.find(valid_classes[i]) != -1):
            return i
    return np.nan

def clean_image(image_path, image_y = IMAGE_Y, image_x = IMAGE_X, grayscale=True):
    im = cv.imread(image_path)
    if grayscale:
        color_channel_processed_im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    else:
        color_channel_processed_im = im
    # TODO try different interpolations here
    res = cv.resize(color_channel_processed_im, (image_y, image_x), interpolation=cv.INTER_CUBIC)
    return (res / 255.0)

def pack_tensor(image_paths, image_y = IMAGE_Y, image_x = IMAGE_X, grayscale=True):
    n = image_paths.shape[0]
    image_tensors = []
    for i in range(n):
        img_data = clean_image(image_paths[i], image_y, image_x, grayscale)
        image_tensors.append(img_data)
    return tf.stack(image_tensors)

def generate_catalog():
    catalog = {}
    for dataset in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
        classes = os.listdir(dataset)
        filepaths = []
        formatts = []
        heights = []
        widths = []
        classifications = []
        cancerous = []
        for res_class in classes:
            class_dir = dataset + '/' + res_class
            classification = parse_classification(res_class)
            cancer_status = 1 if classification != 0 else 0
            samples = os.listdir(class_dir)
            for sample in samples:
                filepath = class_dir + '/' + sample
                filepaths.append(filepath)
                im = Image.open(filepath)
                formatt = im.format
                formatts.append(formatt)
                height, width = im.size
                im.close()
                heights.append(height)
                widths.append(width)
                classifications.append(classification)
                cancerous.append(cancer_status)
        filepaths = pd.Series(filepaths)
        formatts = pd.Series(formatts)
        heights = pd.Series(heights)
        widths = pd.Series(widths)
        classifications = pd.Series(classifications)
        cancerous = pd.Series(cancerous)
        shuffled_index = list(range(len(filepaths)))
        rng = np.random.default_rng(seed=42)
        rng.shuffle(shuffled_index)
        print(shuffled_index)
        catalog[dataset] = pd.DataFrame({'filepath': filepaths, 'height': heights, 'width': widths, 'format': formatts, 'class': classifications, 'is_cancer': cancerous}, index=shuffled_index).reset_index(drop=True)
    return catalog

def gen_y(ys):
    y_tensors = []
    for i in range(ys.shape[0]):
        res = np.array([0,0,0,0])
        res[ys[i]] = 1
        y_tensors.append(res)
    return tf.stack(y_tensors)

def gen_y_binary(ys):
    y_tensors = []
    for i in range(ys.shape[0]):
        res = np.array([0,0])
        res[ys[i]] = 1
        y_tensors.append(res)
    return tf.stack(y_tensors)  

def gen_pickle(image_y = IMAGE_Y, image_x = IMAGE_X, grayscale=True, **kwargs):
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]
    elif image_y != IMAGE_Y or image_x != IMAGE_X:
        prefix = str(image_y) + 'x' + str(image_x)
    else:
        prefix = ""
    if grayscale:
        color_depth = 1
    else:
        color_depth = 3
    
    catalog = generate_catalog()

    X_train = tf.reshape(pack_tensor(catalog[TRAIN_DIR]['filepath'], image_y, image_x, grayscale), [-1,image_x,image_y,color_depth])
    X_test = tf.reshape(pack_tensor(catalog[TEST_DIR]['filepath'], image_y, image_x, grayscale), [-1,image_x,image_y,color_depth])
    X_valid = tf.reshape(pack_tensor(catalog[VALID_DIR]['filepath'], image_y, image_x, grayscale), [-1,image_x,image_y,color_depth])
    y_train = gen_y(catalog[TRAIN_DIR]['class'])
    y_test = gen_y(catalog[TEST_DIR]['class'])
    y_valid = gen_y(catalog[VALID_DIR]['class'])

    #Flattened X data for SVM
    X_train_flat = tf.reshape(X_train, [-1, image_x*image_y*color_depth])
    X_valid_flat = tf.reshape(X_valid, [-1, image_x*image_y*color_depth])
    X_test_flat = tf.reshape(X_test, [-1, image_x*image_y*color_depth])

    # Binary labels for y
    y_train_binary = gen_y_binary(catalog[TRAIN_DIR]['is_cancer'])
    y_valid_binary = gen_y_binary(catalog[VALID_DIR]['is_cancer'])
    y_test_binary = gen_y_binary(catalog[TEST_DIR]['is_cancer'])

    #Combined train-valid data
    X_train_valid = tf.concat([X_train,X_valid],axis=0)
    y_train_valid = tf.concat([y_train,y_valid],axis=0)
    X_train_valid_flat = tf.reshape(X_train_valid, [-1, image_x*image_y*color_depth])
    y_train_valid_binary = tf.concat([y_train_binary, y_valid_binary],axis=0)


    print(f"The shape of X_train is {X_train.shape}")
    print(f"The shape of X_valid is {X_valid.shape}")
    print(f"The shape of X_test is {X_test.shape}")
    print(f"The shape of y_train is {y_train.shape}")
    print(f"The shape of y_valid is {y_valid.shape}")
    print(f"The shape of y_test is {y_test.shape}")

    print(f"The shape of X_train_flat is {X_train_flat.shape}")
    print(f"The shape of X_valid_flat is {X_valid_flat.shape}")
    print(f"The shape of X_test_flat is {X_test_flat.shape}")

    print(f"The shape of y_train_binary is {y_train_binary.shape}")
    print(f"The shape of y_valid_binary is {y_valid_binary.shape}")
    print(f"The shape of y_test_binary is {y_test_binary.shape}")

    print(f"The shape of X_train_valid is {X_train_valid.shape}")
    print(f"The shape of y_train_valid is {y_train_valid.shape}")
    print(f"The shape of X_train_valid_flat is {X_train_valid_flat.shape}")
    print(f"The shape of y_train_valid_binary is {y_train_valid_binary.shape}")

    with open('catalog.pickle', 'wb') as f:
        pickle.dump(catalog, f)

    with open(prefix+'x.pickle', 'wb') as f:
        pickle.dump((X_train, X_valid, X_test), f)

    with open('y.pickle', 'wb') as f:
        pickle.dump((y_train, y_valid, y_test), f)

    with open(prefix+'x_flat.pickle', 'wb') as f:
        pickle.dump((X_train_flat, X_valid_flat, X_test_flat), f)

    with open(prefix+'x_no_valid.pickle', 'wb') as f:
        pickle.dump((X_train_valid, X_test), f)

    with open(prefix+'x_flat_no_valid.pickle', 'wb') as f:
        pickle.dump((X_train_valid_flat, X_test_flat), f)

    with open('y_no_valid.pickle', 'wb') as f:
        pickle.dump((y_train_valid, y_test), f)

    with open('y_binary.pickle', 'wb') as f:
        pickle.dump((y_train_binary, y_valid_binary, y_test_binary), f)

    with open('y_binary_no_valid.pickle', 'wb') as f:
        pickle.dump((y_train_valid_binary, y_test_binary), f)

if __name__=='__main__':
    if (len(sys.argv) == 5):
        grayscale=True
        if int(sys.argv[4]) == 0:
            grayscale=False        
        gen_pickle(image_y=int(sys.argv[1]), image_x=int(sys.argv[2]), prefix=sys.argv[3], grayscale=grayscale)
    else:
        gen_pickle()