import numpy as np
import os
from scipy.misc import imread, imsave, imresize, imshow
import random
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def get_folders(folder):
	folder_file = []
	for folders in os.listdir(folder):
		sub_folder = os.path.join(folder, folders)
		folder_file.append(sub_folder)
	return folder_file

def get_image_count(folder_file):
    total_images = 0
    for sub_folder in folder_file:
        total_images += len(os.listdir(sub_folder))
    return total_images

def process_class_folder(sub_folder):
    image_files = os.listdir(sub_folder)
    return image_files

def process_image(sub_folder, image, img_rows, img_cols):
    image_path = os.path.join(sub_folder, image)
    image_file = imread(image_path)
    image_resized = imresize(image_file, (img_rows, img_cols))
    image_transposed = np.asarray (image_resized.transpose(2,0,1), dtype = 'float32')
    image_data = image_transposed / 255
    return image_data

def process_train(folder, channels, img_rows, img_cols, val_size = None, verbose = 0):
    rand_state = 2016
    folder_file = get_folders(folder)
    nb_classes = len(folder_file)
    print 'Number of classes in training:', nb_classes
    total_images = get_image_count(folder_file)
    print 'Total no. of images in training:', total_images
    
    if val_size is None:
        train_items = list(xrange(total_images))
        random.shuffle(train_items)
        X_train = np.ndarray(shape = (total_images, channels, img_rows, img_cols), dtype = np.float32)
        y_train = np.zeros(total_images)

        image_count = 0
        for sub_folder in folder_file:
            class_no = int(sub_folder[-1])
            print 'Processing class:', class_no
            image_files = process_class_folder(sub_folder)
            for image in image_files:
                image_data = process_image(sub_folder, image, img_rows, img_cols)
                position = int(train_items.index(int(image_count)))
                X_train[position, :, :, :] = image_data
                y_train[position] = class_no
                image_count += 1
                if verbose > 0:
                    if (image_count + 1) % 5000 == 0:
                        print 'Processing training image,', image_count, 'of', total_images
        y_train = np_utils.to_categorical(y_train, nb_classes)

    else:
        train_items = list(xrange(total_images))
        random.shuffle(train_items)
        val_image_size = int(total_images * val_size)
        val_items = train_items[:val_image_size]
        train_items = train_items[val_image_size:]
        X_train = np.ndarray(shape = (len(train_items), channels, img_rows, img_cols), dtype = np.float32)
        X_val = np.ndarray(shape = (len(val_items), channels, img_rows, img_cols), dtype = np.float32)
        y_train = np.zeros(len(train_items))
        y_val = np.zeros(len(val_items))

        image_count = 0
        for sub_folder in folder_file:
            class_no = int(sub_folder[-1])
            print 'Processing class:', class_no
            image_files = process_class_folder(sub_folder)
            for image in image_files:
                image_data = process_image(sub_folder, image, img_rows, img_cols)
                if image_count in train_items:
                    position = int(train_items.index(int(image_count)))
                    X_train[position, :, :, :] = image_data
                    y_train[position] = class_no
                else:
                    position = int(val_items.index(int(image_count)))
                    X_val[position, :, :, :] = image_data
                    y_val[position] = class_no
                image_count += 1
                if verbose > 0:
                    if (image_count + 1) % 5000 == 0:
                        print 'Processing training image,', image_count , 'of', total_images
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_val = np_utils.to_categorical(y_val, nb_classes)

    if val_size is None:
        print 'Completed Processing'
        print 'Size of training set:', X_train.shape
        print 'Size of training labels:', y_train.shape
        return X_train, y_train
    else:
        print 'Completed Processing'
        print 'Size of training set:', X_train.shape
        print 'Size of training labels:', y_train.shape
        print 'Size of validation set:', X_val.shape
        print 'Size of validation labels:', y_val.shape
        return X_train, y_train, X_val, y_val

def display_sample(dataset, data_labels, index_value = None):
    if index_value is None:
        disp_nos = random.sample(xrange(dataset.shape[0]), 1)[0]
    else:
        disp_nos = index_value
    image_data = dataset[disp_nos,:,:,:]
    image_data = image_data.transpose(1,2,0)
    image_data *= 255
    image_data = np.array(image_data, dtype = 'uint8')
    current_label = list(data_labels[disp_nos, :]).index(1)
    print 'Image Classs:', current_label
    plt.imshow(image_data)

def predict_test(test_folder, model, img_rows, img_cols):
    image_files = os.listdir(test_folder)
    no_of_files = len(image_files)
    columns = ['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    index = xrange(0,no_of_files)
    submission = pd.DataFrame(index = index, columns = columns)
    for image_count, image in enumerate(image_files):
        image_path = os.path.join(test_folder, image)
        image_data = imread(image_path)
        image_resized = imresize(image_data, (img_rows, img_cols))
        image_arr = np.asarray (image_resized.transpose(2,0,1), dtype = 'float32')
        dataset = np.ndarray((1, 3, img_rows, img_cols), dtype = np.float32)
        dataset [0, :, :, :] = image_arr
        dataset /= 255
        pred = model.predict_proba(dataset, verbose = 0)
        submission['img'][image_count] = image
        submission['c0'][image_count] = pred[0][0]
        submission['c1'][image_count] = pred[0][1]
        submission['c2'][image_count] = pred[0][2]
        submission['c3'][image_count] = pred[0][3]
        submission['c4'][image_count] = pred[0][4]
        submission['c5'][image_count] = pred[0][5]
        submission['c6'][image_count] = pred[0][6]
        submission['c7'][image_count] = pred[0][7]
        submission['c8'][image_count] = pred[0][8]
        submission['c9'][image_count] = pred[0][9]
        if image_count % 5000 == 0:
            print 'Processing test image', image_count, 'of', no_of_files
    return submission

def create_network(channels, image_rows, image_cols, lr, decay, momentum, nb_classes):
    model = Sequential()
 
    model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape = (channels, image_rows, image_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    return model