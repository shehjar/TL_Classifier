# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 00:33:34 2017

@author: kauls
"""

import os, random, csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions import translation_augmentation, rotate_bound, brightness_augmentation

curdir = os.getcwd()
pardir = os.path.abspath(os.path.join(curdir, os.pardir))
datadir = os.path.join(pardir,'pictures')
datafile = os.path.join(datadir, 'data.csv')

# read csv data
data=[]
with open(datafile, newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in datareader:
        data.append(row)

def read_data(filePath):
    # read csv data
    data=[]
    datafile = os.path.join(filePath, 'data.csv')
    with open(datafile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in datareader:
            data.append(row)
    return data
## check rotate function
#img_file = data[0][0]
#img_label = data[0][1]
##loading image
#image = cv2.imread(img_file)
#RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
##rotate image
#rotated_img = rotate_bound(RGB_image, -45)
#brightness_aug_img = brightness_augmentation(image, 'uniform')
#RGB_bright = cv2.cvtColor(brightness_aug_img, cv2.COLOR_BGR2RGB)
#plt.figure(figsize=(10,5))
##plt.imshow(RGB_image)
##print(img_label)
#img_samples = [RGB_image, rotated_img, RGB_bright]
#for i in range(3):
#    plt.subplot(3,1,i+1)
#    plt.imshow(img_samples[i])
#    plt.axis('off')
#    plt.text(0,-2, img_label)

def normalizing(img):
    img = img.astype("float")
    img = img/255 - 0.5
    return img

def image_augmentation(image, translate_limit, rotation_limit):
    #translate
    shape = image.shape[:2]
    if np.random.random() >= 0.5:
        image = translation_augmentation(image, translate_limit)
    if np.random.random() >= 0.5:
        angle = np.random.uniform(rotation_limit[0], rotation_limit[1])
        image = rotate_bound(image, angle)
    if np.random.random() >= 0.5:
        image = brightness_augmentation(image)
    return cv2.resize(image,dsize= shape,interpolation=cv2.INTER_AREA)

def equalize_distribution(data):
    n_total = len(data)
    np_data = np.array(data)
    n_red = sum(np_data[:,1] == 'red')
    n_green = n_total - n_red
    data_red = np_data[np_data[:,1]=='red',:]
    data_green = np_data[np_data[:,1]=='green',:]
    if n_green < n_red :
        n_choice = np.random.choice(n_red,n_green)
        new_red = data_red[n_choice,:]
        out_np_data = np.concatenate((data_green, new_red))
    elif n_red < n_green:
        n_choice = np.random.choice(n_green,n_red)
        new_green = data_green[n_choice,:]
        out_np_data = np.concatenate((new_green, data_red))
    else:
        out_np_data = np_data
    np.random.shuffle(out_np_data)
    return out_np_data.tolist()

#def add_data(data, n):
#    n_data = len(data)
#    additional_data = []
#    for i in range(n):
#        index=np.random.randint(n_data)

def test_train_split (data, test_size):
    n_data = len(data)
    n_test = int(test_size*n_data)
    #n_train = n_data - n_test
    random.shuffle(data)
    test_data = data[:n_test]
    train_data = data[n_test:]
    return (train_data, test_data)

def gen_test_data (data, image_shape):
    images = []
    labels = []
    for image_file, label in data:
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        resize_img = cv2.resize(image, dsize=image_shape, interpolation=cv2.INTER_AREA)
        image = normalizing(resize_img)
        images.append(image)
        if label == 'red':
            labels.append(0)
        else:
            labels.append(1)
    return np.array(images), np.array(labels)

def gen_batch_function(data, image_shape):
    """
    Generate function to create batches of training data
    :param data: List of path of images
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        #random.shuffle(data)
        new_data = equalize_distribution(data)
        for batch_i in range(0, 10*batch_size, batch_size):
            images = []
            labels = []
            while(len(images) < batch_size):
            #for image_file, image_label in new_data[batch_i:batch_i+batch_size]:
                img_index = np.random.randint(len(new_data))
                image_file = new_data[img_index][0]
                image_label = new_data[img_index][1]
                image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
                resize_img = cv2.resize(image,dsize=image_shape,interpolation= cv2.INTER_AREA)
                # apply image augmentation
                translate_limit = [-10, 10]
                rotate_limit = [-90, 90]
                image = image_augmentation(resize_img, translate_limit,rotate_limit)
                #normalizing image
                image = normalizing(image)
                if image_label == 'red':
                    labels.append(0)
                else:
                    labels.append(1)
                images.append(image)
                #gt_images.append(gt_image)

            yield np.array(images), np.array(labels)
    return get_batches_fn