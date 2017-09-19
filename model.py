# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:10:53 2017

@author: gaor
"""

# load data
import os
import csv
import cv2
import numpy as np
import sklearn
import tensorflow as tf
tf.python.control_flow_ops = tf
# Initial Setup for Keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# include 
# 1. first track normal running
# 2. first track reverse running         
# 3. second track normal running        

samples = []
images = []
angles = []

# data collected by me on 1st track, reverse driving is included
with open('mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #next(reader,None)
    for line in reader:
        samples.append(line)
for line in samples:
    srcpath = line[0]
    filename = srcpath.split('\\')[-1]
    curpath = './mydata/IMG/' + filename
    image = cv2.imread(curpath)    
    images.append(image)
    angle = float(line[3])
    angles.append(angle)     

samples = []    
# data from the example on 1st track
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader,None)
    for line in reader:
        samples.append(line)
for line in samples:
    srcpath = line[0]
    filename = srcpath.split('/')[-1]
    curpath = './data/IMG/' + filename
    image = cv2.imread(curpath)    
    images.append(image)
    angle = float(line[3])
    angles.append(angle)  
        
# data collected for recovery to the center from the sides on 1st track   
samples = []     
with open('track1recovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)        
for line in samples:
    srcpath = line[0] 
    filename = srcpath.split('\\')[-1]
    curpath = './track1recovery/IMG/' + filename
    image = cv2.imread(curpath)    
    images.append(image)
    angle = float(line[3])
    angles.append(angle)             
# data collected on 2nd track
samples = []
with open('track2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)        
for line in samples:
    srcpath = line[0]
    filename = srcpath.split('\\')[-1]
    curpath = './track2/IMG/' + filename
    image = cv2.imread(curpath)    
    images.append(image)
    angle = float(line[3])
    angles.append(angle)     
        
        
X_train = np.array(images)
y_train =  np.array(angles)   
# to show an image
height,width,channels = image.shape
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))      

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5,
        input_shape=( height, width,3)))

model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=7)

model.save('model.h5')
# data augmentation

  



#train_samples, validation_samples = train_test_split(samples, test_size=0.2)
#
#
#
#
#
#
#def generator(samples, batch_size=32):
#    num_samples = len(samples)
#    while 1: # Loop forever so the generator never terminates
#        #shuffle(samples)
#        for offset in range(0, num_samples, batch_size):
#            batch_samples = samples[offset:offset+batch_size]
#
#            images = []
#            angles = []
#            for batch_sample in batch_samples:
#                name = './IMG/'+batch_sample[0].split('/')[-1]
#                center_image = cv2.imread(name)
#                center_angle = float(batch_sample[3])
#                images.append(center_image)
#                angles.append(center_angle)
#
#            # trim image to only see section with road
#            X_train = np.array(images)
#            y_train = np.array(angles)
#            yield sklearn.utils.shuffle(X_train, y_train)
#
## compile and train the model using the generator function
#train_generator = generator(train_samples, batch_size=32)
#validation_generator = generator(validation_samples, batch_size=32)
#
#ch, row, col = 3, height, width
##ch, row, col = 3, 80, 320  # Trimmed image format
#
#model = Sequential()
## Preprocess incoming data, centered around zero with small standard deviation 
#model.add(Lambda(lambda x: x/127.5 - 1.,
#        input_shape=(ch, row, col),
#        output_shape=(ch, row, col)))
#model.add(Flatten(input_shape=(ch, row, col)))
#model.add(Dense(1))
#
#
#model.compile(loss='mse', optimizer='adam')
#model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
#                    validation_data=validation_generator,  nb_val_samples=len(validation_samples), nb_epoch=3)
#
#model.save('model.h5')
# data augmentation
# flip images



# preprocessing
# normalization: add Lambda

# avoid overfitting/underfitting