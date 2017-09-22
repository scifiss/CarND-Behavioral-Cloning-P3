# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:10:53 2017

@author: gaor
"""

# load data

import csv
import cv2
import numpy as np
from PIL import Image
from flask import Flask
from io import BytesIO

import tensorflow as tf
tf.python.control_flow_ops = tf
# Initial Setup for Keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D,Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc
from  skimage.util import random_noise
import sklearn
import keras

from random import shuffle

samples = []
angles = []
correctedAngle = 0.1
# data collected by me on 1st track
# include 
# 1. normal running in the center
# 2. reverse running in the center
# 3. recovery from left and right sides of the lane
# 4. two more runs on the brick bridge
with open('alldata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)  
    for line in reader:        
        angle = float(line[3])
        angles.append(angle)
n0, bins, patches = plt.hist(angles, 50)  
plt.show()                  
angles = []        
with open('alldata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #next(reader,None)   # use this if there is a header
    for line in reader:
        angle = float(line[3])
        if (angle!=0):    # to remove the big block of data with zero angles
            samples.append(line)         
            angles.append(angle)
         
n1, bins, patches = plt.hist(angles, 50)  
plt.show()          

sample = samples[0]        
name = './alldata/IMG/'+sample[0].split('\\')[-1]

image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
height,width,channels = image.shape

gaussian_noise =random_noise(image,mode='gaussian',var=0.001)*255
fig,axarr = plt.subplots(1,3, figsize=(15,25))
axarr[0].imshow(image)
axarr[1].imshow(gaussian_noise.astype('uint8') )
axarr[2].imshow(cv2.flip(image,1))
fig.savefig('examples\image_noised_flipped.png')  

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
Batch_size=128
def generator(samples, batch_size=Batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3]) 
                for i in range(3):
                    name = './alldata/IMG/'+batch_sample[i].split('\\')[-1]                                 
                    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)                       
                    images.append(image)  # orig image
                    # add Gaussian noises to original images
                    gaussian_noise =random_noise(image,mode='gaussian',var=0.001)*255
                    images.append(gaussian_noise.astype('uint8') ) # noised image                  
                    # flip images vertically
                    images.append(cv2.flip(image,1)) # flipped image
                                   
                angles.append(angle)  # orig center
                angles.append(angle)  # noised center
                angles.append(angle * -1.0)  # flipped center
                angles.append(angle+correctedAngle)     # orig left
                angles.append(angle+correctedAngle)     # noised left
                angles.append((angle+correctedAngle) * -1.0) # flipped left
                angles.append(angle-correctedAngle)     # orig right        
                angles.append(angle-correctedAngle)     # noised right    
                angles.append((angle-correctedAngle) * -1.0) # flipped right               
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)        

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=Batch_size)
validation_generator = generator(validation_samples, batch_size=Batch_size)

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5,
        input_shape=( height, width,3)))
# trim image to only see section with road
model.add(Cropping2D(cropping=((80,25),(0,0))))  # (None, 60, 320, 3)
#print(model.layers[-1].output_shape)
model.add(Convolution2D(6, 5, 5, activation="relu"))  # (None, 56, 316, 6)
model.add(MaxPooling2D())  # (None, 28, 158, 6)
model.add(Convolution2D(16, 5, 5, activation="relu")) #(None, 24, 154, 16)
model.add(MaxPooling2D()) # (None, 12, 77, 16)
model.add(Dropout(0.7)) # (None, 12, 77, 16)
model.add(Flatten()) # (None, 14784)
#model.add(Dense(560))
#model.add(Dropout(0.7))
model.add(Dense(120))
model.add(Dropout(0.7))
model.add(Dense(84))
model.add(Dense(10))
model.add(Dense(1))
opt = keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=opt)
#model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*9, 
                    validation_data=validation_generator, nb_val_samples=len(validation_samples)*9, nb_epoch=10)

model.save('model.h5')

gc.collect()
