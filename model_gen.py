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

from random import shuffle
# include 
# 1. first track normal running
# 2. first track reverse running         
# 3. second track normal running        

samples = []
images = []
angles = []
correctedAngle = 0.15
# data collected by me on 1st track, reverse driving is included
with open('alldata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #next(reader,None)
    for line in reader:
        samples.append(line)
        
sample = samples[0]        
name = './alldata/IMG/'+sample[0].split('\\')[-1]
image = cv2.imread(name)
   
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
        
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
                if (angle!=0):
                    for i in range(3):
                        name = './alldata/IMG/'+batch_sample[i].split('\\')[-1]
                        #print(name) 
                        image = cv2.imread(name)                        
                        images.append(image)
                        gaussian_noise =random_noise(image,mode='gaussian',var=0.001)*255
                        images.append(gaussian_noise.astype('uint8') )                         
                        images.append(cv2.flip(image,1))                        
                                       
                    angles.append(angle)
                    angles.append(angle)
                    angles.append(angle * -1.0)
                    angles.append(angle+correctedAngle)     # left
                    angles.append(angle+correctedAngle)     # left
                    angles.append((angle+correctedAngle) * -1.0)
                    angles.append(angle-correctedAngle)     # right        
                    angles.append(angle-correctedAngle)     # right    
                    angles.append((angle-correctedAngle) * -1.0)                    

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)        

# compile and train the model using the generator function
Batch_size=32
train_generator = generator(train_samples, batch_size=Batch_size)
validation_generator = generator(validation_samples, batch_size=Batch_size)



# to show an image
height,width,channels = image.shape
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))      

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5,
        input_shape=( height, width,3)))
model.add(Cropping2D(cropping=((75,25),(0,0))))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(16, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')
# data augmentation

  

gc.collect()


# avoid overfitting/underfitting
