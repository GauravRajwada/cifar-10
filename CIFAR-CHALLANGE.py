# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:31:46 2020

@author: Gaurav
"""
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten,Conv2D,MaxPool2D,AvgPool2D,Dropout
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt

train_path="E:/AI-Application-Implementation/trained_model/Classification/Cifar-10/data/train"
test_path="E:/AI-Application-Implementation/trained_model/Classification/Cifar-10/data/test"

folders=glob("E:/All Data Set/CIFAR10/train/*")

datagen=ImageDataGenerator(rotation_range=0.5,
                                 brightness_range=[0.2,0.5],
                                 zoom_range=[0.1,0.8],
                                 horizontal_flip=True,
                                 validation_split=0.2,
                                 rescale=1./255)

train=datagen.flow_from_directory(directory=train_path,
                                        target_size=(256,256),
                                        # color_mode="grayscale",
                                        shuffle=True,
                                        class_mode='categorical',
                                        subset='training')

test=datagen.flow_from_directory(directory=train_path,
                                        target_size=(256,256),
                                        # color_mode="grayscale",
                                        shuffle=True,
                                        class_mode='categorical',
                                        subset='validation')

model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(256,256,3),activation='relu'))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(AvgPool2D(pool_size=(6,6)))

model.add(Flatten())

model.add(Dense(units=64,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=10,activation='softmax'))


model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

history=model.fit(train,validation_data=test,epochs=5,steps_per_epoch=len(train),validation_steps=len(test))

print(history.history.keys())