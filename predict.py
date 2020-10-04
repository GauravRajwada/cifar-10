# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 05:29:19 2020

@author: Gaurav
"""
from tensorflow.keras.models import load_model
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

model=load_model('E:/AI Application Implementation/trained_model/Classification/Cifar-10/cifar-2.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# img = cv2.imread("00004_test.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (32, 32))
# img = img_to_array(img)
# img = np.expand_dims(img, axis=0)
# k = model.predict(img)[0]
# k=np.argmax(k)
# print(class_names[k])

arr = os.listdir()
result=[]
for i in arr:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    k = model.predict(img)[0]
    k=np.argmax(k)
    result.append(class_names[k])
    print(i)
    
    
dict={"filename":arr,'label':result}
import pandas as pd
df=pd.DataFrame(dict)
df.to_csv(r"E:\AI Application Implementation\trained_model\Classification\Cifar-10\sub.csv",index=False)

# df=pd.read_csv("E:/AI Application Implementation/trained_model/Classification/Cifar-10/sub.csv")
# df.to_csv(r"E:\AI Application Implementation\trained_model\Classification\Cifar-10\sub.csv",index=False)
