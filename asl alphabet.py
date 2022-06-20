#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np
import os 
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.config.list_physical_devices()


# In[2]:


def datas(path):
    imgs=[]
    labs=[]
    imgs1=[]
    labs1=[]
    label=['A','B','C',"D",'del','E','F','J','H','I','J',"K",'L','M','N','nothing','O','P','Q',
           'R','S','space','T',
           'U','V','W','X','Y','Z']
    a=np.zeros((28),dtype=np.uint16)
    b=[]
    for i in range(len(label)):
        b=np.insert(a,i,1)
        new_dir=path+label[i]+"/*"
        for image in glob.glob(new_dir):
            image=cv2.imread(image)
            image=cv2.resize(image,(128,128))
            imgs.append(image)
            labs.append(b)
    imgs1=np.array(imgs,dtype=np.float16)/255
    labs1=np.array(labs)
    return imgs1,labs1
train_dir=r'../Datasets2021/fingerstest/train/asl_alphabet_train/'
test_dir=r'../Datasets2021/fingerstest/test/asl_alphabet_test/'


# In[3]:


x_train=[]
y_train=[]
x_train,y_train=datas(train_dir)
print(x_train.shape,y_train.shape)


# In[34]:


x_test=[]
y_test=[]
x_test,y_test=datas(test_dir)
print(x_test.shape,y_test.shape)


# In[5]:


model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3) ,activation='relu' ,input_shape=(128,128,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(128,(3,3),activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64,activation = 'relu'),
    keras.layers.BatchNormalization(),
    
    keras.layers.Dense(29,activation = 'softmax'),
])
optim=keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optim,loss='categorical_crossentropy',metrics=['accuracy'])
with tf.device("/GPU:0"):
    model1=model
    model1.fit(x_train,y_train,epochs=10)
model1.summary()


# In[ ]:


model.evaluate(x_test,y_test)


# In[14]:


model.save("asl_alifbesi.model",save_format='h5')


# In[62]:


label=['A','B','C',"D",'del','E','F','J','H','I','J',"K",'L',
       'M','N','nothing','O','P','Q','R','S','space','T',
           'U','V','W','X','Y','Z']
video = cv2.VideoCapture(0)
while video.isOpened():
    _,kadr=video.read()
    kop=np.copy(kadr)
    kop=cv2.resize(kop,(128,128))
    kop=np.expand_dims(kop,0)
    bashorat=model.predict(kop)
    text=label[np.argmax(bashorat)]
    cv2.putText(kadr,text,(50,250),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0))
    cv2.imshow('harf',kadr)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video.release()
cv2.destroyAllWindows()


# In[ ]:




