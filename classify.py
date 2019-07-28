# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras as K
from PIL import Image
import cv2
from keras.layers.normalization import BatchNormalization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

X_train = [];
Y_train = [];

X_test = [];
Y_test = [];

Uninfected = "../input/cell_images/cell_images/Uninfected/";
Parasitized = "../input/cell_images/cell_images/Parasitized/";
dims = (50,50);
num = 0;
cap = 13000;
m = int(0.8*cap);
k = cap - m;
pos = np.array([1,0]);
neg = np.array([0,1]);

#preprocessing of images

for f in os.listdir(Parasitized):
    if f.endswith('png'):
        num += 1;
        image = cv2.imread(Parasitized+f)
        img = Image.fromarray(image,'RGB');
        t1 = img.resize(dims,Image.ANTIALIAS);
        r45 = t1.rotate(45)
        r75 = t1.rotate(75)

        if num <= int(0.8 * cap):
            X_train.append(np.array(t1));
            X_train.append(np.array(r45));
            X_train.append(np.array(r75));
            Y_train.append(pos);
            Y_train.append(pos);
            Y_train.append(pos);
        else:
            X_test.append(np.array(t1));
            X_test.append(np.array(r45));
            X_test.append(np.array(r75));
            Y_test.append(pos);
            Y_test.append(pos);
            Y_test.append(pos);
    if num == cap:
        break;
    print(num);
        
num = 0;

for f in os.listdir(Uninfected):
    if f.endswith('png'):
        num += 1;
        image = cv2.imread(Uninfected+f)
        img = Image.fromarray(image,'RGB');
        t1 = img.resize(dims,Image.ANTIALIAS);
        r45 = t1.rotate(45)
        r75 = t1.rotate(75)

        if num <= int(0.8 * cap):
            X_train.append(np.array(t1));
            X_train.append(np.array(r45));
            X_train.append(np.array(r75));
            Y_train.append(neg);
            Y_train.append(neg);
            Y_train.append(neg);
        else:
            X_test.append(np.array(t1));
            X_test.append(np.array(r45));
            X_test.append(np.array(r75));
            Y_test.append(neg);
            Y_test.append(neg);
            Y_test.append(neg);
    if num == cap:
        break;
    print(num);

X_train = np.array(X_train);
Y_train = np.array(Y_train);
Y_train = Y_train.reshape(6*m,2) ;
X_test = np.array(X_test);
Y_test = np.array(Y_test);
Y_test = Y_test.reshape(6*k,2);

#check data dimensions

print(X_test.shape);
print(Y_test.shape);
print(X_train.shape);
print(Y_train.shape);

f = 10;
ks = (3,3);
s = (2,2);

#generate a keras CNN model

model = K.models.Sequential();

model.add(K.layers.Conv2D(32,(3,3),strides=(1,1),padding='valid',data_format='channels_last',input_shape=(50,50,3),activation='relu'));
model.add(K.layers.MaxPooling2D(2,2));
model.add(BatchNormalization(axis = -1));
model.add(K.layers.Dropout(0.2));

model.add(K.layers.Conv2D(32,(3,3),strides=(1,1),padding='valid',data_format='channels_last',activation='relu'));
model.add(K.layers.MaxPooling2D(2,2));
model.add(BatchNormalization(axis = -1));
model.add(K.layers.Dropout(0.2));

model.add(K.layers.Conv2D(32,(3,3),strides=(1,1),padding='valid',data_format='channels_last',activation='relu'));
model.add(K.layers.MaxPooling2D(2,2));
model.add(BatchNormalization(axis = -1));
model.add(K.layers.Dropout(0.2));

model.add(K.layers.Flatten());

model.add(K.layers.Dense(512, activation = 'relu'));
model.add(BatchNormalization(axis = -1));
model.add(K.layers.Dropout(0.5));
model.add(K.layers.Dense(2,activation='softmax'));

#Compile and Run

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']);
model.fit(x=X_train, y=Y_train, batch_size=64, epochs=20);
predict = model.evaluate(X_test,Y_test);
print("LOSS:{}, ACCURACY: {}".format(predict[0],predict[1]));

"""
Program Summary:
1. Open images of positive/negative images
2. Resize image to 50x50 and convert to numpy array
3. Construct CNN Model: Image -> (Conv -> Pool) x 3 -> Dense -> softmax === OUT
4. Compile, Train, Test Model
5. Output relevant metrics
"""
