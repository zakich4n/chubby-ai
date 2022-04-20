# -*- coding: utf-8 -*-
"""
Face Chubby Attribute NN 
Created on 02/03/2022 
@author: 
GOUIZI Zaki
DESCHILDRE Paul
EL KADAOUI Hicham
LEOPOLD Maxime
BLANQUI Julien-Tien
"""
##Imports
#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from tensorflow import keras
import os

"""##Data Spliting
Making Test and Training dir
"""
#%%
from os import path
if (not os.path.exists("/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Test")) :
    dirTrain="Training"
    path="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet"
    pathTrain = os.path.join(path, dirTrain)
    os.mkdir(pathTrain)
    dirTest="Test"
    pathTest = os.path.join(path, dirTest)
    os.mkdir(pathTest)

"""Making Presence_of_feature and Absence_of_feature folders"""
#%%
if (not os.path.exists("/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Training/Absence_of_feature")) :
    pathPres="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Test/Presence_of_feature"
    pathAbs="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Test/Absence_of_feature"

    os.mkdir(pathPres)
    os.mkdir(pathAbs)

    pathPres="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Training/Presence_of_feature"
    pathAbs="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Training/Absence_of_feature"

    os.mkdir(pathPres)
    os.mkdir(pathAbs)

#%%
if (not os.path.exists("/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Validation")) :
    path="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Validation"
    os.mkdir(path)
    
"""Image splitting"""
#%%
import shutil as sh
attributes_raw= pd.read_csv('list_attr_celeba.csv')
attributeID=attributes_raw.iloc[:,0].values
attributeChubby=attributes_raw.iloc[:,14].values
frames=[pd.DataFrame(attributeID), pd.DataFrame(attributeChubby)]
attribute=pd.concat(frames, axis=1, ignore_index=True)
#%%%
"""ONE HOT ENCODING"""
from sklearn.preprocessing import LabelEncoder
LabelEncoder=LabelEncoder()
attribute[:][1]=LabelEncoder.fit_transform(attribute[:][1])
"""#Image classification"""
#%%
train_data=attribute.iloc[0:162770,::].values
validation_data=attribute.iloc[162770:182637,::].values
test_data=attribute.iloc[182637:,::].values
#%%
import shutil
#%%

calc=800 #Balance image with features and images without (plus 800 without for good measure)
for index in train_data:
    origin="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/img_align_celeba/img_align_celeba/"+index[0]
    if(index[1]==1) :
        destination="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Training/Presence_of_feature/"+index[0]
        shutil.copy(origin,destination)
        calc=calc+1
#%%
for index in train_data:
    origin="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/img_align_celeba/img_align_celeba/"+index[0]
    if(index[1]==0 and calc>0) :
        destination="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Training/Absence_of_feature/"+index[0]
        shutil.copy(origin,destination)
        calc=calc-1
#%%
calc=800
for index in test_data:
    origin="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/img_align_celeba/img_align_celeba/"+index[0]
    if(index[1]==1) :
        destination="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Test/Presence_of_feature/"+index[0]
        shutil.copy(origin,destination)
        calc=calc+1

for index in test_data:
    origin="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/img_align_celeba/img_align_celeba/"+index[0]
    if(index[1]==0 and calc>0) :
        destination="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Test/Absence_of_feature/"+index[0]
        shutil.copy(origin,destination)
        calc=calc-1

for index in validation_data:
    origin="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/img_align_celeba/img_align_celeba/"+index[0]
    destination="/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Validation/"+index[0]
    shutil.copy(origin,destination)

#%%
"""BUILDING THE KERAS CNN"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten, Rescaling

classifier=Sequential()

#Step1- Convolution


classifier.add(Conv2D(32, 3, activation='relu', input_shape=(64,64,3)))
classifier.add(Rescaling(scale=1./255))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))

classifier.add(Conv2D(64 ,3 , activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))


# Step3- Flattening
classifier.add(Flatten())

#Step4 - Full Connection
classifier.add(Dense(units=128, activation='relu')) 
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1, activation='sigmoid') )#Output layer (Binary Classification so 1 output + sigmoid)
#%%
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%%
classifier.summary()
#%% #Image Preprocessing and Dataset splitting
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image_dataset_from_directory

train_datagen=ImageDataGenerator(shear_range=0.4,
                                 zoom_range=0.4,
                                 rotation_range=90,
                                 vertical_flip=True,
                                 horizontal_flip=True)


training_set=train_datagen.flow_from_directory(
    '/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Training',
    target_size=(64,64), #size of images
    batch_size=20,
    class_mode='binary'
)

test_set=image_dataset_from_directory(
    'Test',
    image_size=(64,64),
    labels='inferred',
    label_mode='binary',
    batch_size=20
    #class_mode='binary'
)

#%% #Model Training
classifier.fit(training_set, 
               steps_per_epoch=(19578/20),
               epochs=50,
               validation_data=test_set,
               validation_steps=(2916/20))
#%% # Save the model
classifier.save('Chubby_CelebA_model_1stGoodGen.h5')
# %% Confusion Matrix
from sklearn.metrics import classification_report
batch_size = 20
target_names = ['Chubby', 'Normal']
Y_pred = classifier.predict_generator(test_set, 2916/batch_size)
y_pred = np.where(Y_pred>0.5, 1, 0)
Y_result=map(int, test_data[0:2916,1])
y_result=list(Y_result)

from sklearn.metrics import confusion_matrix
print('Confusion Matrix')
cm = confusion_matrix( y_result, y_pred)
print(cm)

# %% Single image prediction
from keras.preprocessing import image
test_image=image.load_img('/Users/zakg/Documents/Cours/HEI4/S8/Intelligence artificielle/Projet/Test/Presence_of_feature/202090.jpg',
                          target_size=(64,64))
test_image=image.img_to_array(test_image).astype('float32')/255
test_image=np.expand_dims(test_image,axis=0)

result=classifier.predict(test_image)
print("Accuracy of test : "+str(result))
