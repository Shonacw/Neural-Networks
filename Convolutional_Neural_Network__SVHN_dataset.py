#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:56:48 2020

@author: ShonaCW

----------------  CNN FOR NUMBER DETECTION ON THE SVHN DATASET  ---------------
Breakdown of Process:
    Step 1 - Pre-process SVHN data.
    Step 2 - Augment data and build auxillary model to determine what learning
             rate to use in the full model. 
    Step 3 - Analyse the auxillary model's performance by plotting loss vs
             learning rate. Decide what learning rate to use.
    Step 4 - Build and train full model.
    Step 5 - Evaluate model on test data.
    Step 6 - Analyse performance using confusion matrix.
-------------------------------------------------------------------------------
NOTES REFERENCED IN CODE
Note A - Augumenting to increase the amount of data I can use to train the
         model. Method to reduce overfitting of model.
         Rotating, zooming, shearing, vertically translating. Not horizontally
         translating as the images contain 'distracting digits' either side of
         the digit of interest. 
Note B - Using an auxillary model to investigate the learning rate parameter.
Note C - Clearing the session removes all any nodes left over from model run 
         previously. This frees memory and prevents slowdown.
Note D - Using Batch Normalization as means of regularisation/ so each layer of 
         the network learns more 'independently'.
         Using the AMSGrad variant of the Adam optimiser, and using padding in 
         order to retain image information located at the image borders.
Note E - Using Early stopping to avoid over or under-fitting. It will stop the
         model training once no improvements have been seen over however many
         epochs are specified using the patience parameter.
Note F - Callback to save the "best" model so far to file 'best_cnn'. The file
         contents will be overwritten by each new better model.
"""
import numpy as np
import keras

from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

##Seed 
np.random.seed(1)

##Load the SVHN dataset
train = loadmat('train_32x32.mat')
test = loadmat('test_32x32.mat')

#%%
""" Step 1 """
##Extract image data/ convert to np array format
Train_Img = np.array(train['X'])
Test_Img = np.array(test['X'])

##Extract the labels (number contained in image)
Train_labels = train['y']
Test_labels = test['y']

## Check shape of imagedata
print(Train_Img.shape)
print(Test_Img.shape)

##Adjust the axis of the images
Train_Img = np.moveaxis(Train_Img, -1, 0)
Test_Img = np.moveaxis(Test_Img, -1, 0)

## Check shape of imagedata after adjustment
print(Train_Img.shape)
print(Test_Img.shape)

##Example of accessing the label of image number 13529
print('Label: ', Train_labels[13529])
#%%
Train_Img = Train_Img.astype('float64')
Test_Img = Test_Img.astype('float64')

##Normalise the images
Train_Img /= 255.0
Test_Img  /= 255.0

##One-hot encoding of label
lb = LabelBinarizer()
Train_labels = lb.fit_transform(Train_labels)
Test_labels = lb.fit_transform(Test_labels)

##Train/ Test splitting on the training images and labels
X_train, X_val, y_train, y_val = train_test_split(Train_Img, Train_labels, \
                                                  test_size=0.15, random_state=22)


#%%
""" Step 2 """

##Data Augmentation -                                           See Note A
datagen = ImageDataGenerator(rotation_range = 8,
                             zoom_range = [0.95, 1.05],
                             height_shift_range = 0.10,
                             shear_range = 0.15)
##Create auxillary Model -                                      See Note B
keras.backend.clear_session() ##Clear the session -             See Note C
Auxillary_Model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding = 'same' , activation = 'relu',
                           input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.4),    
    keras.layers.Dense(10,  activation = 'softmax') ])
    
##Set a callback to gradually increase the learning rate of the optimiser
lr_schedule = keras.callbacks.LearningRateScheduler(
                                         lambda epoch: 1e-4 * 10**(epoch / 10))

optimiser = keras.optimizers.Adam(lr = 1e-4, amsgrad = True)
Auxillary_Model.compile(optimizer = optimiser, loss='categorical_crossentropy',
                        metrics=['accuracy'])


history = Auxillary_Model.fit_generator(datagen.flowg(X_train, y_train, batch_size=128),
                              epochs = 30, validation_data = (X_val, y_val),
                              callbacks = [lr_schedule])

#%%
""" Step 3 """

##Plot learning rate vs. loss 
plt.semilogx(history.history['lr'], history.history['loss'])
plt.xlabel('Learning Rate')
plt.ylabel('Training Loss')
plt.show()

##Choosing a lr in the region where the loss is stable --->  lr = 0.001
#%%
""" Step 4 """

##Build true model (with chosen lr) -                            See Note D
keras.backend.clear_session()

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu',
                           input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.4),    
    keras.layers.Dense(10,  activation = 'softmax') ])


early_stopping = keras.callbacks.EarlyStopping(patience = 8)    #See Note E
model_checkpoint = keras.callbacks.ModelCheckpoint('best_cnn.h5', 
                   save_best_only = True)                       #See Note F

optimiser = keras.optimizers.Adam(lr = 1e-3, amsgrad = True) 
model.compile(optimizer = optimiser, loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 128),
                              epochs = 30, validation_data = (X_val, y_val),
                              callbacks = [early_stopping, model_checkpoint])

#%%
##Determine the train/ validation accuracies and losses
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

##Visualize epochs vs. train/ validation accuracies and losses
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Epochs vs. Training and Validation Accuracy')
    
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title('Epochs vs. Training and Validation Loss')

plt.show()

#%%
""" Step 5 """
test_loss, test_acc = model.evaluate(x = Test_Img, y = Test_labels, verbose=0)

print('Test accuracy: {:0.4f} \nTest loss: {:0.4f}'.
      format(test_acc, test_loss))

""" Step 6 """
##Extract label predictions and apply inverse transformation to the labels
y_pred = model.predict(X_train)

y_pred = lb.inverse_transform(y_pred, lb.classes_)
y_train = lb.inverse_transform(y_train, lb.classes_)  

##Plot the confusion matrix
matrix = confusion_matrix(y_train, y_pred, labels=lb.classes_)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(matrix, annot=True, cmap='Greens', fmt='d', ax=ax)
plt.title('Confusion Matrix for training dataset')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
























