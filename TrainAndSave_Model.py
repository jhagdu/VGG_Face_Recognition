#!/usr/bin/env python
# coding: utf-8

#Importing Modules
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
from keras.models import load_model


pre_model = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)

# Freeze Existing Layers
for layer in pre_model.layers:
  layer.trainable = False


#Adding Fully Connected Layers
x = Dense(units=512, activation='relu')(pre_model.output)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=128, activation='relu')(x)
x = Dense(units=64, activation='relu')(x)
x = Flatten()(x)
prediction = Dense(units=4, activation='softmax')(x)

model = Model(inputs=pre_model.input, outputs=prediction)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


trainDataGen = ImageDataGenerator(rescale = 1./255, zoom_range = 0.2, shear_range = 0.2, horizontal_flip = True)
testDataGen = ImageDataGenerator(rescale = 1./255)

trainingSet = trainDataGen.flow_from_directory('dataset/training_set/', target_size = (224, 224))

testSet = testDataGen.flow_from_directory('dataset/testing_set/', target_size = (224, 224))


r = model.fit_generator(steps_per_epoch=100, epochs=5, validation_steps=10,
                        generator=trainingSet,validation_data=testSet,)


model.save('face_rec_VGG_TL.h5')

trainingSet.class_indices
