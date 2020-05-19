#!/usr/bin/env python
# coding: utf-8


from keras.models import load_model, image
from keras.preprocessing.image import load_img, img_to_array
from numpy import array, expand_dims


#Loading Our Traied Model
model = load_model("face_rec_VGG_TL.h5")


# load an image from file
testing_image = "test_face.jpg"
image = load_img(testing_image, target_size=(224, 224))
#Showing the image
image.show(testing_image)

# Convert to array and make 4D
image = array(image)
image = expand_dims(image, axis=0)

#Decoding Predictions
if model.predict(image)[0][0] > 0.9:
    print("LABEL1")
if model.predict(image)[0][1] > 0.9:
    print("LABEL2")
if model.predict(image)[0][2] > 0.9:
    print("LABEL3")
if model.predict(image)[0][3] > 0.9:
    print("LABEL4")
    

