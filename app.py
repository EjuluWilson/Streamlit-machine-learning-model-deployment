import streamlit as st
st.title("TOMATO DISEASE CLASSIFICATION")
"By Aroma Likambu"


import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# # Exract model from a zip file
# import zipfile
# zipped_model = zipfile.ZipFile("fyp_model_11.zip", "r")
# zipped_model.extractall()
# zipped_model.close()



#loading the "h5" format model
model= tf.keras.models.load_model('fyp_model_11.h5')

# defining class names in oder of the classes
class_names = ['bacterial_spot', 'early_blight', 'healthy']


# preprocess, predict and image plot function
# @st.cache
def preprocess_pred_and_plot(filename, model, class_names, img_shape=256):
    """
    Imports an image located at filename, reshapes it to img_shape, predicts
    on it using model and plots the image with the predicted class_name
    according to the class_names list as the title. img_shape should be the
    image size in which the model was trained.
    """
    ################### preprocessing the image ############
    
    # Read in target file (an image)
    img = tf.io.read_file(filename)

    # Decode the read file into a tensor & ensure 3 colour channels 
    # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
    img = tf.image.decode_image(img, channels=3)

    # Resize the image (to the same size our model was trained on)
    img = tf.image.resize(img, size = [img_shape, img_shape])

    # Rescale the image (get all values between 0 and 1)
    img = img/255.

    ################## Make a prediction ###############
 
    pred = model.predict(tf.expand_dims(img, axis=0))
    print(f'"pred", {pred}')
    print(f'"pred.argmax()", {pred.argmax()}')

    # Get the predicted class
    if len(pred[0]) > 1: # check for multi-class
        pred_class = class_names[pred.argmax()] # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

    print(pred_class)
    
    ################## Make a prediction ###############
    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


filename = f'trail images/early_blight_4.JPG'
preprocess_pred_and_plot(filename, model, class_names, img_shape=256)