import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import pandas as pd
import numpy as np
import string
import random
from random import randint
import warnings
import PIL
from PIL import Image
import streamlit as st
import requests

def main():

    # importing tensorflow model
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    tf.keras.backend.clear_session()

    pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                    include_top = False, 
                                    weights = 'imagenet')


    for layer in pre_trained_model.layers:
        layer.trainable = False
    
    # pre_trained_model.summary()

    # cut off at the layer named 'mixed7'
    last_layer = pre_trained_model.get_layer('mixed7')

    # know its output shape
    last_output = last_layer.output

    from tensorflow.keras.optimizers import RMSprop

    # Feed the last cut-off layer to our own layers
    # Flatten the output layer to 1 dimension
    x = tf.keras.layers.Flatten()(last_output)
    # Add a dropout rate of 0.4
    x = tf.keras.layers.Dropout(0.4)(x) 
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    # Add a dropout rate of 0.3
    x = tf.keras.layers.Dropout(0.3)(x)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    # Add a final dropout layer
    x = tf.keras.layers.Dropout(0.2)(x)                  
    # Add a final softmax layer for classification
    x = tf.keras.layers.Dense  (3, activation='softmax')(x)           

    model_inception = tf.keras.Model(pre_trained_model.input, x) 

    model_inception.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

    model_inception.load_weights('model_inception_weights.h5')

    st.sidebar.title('Navigation')
    pages = st.sidebar.radio("Pages", ("Home Page", "Image Classifier", "About the Project", "About the Author"))
    if pages == "Home Page":
        st.title('Welcome to the Apple Products Image Clasification Project')
        st.image('apple.jpg', width=650)
        st.markdown('This is a deployment page to house a deep learning model which has been trained to classify \
            three Apple products (Macbook, iPad, and iPhone) based on their images.')

    elif pages == "Image Classifier":
        st.title("Image Classifier Testing Page")
        st.markdown("Upload an image and the deployed deep learning model will classify it as either a Macbook, iPad, or iPhone.")
        st.markdown("Disclaimer: There are a lot of other Apple products such as AirPods, AppleWatch, or their full tower Mac, but in this project, \
            we'll focus on the three products: Macbook, iPad, and iPhone.")
        image_upload = st.file_uploader("Upload your image")
        st.write(image_upload)


if __name__ == '__main__':
    main()