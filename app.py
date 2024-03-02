import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from PIL import Image

def resize_image(image, target_size):
    # Resize the image
    resized_image = image.resize(target_size[:2])

    return resized_image

def main():
    st.title("Kidney disease classification")
    st.write("Upload images")

    file = st.file_uploader("Upload file", type=['jpg', 'png', 'jpeg'])

    if file:
        # Display the uploaded image
        st.image(file, caption='Uploaded Image.', use_column_width=True)

        # Resize the image to [224, 224, 3]
        target_size = (224, 224, 3)
        pil_image = Image.open(file)
        resized_image = resize_image(pil_image, target_size)

        # Display the resized image
        
        model = load_model("E:\ML Project\kidney_disease_detection\artifacts\prepare_base_model\base_model_updated.h5")

        predictions = model.predict(resize_image)

