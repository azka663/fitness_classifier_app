import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown

# Automatically download the model if not present
model_path = "fitness_model_3class.h5"
if not os.path.exists(model_path):
    # Corrected URL format for gdown
    url = "https://drive.google.com/file/d/1KLcdAgZ7lUqI0HZpl89FbeBtUU-VWAsr/view?usp=share_link"
    gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# Define class labels
labels = ["Fit", "Overweight", "Underweight"]

# Page title
st.set_page_config(page_title="Body Fitness Image Classifier", layout="centered")
st.markdown("<h1 style='text-align: center;'> Body Fitness Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Upload a body image to classify as Fit, Overweight, or Underweight</h4>", unsafe_allow_html=True)
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]

    # Display result
    st.markdown("---")
    st.markdown(f"<h3 style='text-align: center;'>Prediction: <span style='color:#4CAF50;'>{labels[predicted_index]}</span></h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'>Confidence Score: {confidence:.2f}</h4>", unsafe_allow_html=True)
