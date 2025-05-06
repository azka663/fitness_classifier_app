import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import requests

# Set the page configuration first
st.set_page_config(page_title="Body Fitness Image Classifier", layout="centered")

# Dropbox public URL for the model
model_filename = "fitness_model_3class.h5"
model_url = "https://www.dropbox.com/scl/fi/srglh38vvsekz7asxl3ur/fitness_model_3class.h5?rlkey=45n06xzy5au2q8hjw7ksv7pa9&st=11sd0sye&dl=1"  # Replace with your Dropbox link

# Check if model file exists, if not, download it
if not os.path.exists(model_filename):
    st.write("Model not found. Downloading...")
    try:
        # Download the model using requests
        r = requests.get(model_url, allow_redirects=True)
        with open(model_filename, 'wb') as f:
            f.write(r.content)
        st.write("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Error downloading the model: {e}")
        st.stop()

# Load the model
try:
    model = load_model(model_filename)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define class labels
labels = ["Fit", "Overweight", "Underweight"]

# Page title and instructions
st.markdown("<h1 style='text-align: center;'>Body Fitness Classifier</h1>", unsafe_allow_html=True)
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
    try:
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        confidence = prediction[predicted_index]

        # Display result
        st.markdown("---")
        st.markdown(f"<h3 style='text-align: center;'>Prediction: <span style='color:#4CAF50;'>{labels[predicted_index]}</span></h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>Confidence Score: {confidence:.2f}</h4>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        st.stop()
