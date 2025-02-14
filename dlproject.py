import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = r"C:\Users\nadip\Downloads\bird_species_model.h5"
model = load_model(MODEL_PATH)

# Manually paste class labels extracted from Colab
class_labels = ['AMERICAN GOLDFINCH', 'BARN OWL', 'CARMINE BEE-EATER', 'DOWNY WOODPECKER', 'EMPEROR PENGUIN', 'FLAMINGO']
print("✅ Class Labels Loaded:", class_labels)

# Streamlit UI
st.title("🦜 Bird Species Classification ")
st.write("Upload an image of a bird, and the model will predict its species.")

# File uploader
uploaded_file = st.file_uploader("Choose a bird image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess image
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    predicted_label = class_labels[class_index]

    # Display prediction
    st.success(f"Predicted Bird Species :  **{predicted_label}** 🦉🦆🐦")

