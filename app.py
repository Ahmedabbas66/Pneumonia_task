import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import gdown
# Title
st.title("X-Ray Image Classification")

# File uploader
img_file = st.file_uploader("Upload your X-ray image", type=["jpg", "jpeg", "png"])
# Load pre-trained model
url = 'https://drive.google.com/uc?id=1S3_FR8xU3DarNMovRvIfoGsLc0BSgmd3'  # Replace with your model link
output = 'my_model.keras'  # Name to save the model locally

# Download the model using gdown
gdown.download(url, output, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(output)

# Prediction button
if img_file is not None:
    img = Image.open(img_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert image to grayscale if not already
    if len(np.array(img).shape) != 2:
        img = ImageOps.grayscale(img)
    
    # Predict button
    if st.button("Predict"):
        # Resize image
        img_array = np.array(img)

        # Expand dimensions if grayscale (2D)
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

        # Resize and normalize
        img_resized = tf.image.resize(img_array, (224, 224))
        img_normalized = img_resized / 255.0

        # Make prediction
        pred = model.predict(np.expand_dims(img_normalized, axis=0))

        # Display result
        st.text(f"The prediction is: {pred}")
else:
    st.warning("Please upload an image to proceed.")
