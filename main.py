import streamlit as st
from PIL import Image
import torch
import os
import warnings
warnings.filterwarnings("ignore")

def load_model():
    # Load the YOLOv5 model
    model_path = "/Users/kartikraktate/PycharmProjects/Mainafas/yolov5/runs/train/exp12"  # Update with your model path
    model = torch.hub.load('/Users/kartikraktate/PycharmProjects/Mainafas/yolov5', 'custom', path="/Users/kartikraktate/PycharmProjects/Mainafas/yolov5/runs/train/exp12/weights/best.pt",
                           source='local')
    return model

def predict(model, image):
    # Perform inference
    results = model(image)
    return results

# Streamlit App
st.title("SHM Through UAV'S")

# Load YOLOv5 model

model = load_model()


# File uploader for image
uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    st.write("Running prediction...")
    results = predict(model, image)

    # Display predictions
    st.write("Predictions:")
    st.image(results.render()[0], caption="Predicted Image", use_column_width=True)

    # Optional: Display detailed results
    st.write(results.pandas().xyxy[0])  # Display bounding box info as a DataFrame
