import streamlit as st

import io
import os
import yaml

from PIL import Image

from predict import load_model, get_prediction

st.title("Mask Classification Model with Image")

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


model = load_model()
model.eval()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))

    st.image(image, caption='Uploaded Image')
    st.write("Classifying...")
    _, y_hat = get_prediction(model, image_bytes)
    label = config['classes'][y_hat.item()]

    col1, col2, col3 = st.columns(3)
    col1.metric("Mask", label[0])
    col2.metric("Gender", label[1])
    col3.metric("Age", label[2])
