import streamlit as st
import torch
import torchvision
import yaml

from predict import get_prediction, load_model

model = load_model()
model.eval()

st.title("Mask Classification inference with Camera")
st.sidebar.success("Select a page above.")

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


img_file_buffer = st.camera_input("Take a picture")
if img_file_buffer is not None:

    bytes_data = img_file_buffer.getvalue()
    with st.spinner("Classifying..."):
        _, y_hat = get_prediction(model, bytes_data)

    label = config["classes"][y_hat.item()]

    col1, col2, col3 = st.columns(3)
    col1.metric("Mask", label[0])
    col2.metric("Gender", label[1])
    col3.metric("Age", label[2])
