import streamlit as st
import yaml
import io
from PIL import Image

from predict import load_model, get_image_prediction

model = load_model()
model.eval()

def get_inference(image):
    _, y_hat = get_image_prediction(model, image)
    label = config['classes'][y_hat.item()]

    col1, col2, col3 = st.columns(3)
    col1.metric("Mask", label[0])
    col2.metric("Gender", label[1])
    col3.metric("Age", label[2])

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

st.title("Mask Classification with Streamlit")
st.sidebar.success("Select a page above.")

st.text("Click 'camera' or 'image' on the left to try with your photo or webcam.")

image = Image.open('./1.jpg')
image_view = image.resize((350, 450))

# image center align
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image(image_view, caption="example Image")

with col3:
    st.write(' ')

col4, col5, col6, col7, col8 = st.columns(5)

with col4:
    st.write(' ')

with col5:
    st.write(' ')

with col6:
    inference_button = st.button(label="iamge inference")

with col7:
    st.write(' ')

with col8:
    st.write(' ')

if inference_button:
    get_inference(image)

