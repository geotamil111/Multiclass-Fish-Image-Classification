import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json


st.set_page_config(
    page_title="Fish Image Classification",
    layout="centered"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_fish_model.h5")

model = load_model()


with open("class_names.json", "r") as f:
    class_dict = json.load(f)


class_names = list(class_dict.keys())


st.title("🐟 Fish Image Classification")
st.write("Upload a fish image to predict the category")


uploaded_file = st.file_uploader(
    "Choose a fish image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    
    predictions = model.predict(image_array)
    predicted_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100

    predicted_class = class_names[predicted_index]

    
    st.success(f"✅ Predicted Fish: **{predicted_class.upper()}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
