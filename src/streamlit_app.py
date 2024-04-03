import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

st.title('Cat-Faces')

#load model, set cache to prevent reloading
@st.cache_resource()
def load_model():
    model=tf.keras.models.load_model('models/cat_model')
    return model


with st.spinner("Loading Model...."):
    model=load_model()



# model_path = r".\models\cat_model"
# new_model = tf.keras.models.load_model(model_path)



st.header("Cat-Faces - Generator")
st.button("Generate Random image")

img = tf.random.normal((1,10,10,1))
col1, col2 = st.columns(2,)
with col1:
    st.image(img.numpy()[0],
             caption="Random image",
             clamp=True,
             use_column_width=True
             )



with col2:
    with st.spinner("Generating Image"):
        gen_image = model.predict(tf.reshape(img, (1, 1, 1, 100)))

        img = (gen_image[0, :, :, :] * 127.5 + 127.5)/256.

    st.image(img,
             caption="Genrierte Katze",
             clamp=False,
             use_column_width=True)