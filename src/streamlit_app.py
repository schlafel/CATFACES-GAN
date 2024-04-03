import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

st.title('Cat-Faces')






model_path = r".\models\cat_model"
new_model = tf.keras.models.load_model(model_path)



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


gen_image = new_model.predict(tf.reshape(img,(1,1,1,100)))
with col2:
    st.image(gen_image[0],
             caption="Genrierte Katze",
             clamp=True,
             use_column_width=True)