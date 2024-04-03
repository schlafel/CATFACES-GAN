import tensorflow as tf
from src.train import build_generator
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    latent_size = 100
    generator = build_generator(latent_size=latent_size)
    checkpoint_dir = r"./../models"
    model_path = r"C:\Users\fs.GUNDP\Python\CATFACES-GAN\models\cat_model"
    new_model = tf.keras.models.load_model(model_path)

    noise = tf.random.normal([1, 1, 1, latent_size])

    gen_image = new_model(noise,training=False)

    fig, ax = plt.subplots(1)
    ax.imshow((gen_image[0, :, :, :].numpy() * 127.5 + 127.5).astype(int))
    plt.show()


