import tensorflow as tf
import os, shutil
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
import tensorflow_datasets as tfds
import time
import pathlib
from tensorflow.keras.utils import Progbar

import matplotlib
from tensorflow import keras
from tensorflow.keras import layers

from src.generate import generate, save_images

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


import pandas as pd



def process(image):
    image = tf.cast((image)/256. ,tf.float32)
    return image

def load_catData(data_dir = "data/cats",batch_size = 128):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        label_mode=None,
        class_names=None,
        validation_split=None,
        seed=None,
        image_size=(64, 64),
        batch_size=batch_size)



    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.experimental.preprocessing.Rescaling(scale=1. / 127.5, offset=-1)
    normalized_ds = train_ds.map(lambda x: normalization_layer(x))

    return normalized_ds



def build_discriminator():
    """
    Function to build discriminator. Currently it is implemented as a fixed discriminator (Fixed architecture)
    :return:
    """
    inputs = keras.Input(shape=(64, 64, 3), name='input_layer')
    # Block 1: input is 64 x 64 x (3)
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same',
                      kernel_initializer=tf.keras.initializers.RandomNormal(
                          mean=0.0, stddev=0.02), use_bias=False, name='conv_1')(inputs)
    x = layers.LeakyReLU(0.2, name='leaky_relu_1')(x)

    # Block 2: input is 32 x 32 x (64)
    x = layers.Conv2D(64 * 2, kernel_size=4, strides=2, padding='same',
                      kernel_initializer=tf.keras.initializers.RandomNormal(
                          mean=0.0, stddev=0.02), use_bias=False, name='conv_2')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
    x = layers.LeakyReLU(0.2, name='leaky_relu_2')(x)

    # Block 3: input is 16 x 16 x (64*2)
    x = layers.Conv2D(64 * 4, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_3')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_2')(x)
    x = layers.LeakyReLU(0.2, name='leaky_relu_3')(x)

    # Block 4: input is 8 x 8 x (64*4)
    x = layers.Conv2D(64 * 8, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_4')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_3')(x)
    x = layers.LeakyReLU(0.2, name='leaky_relu_4')(x)

    # Block 5: input is 4 x 4 x (64*4)
    outputs = layers.Conv2D(1, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, activation='sigmoid', name='conv_5')(x)
    # Output: 1 x 1 x 1
    model = tf.keras.Model(inputs, outputs, name="Discriminator")
    return model

def build_generator(latent_size):
    inputs = keras.Input(shape=(1, 1, latent_size), name='input_layer')
    # Block 1:input is latent(100), going into a convolution
    x = layers.Conv2DTranspose(64 * 8, kernel_size=4, strides=4, padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(
                                   mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_1')(inputs)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
    x = layers.ReLU(name='relu_1')(x)

    # Block 2: input is 4 x 4 x (64 * 8)
    x = layers.Conv2DTranspose(64 * 4, kernel_size=4, strides=2, padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(
                                   mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_2')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_2')(x)
    x = layers.ReLU(name='relu_2')(x)

    # Block 3: input is 8 x 8 x (64 * 4)
    x = layers.Conv2DTranspose(64 * 2, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_3')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_3')(x)
    x = layers.ReLU(name='relu_3')(x)

    # Block 4: input is 16 x 16 x (64 * 2)
    x = layers.Conv2DTranspose(64 * 1, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_4')(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_4')(x)
    x = layers.ReLU(name='relu_4')(x)

    # Block 5: input is 32 x 32 x (64 * 1)
    outputs = layers.Conv2DTranspose(3, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02), use_bias=False, activation='tanh', name='conv_transpose_5')(x)
    # Output: output 64 x 64 x 3
    model = tf.keras.Model(inputs, outputs, name="Generator")
    return model








def show_catsDataSet(ds_in):
    plt.figure(figsize=(10, 10))
    for images in ds_in.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow((images[i].numpy()*127.5 + 127.5).astype(int))
            # plt.title(class_names[labels[i]])
            plt.axis("off")

    plt.tight_layout()
    plt.savefig("cats3x3.png")
    plt.show()
    return plt


binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(label, output):
    disc_loss = binary_cross_entropy(label, output)
    #print(total_loss)
    return disc_loss


def generator_loss(label, fake_output):
    gen_loss = binary_cross_entropy(label, fake_output)
    #print(gen_loss)
    return gen_loss

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# tf.function
def train_step(images):
    # Train Discriminator with real labels
    with tf.GradientTape() as disc_tape1:
        real_output = discriminator(images, training=True)
        real_targets = tf.ones_like(real_output)
        disc_loss1 = discriminator_loss(real_targets, real_output)

    # gradient calculation for discriminator for real labels
    gradients_of_disc1 = disc_tape1.gradient(disc_loss1, discriminator.trainable_variables)

    # parameters optimization for discriminator for real labels
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc1,
                                                discriminator.trainable_variables))

    # noise vector sampled from normal distribution
    noise = tf.random.normal([batch_size, 1, 1, latent_size])
    #Generate Fake images
    generated_images = generator(noise, training=True)
    # Train Discriminator with fake labels
    with tf.GradientTape() as disc_tape2:
        fake_output = discriminator(generated_images, training=True)
        fake_targets = tf.zeros_like(fake_output)
        disc_loss2 = discriminator_loss(fake_targets, fake_output)
    # gradient calculation for discriminator for fake labels
    gradients_of_disc2 = disc_tape2.gradient(disc_loss2, discriminator.trainable_variables)

    # parameters optimization for discriminator for fake labels
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc2,
                                                discriminator.trainable_variables))
    # Train Generator with real labels
    with tf.GradientTape() as gen_tape:
        # Generate Fake images
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        real_targets = tf.ones_like(fake_output)
        gen_loss = generator_loss(real_targets, fake_output)

        # gradient calculation for generator for real labels
    gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)

    # parameters optimization for generator for real labels
    generator_optimizer.apply_gradients(zip(gradients_of_gen,
                                            generator.trainable_variables))
    return gen_loss,disc_loss1,disc_loss2,real_output,fake_output


def train(dataset, epochs):
    #Define outputList
    l_out = []

    for epoch in range(epochs):

        print("Epoch",epoch+1, "of ",epochs)
        pb_i = Progbar(int(tf.data.experimental.cardinality(dataset)))
        _i = 1
        for image_batch in dataset:
            gen_loss,disc_loss,real_output,fake_output,_ = train_step(image_batch)
            pb_i.add(_i, values=[("Gen. Loss", gen_loss.numpy()),
                               ("Disc. Loss", disc_loss.numpy()),
                                 ("Real score:", tf.nn.sigmoid(tf.math.reduce_mean(real_output)).numpy()),
                                 ("Fake score:", tf.nn.sigmoid(tf.math.reduce_mean(fake_output)).numpy()),
                                 ])

        #collect infos
        l_out.append(dict({
            "epoch":epoch,
            "GenLoss":gen_loss.numpy(),
            "Real score":tf.nn.sigmoid(tf.math.reduce_mean(real_output)).numpy(),
            "DiscLoss":tf.nn.sigmoid(tf.math.reduce_mean(fake_output)).numpy(),

        }))



        # Produce images for the GIF as you go
        # display.clear_output(wait=True)
        gen_image = generate(generator, seed)
        save_images(gen_image,epoch = epoch+1)


        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

    generator.save("./../models/cat_model.pb")
    return l_out



if __name__ == '__main__':
    batch_size = 128
    latent_size = 100
    N_EPOCHS = 100

    cats_dir  = pathlib.Path(r'C:\Users\fs.GUNDP\Python\CATFACES-GAN\data\cats')
    #Load the cat data....
    cat_ds = load_catData(batch_size= batch_size,
                          data_dir=cats_dir)


    #show some of the cute cats
    show_catsDataSet(cat_ds)
    plt.show()

    #Define the optimizer
    learning_rate = 0.0002
    generator_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)

    # build discriminator
    discriminator = build_discriminator()

    # bild generator
    generator = build_generator(latent_size)

    noise = tf.random.normal([1, 1, 1, latent_size])

    gen_image = generator(noise)

    fig, ax = plt.subplots(1)
    ax.imshow((gen_image[0, :, :, :].numpy() * 127.5 + 127.5).astype(int))
    plt.show()


    #Build the discriminator
    d_pred = discriminator(gen_image)


    num_examples_to_generate = 9

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate,1,1, latent_size])

    checkpoint_dir = './../models'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)


    hist = train(cat_ds,epochs = N_EPOCHS)

    pd.DataFrame(hist).to_csv("TrainingHistory.csv")





