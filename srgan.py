from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
import cv2
import os
import numpy as np
from glob import glob

from tensorflow.keras import layers

def tensorf(x):
    return tf.image.resize(x, (256, 256))

def build_generator():
    inputs = tf.keras.Input(shape=(None, None, 3))  # Example input shape

    # Example layers, adjust according to your model's architecture
    x = layers.Conv2D(64, kernel_size=3, padding="same")(inputs)  # First convolutional layer
    x = layers.ReLU()(x)
    
    # Add more layers as needed
    
    # Ensure the final output has 3 channels (RGB)
    outputs = layers.Conv2D(3, kernel_size=3, padding="same", activation="tanh")(x)

    # Resize to the target output size
    # outputs = layers.Lambda(lambda x: tf.image.resize(x, (256, 256)))(outputs)

    # outputs = layers.Lambda(
    #     # lambda x: tf.image.resize(x, (256, 256)),
    #     lambda x: tensorf(x),
    #     output_shape=(256, 256, 3)  # Specify the output shape explicitly
    # )(outputs)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Discriminator Model
def build_discriminator(input_shape=(None, None, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), strides=2, padding='same')(inputs)
    x = LeakyReLU()(x)
    for _ in range(3):
        x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs, x)

# Load and preprocess images with resizing
def load_and_preprocess_image(lr_path, hr_path, target_size=(256, 256)):
    lr_path = lr_path.numpy().decode('utf-8')
    hr_path = hr_path.numpy().decode('utf-8')
    lr_image = cv2.imread(lr_path)
    hr_image = cv2.imread(hr_path)

    # Resize images to the target size
    lr_image = cv2.resize(lr_image, target_size) / 255.0  # Normalize to [0, 1]
    hr_image = cv2.resize(hr_image, target_size) / 255.0  # Normalize to [0, 1]

    return np.array(lr_image, dtype=np.float32), np.array(hr_image, dtype=np.float32)

# Wrapper for TensorFlow Dataset compatibility
def tf_load_and_preprocess_image(lr_path, hr_path):
    lr_image, hr_image = tf.py_function(
        func=lambda lr, hr: load_and_preprocess_image(lr, hr, target_size=(256, 256)),
        inp=[lr_path, hr_path],
        Tout=[tf.float32, tf.float32]
    )
    lr_image.set_shape((256, 256, 3))
    hr_image.set_shape((256, 256, 3))
    return lr_image, hr_image


# Create tf.data.Dataset pipeline
def create_dataset(lr_dir, hr_dir, batch_size=8):
    print("Fetching Images from LR and HR Directories")
    lr_paths = glob(os.path.join(lr_dir, "*.png"))
    hr_paths = glob(os.path.join(hr_dir, "*.png"))
    print("Fetched Paths")

    dataset = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths))
    dataset = dataset.map(
        lambda lr, hr: tf_load_and_preprocess_image(lr, hr),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    print("Finished Processing Dataset")
    return dataset


# Training loop
def train(generator, discriminator, epochs=1, batch_size=8):
    print("Begin Initializing Data")
    lr_dir = "processed_train_lr"
    hr_dir = "processed_train_hr"
    dataset = create_dataset(lr_dir, hr_dir, batch_size)
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy()
    mse_loss = tf.keras.losses.MeanSquaredError()
    gen_optimizer = tf.keras.optimizers.Adam(1e-4)
    disc_optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(lr_images, hr_images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_hr = generator(lr_images)
            real_output = discriminator(hr_images)
            fake_output = discriminator(fake_hr)
            gen_loss = binary_crossentropy(tf.ones_like(fake_output), fake_output) + mse_loss(hr_images, fake_hr)
            disc_loss = binary_crossentropy(tf.ones_like(real_output), real_output) + binary_crossentropy(tf.zeros_like(fake_output), fake_output)
        gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        return gen_loss, disc_loss

    print("Begin Training Model")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for lr_images, hr_images in dataset:
            gen_loss, disc_loss = train_step(lr_images, hr_images)
            print(f"Gen loss: {gen_loss:.4f} | Disc loss: {disc_loss:.4f}")
        # generator.save_weights(f"generator_epoch_{epoch+1}.weights.h5")
        # discriminator.save_weights(f"discriminator_epoch_{epoch+1}.weights.h5")
    
    # save the entire model
    generator.save('generator_model.keras')


if __name__ == '__main__':
    generator = build_generator()
    discriminator = build_discriminator()
    train(generator, discriminator, epochs=5)
