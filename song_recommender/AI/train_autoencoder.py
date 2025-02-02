import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.python.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.callbacks import EarlyStopping


def create_autoencoder(input_shape=(128, 128, 1), embedding_dim=512):

    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same',
        kernel_regularizer=l2(1e-4)
    )(encoder_input)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same',
        kernel_regularizer=l2(1e-4)
    )(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same',
        kernel_regularizer=l2(1e-4)
    )(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same',
        kernel_regularizer=l2(1e-4)
    )(x)
    x = layers.Dropout(0.4)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Flatten()(x)
    embedding = layers.Dense(
        embedding_dim, activation='relu',
        kernel_regularizer=l2(1e-4), name='embedding_layer'
    )(x)

    x = layers.Dense(
        16 * 16 * 512, activation='relu',
        kernel_regularizer=l2(1e-4)
    )(embedding)
    x = layers.Reshape((16, 16, 512))(x)

    x = layers.Conv2DTranspose(
        512, (3, 3), activation='relu', padding='same',
        kernel_regularizer=l2(1e-4)
    )(x)
    x = layers.Dropout(0.4)(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(
        256, (3, 3), activation='relu', padding='same',
        kernel_regularizer=l2(1e-5)
    )(x)
    x = layers.Dropout(0.4)(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(
        128, (3, 3), activation='relu', padding='same',
        kernel_regularizer=l2(1e-5)
    )(x)
    x = layers.Dropout(0.3)(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), activation='relu', padding='same',
        kernel_regularizer=l2(1e-5)
    )(x)
    x = layers.Dropout(0.3)(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoder_output = layers.Conv2DTranspose(
        1, (3, 3), activation='sigmoid', padding='same'
    )(x)

    autoencoder = models.Model(encoder_input, decoder_output)
    encoder = models.Model(encoder_input, embedding)

    return autoencoder, encoder


input_shape = (256, 256, 1)
embedding_dim = 128
autoencoder, encoder = create_autoencoder(input_shape, embedding_dim)

def read_spectrogram_image(filepath: str) -> np.array:
    spectrogram = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    spectrogram = cv2.resize(spectrogram, (256, 256), interpolation=cv2.INTER_AREA)
    spectrogram = spectrogram / 255.0
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    print(spectrogram.shape)
    return spectrogram

def read_spectrogram(filepath: str) -> np.array:
    spectrogram = np.load(filepath)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    print(spectrogram.shape)
    print(f"Min value: {np.min(spectrogram)}, Max value: {np.max(spectrogram)}")
    return spectrogram



dataset_path = "spectrograms"
tracks = os.listdir(dataset_path)


spectrograms = []
for track in tracks:
    file_path = os.path.join(dataset_path, track)
    spectrograms.append(read_spectrogram_image(file_path))
    #spectrograms.append(read_spectrogram(f"{dataset_path}/{track}"))

combined_array = np.stack(spectrograms, axis=0)
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

learning_rate = 0.0015
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss= "mae",
    metrics=["mse"]
)
epochs = 50
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)
history = autoencoder.fit(combined_array, combined_array, epochs=epochs, batch_size=8, validation_split=0.2, callbacks=[early_stopping])
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(history.history['loss'], color='b', label="Training loss")
ax.plot(history.history['val_loss'], color='r', label="Validation loss")
ax.set_xticks(np.arange(1, epochs, 1))
ax.set_yticks(np.arange(0, 1, 0.1))
ax.set_title("Training and Validation Loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
plt.show()
plt.savefig('training_validation_loss.png')
autoencoder.save('autoencoder_model.h5')
encoder.save('encoder_model.h5')

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(history.history['mse'], color='g', label="Training MSE")
ax.plot(history.history['val_mse'], color='orange', label="Validation MSE")
ax.set_xticks(np.arange(1, epochs, 1))
ax.set_yticks(np.arange(0, 0.1, 0.01))
ax.set_title("Training and Validation MAE")
ax.set_xlabel("Epochs")
ax.set_ylabel("MSE")
ax.legend()
plt.show()
plt.savefig('training_validation_mse.png')