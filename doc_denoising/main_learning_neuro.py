import os

import cv2
import numpy as np
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import random_brightness, ImageDataGenerator
from matplotlib import pyplot as plt
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.get_visible_devices()

from keras.models import Model
from keras import Input
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Flatten, Dense, Reshape

DIRTY_PATH = "denoising_dirty_documents/train/train/"
CLEANED_PATH = "denoising_dirty_documents/train_cleaned/train_cleaned/"
TEST_PATH = "denoising_dirty_documents/test/test/"


def get_images_paths(path):
    return [path + x for x in os.listdir(path)]


def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (540, 420))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (420, 540, 1))
    return img


def train_test_split(X_train, Y_train, test_size):
    X_test = X_train[int(len(X_train) * (1 - test_size)):]
    Y_test = Y_train[int(len(Y_train) * (1 - test_size)):]
    X_train = X_train[:int(len(X_train) * (1 - test_size))]
    Y_train = Y_train[:int(len(Y_train) * (1 - test_size))]
    return X_train, X_test, Y_train, Y_test


def model1():
    input_layer = Input(shape=(420, 540, 1))
    # encoding
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Dropout(0.5)(x)

    # decoding
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)

    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model


def model2():
    input_img = Input(shape=(420, 540, 1), name='image_input')

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='Conv1')(input_img)
    x = MaxPooling2D((2, 2), padding='same', name='pool1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='pool2')(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv3')(x)
    x = UpSampling2D((2, 2), name='upsample1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='Conv4')(x)
    x = UpSampling2D((2, 2), name='upsample2')(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='Conv5')(x)

    # Model
    autoencoder = Model(inputs=input_img, outputs=x)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return autoencoder


def prepare_imgs(paths):
    return np.array([process_image(path) for path in paths])


# чтение и обработка изображений
train_img = sorted(get_images_paths(DIRTY_PATH))
train_cleaned_img = sorted(get_images_paths(CLEANED_PATH))
test_img = sorted(get_images_paths(TEST_PATH))
train = prepare_imgs(train_img)
train_cleaned = prepare_imgs(train_cleaned_img)
test = prepare_imgs(test_img)

# вывод изображений
plt.figure(figsize=(15, 25))
for i in range(0, 8, 2):
    plt.subplot(4, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train[i][:, :, 0], cmap='gray')
    plt.title('Noise image: {}'.format(train_img[i]))

    plt.subplot(4, 2, i + 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_cleaned[i][:, :, 0], cmap='gray')
    plt.title('Denoised image: {}'.format(train_img[i]))

plt.show()

# преобразование данных для обучения
X_train = np.asarray(train)
Y_train = np.asarray(train_cleaned)
X_test = np.asarray(test)

# разделение данных
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)

# инициализация и компиляция модели обучения
model = model1()
model.summary()

# аугментация данных
datagen = ImageDataGenerator(
    brightness_range=(1.0, 10.0)
)
datagen.fit(X_train)

# обучение модели
callback = EarlyStopping(monitor='loss', patience=30)
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=14, batch_size=1, verbose=1,
                    callbacks=[callback])

# сохранение данных обучения модели
model.save('model_data/')

# вывод данных об обучении модели
epoch_loss = history.history['loss']
epoch_val_loss = history.history['val_loss']
epoch_mae = history.history['mae']
epoch_val_mae = history.history['val_mae']

plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
plt.plot(range(0, len(epoch_loss)), epoch_loss, 'b-', linewidth=2, label='Train Loss')
plt.plot(range(0, len(epoch_val_loss)), epoch_val_loss, 'r-', linewidth=2, label='Val Loss')
plt.title('Evolution of loss on train & validation datasets over epochs')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(range(0, len(epoch_mae)), epoch_mae, 'b-', linewidth=2, label='Train MAE')
plt.plot(range(0, len(epoch_val_mae)), epoch_val_mae, 'r-', linewidth=2, label='Val MAE')
plt.title('Evolution of MAE on train & validation datasets over epochs')
plt.legend(loc='best')

plt.show()

# тестирование модели
Y_test = model.predict(X_test, batch_size=1)

# вывод некоторых изображений из тестовой выборки, обработанных моделью
plt.figure(figsize=(15, 25))
for i in range(0, 8, 2):
    plt.subplot(4, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i][:, :, 0], cmap='gray')
    plt.title('Noisy image: {}'.format(test_img[i]))

    plt.subplot(4, 2, i + 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(Y_test[i][:, :, 0], cmap='gray')
    plt.title('Denoised by autoencoder: {}'.format(test_img[i]))

plt.show()
