import os
import tensorflow as tf
import cv2
import numpy as np

DIRTY_PATH = "denoising_dirty_documents/train/train/"
CLEANED_PATH = "denoising_dirty_documents/train_cleaned/train_cleaned/"


def get_images_paths(path):
    return [path + x for x in os.listdir(path)]


def prepare_imgs(paths):
    return np.array([cv2.imread(path) for path in paths])


# чтение и обработка изображений
train_img = sorted(get_images_paths(DIRTY_PATH))
train_cleaned_img = sorted(get_images_paths(CLEANED_PATH))
train = prepare_imgs(train_img)
train_cleaned = prepare_imgs(train_cleaned_img)

train_im = train[0]

cv2.imshow("2", train_im)

# layer = tf.keras.layers.RandomContrast([1.0, 10.0])
# layer = tf.keras.Sequential([
#   tf.keras.layers.RandomFlip("horizontal_and_vertical"),
#   tf.keras.layers.RandomRotation(0.2),
# ])

# layer = tf.keras.Sequential([
#   tf.keras.layers.RandomZoom(.5, .2, interpolation='nearest')
# ])

layer = tf.keras.Sequential([
  tf.keras.layers.RandomContrast([1.0, 10.0])
])

cv2.imwrite("new.png", layer(train_im).numpy().astype("uint8"))
os.system("new.png")
cv2.waitKey(0)
