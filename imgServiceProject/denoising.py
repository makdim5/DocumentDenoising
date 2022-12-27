import cv2
import numpy as np
import tensorflow as tf

TEMP_IMG_PATH = "new.png"


def denoise_image(im_path: str) -> None:
    model = tf.keras.models.load_model("model_data/")
    cv2.imwrite(TEMP_IMG_PATH, model.predict(np.array([process_image(im_path)]),
                                             batch_size=1)[0] * 255.0)


def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (540, 420))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (420, 540, 1))
    return img
