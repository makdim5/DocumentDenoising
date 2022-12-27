import os
import keras
from matplotlib import pyplot as plt
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.get_visible_devices()

from PIL import Image
from pytesseract import pytesseract
import cv2
import numpy as np


# before use install tesseract from https://tesseract-ocr.github.io/tessdoc/Installation.html
def get_text_from_img(path_to_image):
    path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.tesseract_cmd = path_to_tesseract

    img = Image.open(path_to_image)

    return pytesseract.image_to_string(img)


def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (540, 420))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (420, 540, 1))
    return img


model = keras.models.load_model("model_data/")
model.summary()

DIRTY_PATH = "testim.jpg"
dirty = process_image(DIRTY_PATH)

# predict/clean test images
Y_test = model.predict(np.array([dirty]), batch_size=1)

plt.figure(figsize=(15, 25))

plt.subplot(4, 2, 1)
plt.xticks([])
plt.yticks([])
plt.imshow(dirty[:, :, 0], cmap='gray')
plt.title('Noisy image: ')

plt.subplot(4, 2, 2)
plt.xticks([])
plt.yticks([])
plt.imshow(Y_test[0][:, :, 0], cmap='gray')
plt.title('Denoised by autoencoder')

plt.show()

new_img = Y_test[0] * 255.0
cv2.imwrite("new.png", new_img)
print(get_text_from_img("new.png"))
os.system("new.png")

blur = np.asarray(process_image("new.png") * 255, dtype="int32")
org = np.asarray(process_image(DIRTY_PATH) * 255, dtype="int32")
print("MSE: ", mse(blur, org))
print("RMSE: ", rmse(blur, org))
print("PSNR: ", psnr(blur, org))
print("SSIM: ", ssim(blur, org))
print("UQI: ", uqi(blur, org))
print("MSSSIM: ", msssim(blur, org))
print("ERGAS: ", ergas(blur, org))
print("SCC: ", scc(blur, org))
print("RASE: ", rase(blur, org))
print("SAM: ", sam(blur, org))
print("VIF: ", vifp(blur, org))
