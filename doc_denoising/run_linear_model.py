import os
from PIL import Image
from pytesseract import pytesseract
import cv2
import numpy as np
from linear_model import get_params_img


# before use install tesseract from https://tesseract-ocr.github.io/tessdoc/Installation.html
def get_text_from_img(path_to_image):
    path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.tesseract_cmd = path_to_tesseract

    img = Image.open(path_to_image)

    return pytesseract.image_to_string(img)


if __name__ == "__main__":
    dirty = cv2.imread("denoising_dirty_documents/test/test/1.png")  # write filepath to your image here
    a, b = get_params_img(dirty)

    dirty = cv2.cvtColor(dirty, cv2.COLOR_BGR2GRAY)
    dirty_flat = dirty.flatten()
    x = np.reshape(dirty_flat, (dirty.shape[0] * dirty.shape[1], 1))
    result = np.array([(item * a + b) for item in x])
    result = np.reshape(result, (dirty.shape[0], dirty.shape[1]))

    cv2.imwrite("new.png", result)
    print(get_text_from_img("new.png"))
    os.system("new.png")
