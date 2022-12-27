import os
from functools import reduce
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

CLEANED_PATH = "denoising_dirty_documents/train_cleaned/train_cleaned/"
DIRTY_PATH = "denoising_dirty_documents/train/train/"


def pHash(img):
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:8, 0:8]
    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            hash.append(1 if dct_roi[i, j] > avreage else 0)

    return int(reduce(lambda a, b: a * 2 | b, hash))


def get_params_img(new_img):
    clean_paths = [CLEANED_PATH + x for x in os.listdir(CLEANED_PATH)]
    cleaned_images = [cv2.imread(x) for x in clean_paths]
    cleaned_images = [item for item in cleaned_images]
    dirty_paths = [DIRTY_PATH + x for x in os.listdir(DIRTY_PATH)]
    dirty_images = [cv2.imread(x) for x in dirty_paths]
    dirty_images = [item for item in dirty_images]

    x = [np.reshape(dirty.flatten(), (dirty.shape[0] * dirty.shape[1], 1)) for dirty in
         [cv2.cvtColor(item, cv2.COLOR_BGR2GRAY) for item in dirty_images]]
    y = [np.reshape(clean.flatten(), (clean.shape[0] * clean.shape[1], 1)) for clean in
         [cv2.cvtColor(item, cv2.COLOR_BGR2GRAY) for item in cleaned_images]]

    a = []
    b = []
    regressor = LinearRegression(copy_X=True, fit_intercept=True)
    for x_item, y_item in zip(x, y):
        regressor.fit(x_item, y_item)
        a.append(regressor.coef_[0])
        b.append(regressor.intercept_)
    fig, axs = plt.subplots(nrows=1, ncols=3)
    hashes = [pHash(img) for img in dirty_images]

    axs[0].plot(hashes, a, "o", color="blue")
    axs[0].set_xlabel("a", fontdict={"size": 20})
    axs[1].plot(hashes, b, "o", color="red")
    axs[1].set_xlabel("b", fontdict={"size": 20})
    axs[2].plot(hashes, "o", color="red")
    axs[2].set_xlabel("hash", fontdict={"size": 20})
    # plt.show()

    regressor.fit(np.array(hashes).reshape(-1, 1), a)
    pr_item = np.array(pHash(new_img)).reshape(-1, 1)
    w0 = regressor.predict(pr_item)

    regressor.fit(np.array(hashes).reshape(-1, 1), b)
    w1 = regressor.predict(pr_item)

    return w0, w1
