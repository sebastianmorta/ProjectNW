import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import easyocr
im_1_path = './datasets/000/a01-000u.png'
def recognize_text(img_path):
    reader = easyocr.Reader(['en'])
    return  reader.readtext(img_path, detail=0, paragraph=True)
#result = recognize_text(im_1_path)
#print(result)

img = cv2.imread(im_1_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
# digits = datasets.load_digits()
#
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Training: %i' % label)