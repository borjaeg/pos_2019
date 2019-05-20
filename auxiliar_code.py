from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras.applications.inception_v3 import preprocess_input
import matplotlib
import tensorflow as tf
import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.preprocessing import image

def load_image(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (299, 299))
    
    return im

def preprocess_image(im):
    im = preprocess_input(im)
    im = np.expand_dims(im, axis = 0)
    
    return im

def denormalise(im):
    im/=2
    im+=0.5
    im*=255
    
    return im.astype(np.uint8)

def load_neural_network():
    nn = InceptionV3()
    return nn

def plot_images_contrast(im_1, im_2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 10))

    axs[0].set_title("Original Image")
    axs[0].imshow(im_1)
    axs[0].axis("off")
    axs[1].set_title("Hacked Image")
    axs[1].imshow(im_2)
    axs[1].axis("off")
    plt.show()

def plot_images_noise(im_1, im_2):
    fig, axs = plt.subplots(1, 3, figsize=(12, 10))

    axs[0].set_title("Original Image")
    axs[0].imshow(im_1)
    axs[0].axis("off")
    axs[1].set_title("Noise")
    axs[1].imshow(im_1-im_2)
    axs[1].axis("off")
    axs[2].set_title("Hacked Image")
    axs[2].imshow(im_2)
    axs[2].axis("off")
    plt.show()
    
def predict(nn, image):
    pred = decode_predictions(nn.predict(preprocess_image(image)), top=1)[0][0][1]
    print(pred)