import numpy as np
from PIL import Image
from keras.src.applications.mobilenet import preprocess_input
from keras.src.utils import img_to_array


def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), resample=Image.NEAREST)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x