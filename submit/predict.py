import preprocessing
import cv2
import numpy as np
from keras.models import load_model

def one_hot_to_char(vec):
    return chr(np.where(vec[0] == 1)[0] + 65)
    
    
def decaptcha(filenames):
    model = load_model('model.h5')
    codes = []
    numChars = np.zeros((len(filenames),))
    index = 0
    for filename in filenames:
        image = cv2.imread(filename)
        letters = preprocessing.preprocess_image(image)
        outputs = []
        for letter in letters:
            outputs.append(one_hot_to_char(model.predict(letter.reshape(1, letter.shape[0], letter.shape[1], 1))))
        codes.append(''.join(outputs))
        numChars[index] = len(outputs)
        index += 1
    return (numChars, codes)
