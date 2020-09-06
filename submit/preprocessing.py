import cv2
import numpy as np
import imutils

def resize_to_fit(image, width, height):
    (h, w) = image.shape[:2]
    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    return image


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    image[image==image[0,0]] = 255
    kernel_1 = np.ones((7,7), np.uint8) 
    kernel_2 = np.ones((4,4), np.uint8) 
    img_dilated_1 = cv2.dilate(image, kernel_1, iterations=1)
    image = cv2.cvtColor(img_dilated_1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image, (5,5), 0)
    retval_blur ,thresh_blur = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image = cv2.bitwise_not(thresh_blur)
    cols = np.sum(image, axis=0)
    rows = np.sum(image, axis=1)
    flag = 0
    thresh = 255*5
    temp = 0
    start = []
    end = []
    for i in range(len(cols)):
        if cols[i] > thresh and flag==0:
            flag = 1
            temp = i
        if cols[i] <= thresh and flag==1:
            if (i-temp)>5 :
                start.append(temp)
                end.append(i)
            flag = 0
    char = np.zeros((len(start), 140, 140))
    for i in range(len(start)):
        l = end[i] - start[i]
        char[i][:, 75-(l//2):75+l-(l//2)] = image[5:-5,start[i]:end[i]]
    resized = np.zeros((len(start),30,30))
    for i in range(len(start)):
        char[i] = 255*np.ones((140,140)) - char[i]
        resized[i]=resize_to_fit(char[i], 30, 30)
    return resized