from skimage import io
import cv2
import os
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

a = 0 
kernel_c = np.ones((101,101),np.uint8)
kernel_o = np.ones((3,3),np.uint8)
kernel_d = np.ones((3,3),np.uint8)

#'directory_name' is the foder name with images.
#directory_name = input('Directory name:')
#the dir where the imgs folder is 
directory_name = '/Users/guopinchen/Desktop/digital_image_prossessing/image_processing_theory_and_applications/上傳github/fish_segmentation拷貝/imgs'


#Read each image in this foder
for filename in os.listdir(r"/"+directory_name):
    a+=1
    inp = input('press enter for next pic(type exit for exit) :')
    if inp == 'exit':
        break
    url = directory_name + "/" + filename
    try:
        image = cv2.imread(url)        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray = 255-gray
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,51,3)
        thresh = 255-thresh
        #remove noise
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_o)
        thresholding = 80
        #blur to remove noise
        blur = gaussian_filter(opening, sigma=5)
        thresh2 = cv2.threshold(blur,thresholding,255, cv2.THRESH_BINARY)[1]
        #Connect seperated area of the fish body
        closing = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel_c)
        blur2 = gaussian_filter(closing, sigma=5)
        thresh3 = cv2.threshold(blur2,thresholding,255, cv2.THRESH_BINARY)[1]
        imageOK = thresh3

        #Find contours(detection)
        cnts = cv2.findContours(imageOK, \
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        min_area = 1000
        max_area = 15000
        text = 'frame: %s' %a
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area and area < max_area:
                x,y,w,h = cv2.boundingRect(c)
                try:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 5)
                    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0, 255, 255), 5, cv2.LINE_AA)
                except:
                    pass
        Image.fromarray(image).show()
    except:
        print('error at frame: %s' %a)