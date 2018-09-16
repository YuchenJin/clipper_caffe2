from __future__ import print_function
import os
import sys
import glob
import numpy as np
import skimage.io
import skimage.transform
import base64
from StringIO import StringIO
from skimage.io import imsave

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def rescale(img, input_height, input_width):
    # Get original aspect ratio
    aspect = img.shape[1]/float(img.shape[0])
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    return imgScaled

def predict_function():
    max_count = 1999
    data_dir = "images/"
    for fn in os.listdir(data_dir):
	try:
            curr_img = skimage.io.imread(os.path.join(data_dir, fn))
            img = skimage.img_as_float(curr_img).astype(np.float32)
            img = rescale(img, 300,300)
            img = crop_center(img, 300, 300)
            imsave('./resized_images/{}'.format(fn), img)
	    max_count -= 1
            if max_count <= 0:
                break
	except:
	    pass

predict_function()

