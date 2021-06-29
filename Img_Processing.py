# Author: Andres Graterol 
# Date: 06/10/21
''' 
This code was based on the tutorial given by Prince Canuma on towards data science: 
https://towardsdatascience.com/image-pre-processing-c1aec0be3edf
And uses the Cambridge-driving Labeled Video Database (CamVid) 
'''

# TODO: Add Image Normalisation, edge detection, and Histogram Equalization Functionality

import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import cv2

# Location of the dataset (GLOBAL VARIABLE)
data_path = "camvid"

# testing if path is valid 
#print(os.path.isdir(data_path))

''' Put files into lists and return them as one list 
with all images in the folder'''
def loadImages(path):
    image_files = sorted([os.path.join(path, '701_StillsRaw_full', file)
                        for file in os.listdir(path + "/701_StillsRaw_full")
                        if file.endswith('.png')])
    
    return image_files

# Function for plotting the grayscale histogram for our information 
def plot_grayHist(gray_img, gray_hist):
    # Showing grayscale image 
    plt.figure()
    plt.axis("off")
    plt.imshow(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB))
    
    # Showing corresponding grayscale histogram 
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(gray_hist)
    plt.xlim([0, 256])
    plt.show()

# Function for plotting the normalized grayscale histogram for our information
def plot_normGrayHist(gray_img, gray_hist):
    # Showing grayscale image 
    plt.figure()
    plt.axis("off")
    plt.imshow(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB))
    
    gray_hist /= gray_hist.sum()

    plt.figure()
    plt.title("Grayscale Histogram (Normalized)")
    plt.xlabel("Bins")
    plt.ylabel("% of Pixels")
    plt.plot(gray_hist)
    plt.xlim([0, 256])
    plt.show()

# Using the CLAHE function to increase contrast in a gray image 
def equalize_hist(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray_img)
    return cl1

# Functions for displaying images
def display_two(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()

def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()

# Function for normalizing an image 
def norm(res_img):
    normed = cv2.normalize(res_img, np.zeros((800, 800)), 0, 255, cv2.NORM_MINMAX)
    return normed

def adjust_gamma(res_img, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma 
    table = np.array([((i/ 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table 
    return cv2.LUT(res_img, table)

def gamma_loop(res_img):
    for gamma in np.arange(0.0, 3.5, 0.5):
        # ignore when gamma is 1 (there will be no change to the image)
        if gamma == 1:
            continue
        
        # apply gamma correction and show the images
        gamma = gamma if gamma > 0 else 0.1
        adjusted = adjust_gamma(res_img, gamma=gamma)
        cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3) 
        cv2.imshow("Images", np.hstack([res_img, adjusted]))
        cv2.waitKey(0) 

# Function for noise removal -- Gaussian Smoothing
def gaussian_blur(res_img):
    no_noise = []
    for i in range(len(res_img)):
        # An adjustment to the kernel size here
        # makes a large difference in the 
        blur = cv2.GaussianBlur(res_img[i], (3, 3), 0)
        no_noise.append(blur)

    # Returns array of images all gaussian blurred
    return no_noise

# Function for turning images into grayscale 
def grayscale(res_img):
    gray = cv2.cvtColor(res_img, cv2.COLOR_RGB2GRAY)

    return gray 

# Function for the first iteration of image segmentation 
# Takes a grayed & blurred image as an argument 
def segment(grayblur_img):

    # Using Otsu's method to segment the foreground and the background 
    # NOTE: Otsu's method works best when the grayscale histogram of our image is bi-modal
    # When this does not occur, it is better to use Adaptive Thresholding 
    (ret, thresh) = cv2.threshold(grayblur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print("[INFO] otsu's thresholding value: {}".format(ret))

    return thresh  

# Morphology to further reduce noise & improve image segmentation 
# Takes thresholded (segmented) image as an argument 
def morph(thresh_img):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    (ret, sure_fg) = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    print("[INFO] Thresholding value for foreground: {}".format(ret))

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # returns a tuple containing all the segments of the image 
    return sure_bg, sure_fg, unknown

# Function to complete image segmentation using the watershed algorithm
# Takes a blurred image as an argument
def watershed(sure_fg, unknown, blur_img):
    (ret, markers) = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(blur_img, markers)
    blur_img[markers == -1] = [255, 0, 0]

    return markers 

# Function for performing adaptive thresholding 
def adaptiveThresholding(grayblur_img):
    adaptThresh = cv2.adaptiveThreshold(grayblur_img, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    return adaptThresh

# Function for perfroming gaussian adaptive thresholding 
def gaussianThresholding(grayblur_img):
    gaussianThresh = cv2.adaptiveThreshold(grayblur_img, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    return gaussianThresh

# Preprocessing steps:
# 1. Read Image 
# 2. Resize Image 
# 3. Remove Noise
# 4. Segmentation 
# 5. Morphology
def processing(data, index):
    
    if (index > len(data)):
        print("THIS IMAGE IS OUT OF RANGE")    
        sys.exit()

    # 1. 
    # This loop reads in all the images  
    # data[:index] if you want to narrow down the dataset!!!
    img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in data]
    try:
        print('Original size', img[index].shape) # <---------------- Image control
    except AttributeError:
        print("shape not found")
   
    # 2. --------------------------------
    # Setting dimensions of the resize
    height = 220
    width = 220
    dim = (width, height)
    res_img = []
    for i in range(len(img)):
        res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)

    # Checking the size
    # Resized dimensions should be: (220, 220, 3)
    try:
        print('RESIZED', res_img[index].shape) # <--------------------- Image control
    except AttributeError:
        print("shape not found")
    
    # Visualizing one of the images in the array
    original = res_img[index] # <------------------- Image control
    display_one(original)

    # Normalizing the image and comparing the two 
    normalized = norm(original)
    display_two(original, normalized, 'Original', 'Normalized')

    # Applying Gamma Correction 
    # loop over various values of gamma
    gamma_loop(original)

    # Displaying image in grayscale
    gray = grayscale(original)
    display_two(original, gray, 'Original', 'Grayscale')

    # Displaying the grayscale histogram 
    gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plot_grayHist(gray, gray_hist)

    # Displaying the normalized grayscale histogram 
    plot_normGrayHist(gray, gray_hist)

    # Applying histogram equalization using the CLAHE function 
    clahe = equalize_hist(gray)

    # Comparing the two images:
    display_two(gray, clahe, 'Gray', 'CLAHE Applied')

    # 3. ----------------------------------
    # Removing noise using Gaussian noise removal 
    no_noise = gaussian_blur(res_img)
    image = no_noise[index]  # <----------------------- Image control
    display_two(original, image, 'Original', 'Gaussian Smoothed')

    # 4. ---------------------------------
    # Segmentation
    grayblur = grayscale(image) 
    
    # Displaying the image both grayed and blurred
    display_two(original, grayblur, 'Original', 'Grayed & Blurred')

    thresh = segment(grayblur)

    # Displaying segmented images
    display_two(original, thresh, 'Original', 'Segmented')

    # 5. --------------------------------
    # Further noise removal (Morphology)
    sure_bg, sure_fg, unknown = morph(thresh)

    # Displaying segmented back ground
    display_two(original, sure_bg, 'Original', 'Segmented Background')

    # Now to separate different objects in the image with markers 
    # Marker labeling
    markers = watershed(sure_fg, unknown, image)

    # Displaying markers on the image
    display_two(original, markers, 'Original', 'Marked')

    # Now to apply adaptive thresholding to see 
    # how it compares to OTSU's global thresholding images 
    adaptThresh = adaptiveThresholding(grayblur)

    display_two(original, adaptThresh, 'Original', 'Mean Adaptive Thresholding')

    gaussianThresh = gaussianThresholding(grayblur)
    
    display_two(original, gaussianThresh, 'Original', 'Gaussian Adaptive Thresholding')

def main():
    # Calling global variable
    global data_path
    '''The var Dataset is a list with all images in the folder '''          
    dataset = loadImages(data_path)
    print("This dataset has {}".format(len(dataset)) + " images!")

    '''
    print("The first 3 in the folder:\n", dataset[:3])
    print("--------------------------------")
    '''

    index = 1
    print("Preproccessing image #{}".format(index))
    print("Here we go...")
    
    # Sending the images to pre-processing
    processing(dataset, index)

main()