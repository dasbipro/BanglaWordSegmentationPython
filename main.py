from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image 
from numpy import *
def main():
    print('Hello AI')
    image = cv2.imread('WordInputFile/WordPage000000.bmp')
    #cv2.imshow("image",image)
    #cv2.waitKey()
    # define parameters of HOG feature extraction
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    threshold = .3
    data = []
    labels = []

    pos_im_path = r"./Data/positive" 
    neg_im_path = r"./Data/negative"

    pos_im_listing = os.listdir(pos_im_path) 
    neg_im_listing = os.listdir(neg_im_path)
    num_pos_samples = len(pos_im_listing) 
    num_neg_samples = len(neg_im_listing)
    print(num_pos_samples,num_neg_samples)
    for file in pos_im_listing:
        img = Image.open(pos_im_path + '/' + file) # open the file
        # print(pos_im_path + '/' + file)
        #img = img.resize((64,128))
        gray = img.convert('L') 
        # calculate HOG for positive features
        fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
        # cv2.imshow('abcd',fd)
        # cv2.waitKey()
        labels.append(1)
    for file in neg_im_listing:
        img= Image.open(neg_im_path + '//' + file)
        #img = img.resize((64,128))
        gray= img.convert('L')
        # Now we calculate the HOG for negative features
        fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
        data.append(fd)
        cv2.waitKey()
        labels.append(0)
    # encode the labels, converting them from strings to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    print(" Constructing training/testing split...")
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        data, labels, test_size=0.20, random_state=42)
    split_point = 75*num_pos_samples //100
    trainData = data[:split_point]
    testData = data[split_point:]
    trainLabels = labels[:split_point]
    testLabels = labels[split_point:]

    split_point = 75*num_pos_samples //100
    trainData = np.asarray(trainData)
    testData = np.asarray(testData)
    trainLabels = np.asarray(trainLabels)
    testLabels = np.asarray(testLabels)

    print(" Training Linear SVM classifier...")
    model = LinearSVC()
    model.fit(trainData, trainLabels)
    
if(__name__=='__main__'):
    main()