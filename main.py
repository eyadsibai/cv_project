import cv2
import numpy as np
from glob import glob
import os
from sklearn.cluster import KMeans

def extract_sift_features(image_path):
    img = cv2.imread(image_path)
    # print(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors
