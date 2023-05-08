import cv2
import numpy as np
from sklearn.cluster import KMeans


def extract_features(X, **kwargs):
    extractor = cv2.SIFT_create(**kwargs)
    features = []
    for x in X:
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        _, descriptors = extractor.detectAndCompute(gray, None)
        if descriptors is not None:
            features.extend(descriptors)
    return features


def create_vocabulary(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init="auto")
    kmeans.fit(features)
    return kmeans


def create_histograms(X, kmeans):
    histograms = []
    X_ = np.array(X).astype(np.float64)
    predictions = kmeans.predict(X_)
    for i in predictions:
        histogram = np.zeros(len(kmeans.cluster_centers_))
        histogram[i] += 1
        histograms.append(histogram)

    return histograms
