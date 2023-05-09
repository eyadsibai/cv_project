import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from dask import delayed
import dask
from tqdm import tqdm


def extract_features2(X, extractor):
    features = []
    for x in tqdm(X):
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        _, descriptors = extractor.detectAndCompute(gray, None)
        if descriptors is None:
            continue
        features.extend(descriptors)
    return features


@delayed
def extract_single_image_features(x, extractor):
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    _, descriptors = extractor.detectAndCompute(gray, None)
    return descriptors if descriptors is not None else []


def extract_features(X, extractor):

    features = []
    for x in X:
        features.append(extract_single_image_features(x, extractor))

    # Compute the result
    features = dask.compute(*features)
    return features


def create_vocabulary(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init="auto")
    kmeans.fit(features)
    return kmeans


def create_histograms2(X, extractor, kmeans):
    histograms = []
    for x in X:
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        _, descriptors = extractor.detectAndCompute(gray, None)
        histogram = np.zeros(len(kmeans.cluster_centers_))
        if descriptors is None:
            histograms.append(histogram)
            continue
        for desc in descriptors:
            idx = kmeans.predict([desc])[0]
            histogram[idx] += 1
        histograms.append(histogram)
    return histograms


def create_histograms(X, extractor, kmeans):
    histograms = []
    for x in tqdm(X):
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        _, descriptors = extractor.detectAndCompute(gray, None)
        histogram = np.zeros(len(kmeans.cluster_centers_))
        if descriptors is not None:
            # Use vq function to quantize descriptors to closest cluster
            idx, _ = vq(descriptors, kmeans.cluster_centers_)
            # Count occurrences of each cluster using bincount and add to histogram
            bin_count = np.bincount(idx, minlength=len(kmeans.cluster_centers_))
            histogram += bin_count
        histograms.append(histogram)
    return histograms
