import cv2
import numpy as np
from sklearn.cluster import KMeans
import json
import random

def sliding_window(image, window_size, step):
    for y in range(0, image.shape[0] - window_size[1], step):
        for x in range(0, image.shape[1] - window_size[0], step):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def create_histograms(images, extractor, kmeans):
    histograms = []
    for img in images:
        keypoints, descriptors = extractor.detectAndCompute(img, None)
        histogram = np.zeros(len(kmeans.cluster_centers_))
        if descriptors is not None:
            for desc in descriptors:
                idx = kmeans.predict([desc])[0]
                histogram[idx] += 1
            histograms.append(histogram)
        else:
            histograms.append(histogram)
    return histograms

def create_vocabulary(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(features)
    return kmeans

def extract_features(images, extractor):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = extractor.detectAndCompute(gray, None)
        if descriptors is not None:
            features.extend(descriptors)
    return features



def object_detection(image, window_size, step, extractor, kmeans, scaler, classifier, score_threshold, nms_threshold):
    # Initialize detection results
    detections = []

    # Perform sliding window
    for x, y, patch in sliding_window(image, window_size, step):

        # Create a histogram of visual words
        patch_histogram = create_histograms([patch], extractor, kmeans)[0]

        # Normalize the histogram
        patch_histogram = scaler.transform([patch_histogram])

        # Classify the patch
        probabilities = classifier.predict_proba(patch_histogram)[0]
        class_idx = np.argmax(probabilities)
        class_score = probabilities[class_idx]

        # Check if the score is above the threshold
        if class_score > score_threshold:
            detections.append([x, y, class_score, class_idx])

    # Perform Non-Maximum Suppression (NMS)
    # detections = non_maximum_suppression(detections, nms_threshold)

    # Draw bounding boxes for the detected objects
    for (x, y, score, class_idx) in detections:
        if class_idx == 1:
            image = cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
            image = cv2.putText(image, f"Class: {class_idx}, Score: {score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image



def generate_samples(image, annotations, window_size, step, iou_threshold):
    positive_samples = []
    negative_samples = []

    for x, y, patch in sliding_window(image, window_size, step):
        patch_bbox = [x, y, x + window_size[0], y + window_size[1]]

        max_iou = 0
        for annotation in annotations:
            iou = compute_iou(patch_bbox, annotation)
            if iou > max_iou:
                max_iou = iou

        if max_iou >= iou_threshold:
            positive_samples.append((patch, 1))
        else:
            negative_samples.append((patch, 0))
    # downsample the negative samples
    negative_samples = random.sample(negative_samples, len(positive_samples))
    return positive_samples, negative_samples

def read_json(path):
    with open(path, 'r') as f:
        return json.loads(f.read())
    


def compute_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Each bounding box is a list of [x1, y1, x2, y2] coordinates.
    """
    x1 = np.maximum(bbox1[0], bbox2[0])
    y1 = np.maximum(bbox1[1], bbox2[1])
    x2 = np.minimum(bbox1[2], bbox2[2])
    y2 = np.minimum(bbox1[3], bbox2[3])
    intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou

def non_maximum_suppression(detections, overlap_thresh):
    # if there are no detections, return an empty list
    if len(detections) == 0:
        return []
    
    # initialize the list of picked indices
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indices still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap between the bounding box
        # and the other bounding boxes
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked
    return detections[pick]
