import cv2
import numpy as np
from sklearn.cluster import KMeans
import json
import random
import tqdm
import os
import ast
from glob import glob


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
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
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
    detections = non_maximum_suppression(detections, nms_threshold, window_size)

    # Draw bounding boxes for the detected objects
    for (x, y, score, class_idx) in detections:
        if class_idx == 1:
            x, y = int(x), int(y)
            image = cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 1)
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

# TODO: Check if it works correctly
def non_maximum_suppression(detections, iou_threshold, window_size):
    if len(detections) == 0:
        return []

    # Convert bounding boxes format to (x1, y1, x2, y2)
    detections = np.array(detections)
    boxes = np.array(
        [(x, y, x + window_size[0], y + window_size[1]) for (x, y) in detections[:, :2]]
    )
    scores = detections[:, 2]

    # Sort the detections by their scores
    idxs = np.argsort(scores)

    # Initialize the list of picked indexes
    picked = []

    while len(idxs) > 0:
        # Grab the last index in the idxs list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        picked.append(i)

        # Compute the IoU between the picked box and the remaining boxes
        iou = np.array([compute_iou(boxes[i], boxes[j]) for j in idxs[:last]])

        # Find the indexes of all boxes with an IoU less than the threshold
        to_remove = np.where(iou >= iou_threshold)[0]

        # Remove the overlapping boxes
        idxs = np.delete(idxs, np.concatenate(([last], to_remove)))

    return detections[picked]

def _get_data_paths():
    return [
        os.path.dirname(path)
        for path in glob(
             "output/**/*.npy", recursive=True
        )
    ]


def read_labels(path):
    with open(path, "r") as f:
        return ast.literal_eval(json.loads(f.read()))["bboxes"]


def read_data():
    paths = _get_data_paths()

    for path in paths:
        for i in [1, 2]:
            label_path = os.path.join(path, f"{i}_224p.mkv_bbox.json")
            video_path = os.path.join(path, f"{i}_224p.mkv.npy")
            labels = read_labels(label_path)
            video = np.load(video_path)
            yield video, labels
            
def remove_duplicates(X, y):
    X = np.array(X)
    y = np.array(y)
    Xy = np.hstack((X, y.reshape(-1, 1)))
    Xy = np.unique(Xy, axis=0)
    X = Xy[:, :-1]
    y = Xy[:, -1]
    return X, y

def plot_boxes(img, cords):
    
    for (x1, y1, x2, y2) in cords:
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # img = cv2.putText(img, f"Class: Player", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img