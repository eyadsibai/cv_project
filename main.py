import numpy as np
from glob import glob
from utils import generate_samples

import ast
import os
from sklearn.model_selection import GroupShuffleSplit
from feature_engineering import extract_features, create_vocabulary, create_histograms


from config import GENERATED_IMAGES_PATH, RESOLUTION, model1
import json


def _get_data_paths():
    return [
        os.path.dirname(path)
        for path in glob(GENERATED_IMAGES_PATH + "/**/*.npy", recursive=True)
    ]


def read_labels(path):
    with open(path, "r") as f:
        return ast.literal_eval(json.loads(f.read()))["bboxes"]


def read_data():
    paths = _get_data_paths()

    for path in paths:
        for i in [1, 2]:
            label_path = os.path.join(path, f"{i}_{RESOLUTION}.mkv_bbox.json")
            video_path = os.path.join(path, f"{i}_{RESOLUTION}.mkv.npy")
            labels = read_labels(label_path)
            video = np.load(video_path)
            yield video, labels


def train_test_split(X, y, test_size, random_state, groupkey):
    X = np.array(X)
    y = np.array(y)
    # assume X is your feature data and y is your target data
    # groupkey is a list or array that identifies the group each sample belongs to
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Split the data into training and test sets
    train_indices, test_indices = next(gss.split(X, None, groupkey))
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


if __name__ == "__main__":

    print("Reading data...")
    videos = []
    labels = []
    for i, (video, labels_) in enumerate(read_data()):
        videos.append(video)
        labels.append(labels_)
        if i == 5:
            break

    print("size of videos", len(videos))
    print("size of labels", len(labels))

    print("Generating Samples...")

    # Generate samples for each image
    samples = []
    group_ids = []

    for i, (vid, label) in enumerate(zip(videos, labels)):
        for frame, bboxes_per_frame in zip(vid, label):
            generated_samples = generate_samples(
                frame,
                bboxes_per_frame,
                window_size=model1["window_size"],
                step=model1["window_step"],
                iou_threshold=model1["iou_threshold"],
            )
            samples.extend(generated_samples)
            group_ids.extend([i] * len(generated_samples))

    print("size of samples", len(samples))
    print("size of group_ids", len(group_ids))

    X, y = zip(*samples)

    print('size of X', len(X))
    print('size of y', len(y))

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=42, groupkey=group_ids
    )


    print("size of X_train", len(X_train))
    print("size of y_train", len(y_train))
    print("size of X_test", len(X_test))
    print("size of y_test", len(y_test))

    print("Extracting features...")
    # Extract features from the training images using a keypoint descriptor
    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    print("size of X_train", len(X_train))
    print("size of X_test", len(X_test))
    # Create a visual vocabulary using KMeans clustering
    kmeans = create_vocabulary(X_train, model1["num_clusters"])

    # Create histograms of visual words for the training and testing images
    X_train = create_histograms(X_train, kmeans)
    X_test = create_histograms(X_test, kmeans)

    # # Normalize the histograms using the StandardScaler
    # scaler = StandardScaler()
    # train_histograms_normalized = scaler.fit_transform(train_histograms)
    # test_histograms_normalized = scaler.transform(test_histograms)

    # # Train a classifier (e.g., SVM) on the training set histograms
    # svc = SVC(kernel="linear", probability=True)
    # classifier = train_classifier(train_histograms_normalized, y_train, svc)

    # # Evaluate the classifier on the test set histograms
    # acc = evaluate_classifier(classifier, test_histograms_normalized, y_test)
    # print("Test Accuracy of SVC:", acc)

    # # Test on one image
    # image = video[4]
    # score_threshold = 0.9
    # nms_threshold = 0.7
    # image = object_detection(
    #     image,
    #     window_size,
    #     step,
    #     extractor,
    #     kmeans,
    #     scaler,
    #     classifier,
    #     score_threshold,
    #     nms_threshold,
    # )

    # cv2.imwrite("output/detection_example.jpg", image)
