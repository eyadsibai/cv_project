import pickle
import cv2
import numpy as np
from glob import glob

from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression


import ast
import os
from sklearn.model_selection import GroupShuffleSplit
from feature_engineering import extract_features, create_vocabulary, create_histograms
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from config import GENERATED_IMAGES_PATH, RESOLUTION, model1,DOWNLOAD_PATH
import json
from metrics import compute_ap, compute_precision_recall

from utils import generate_samples


def remove_duplicates(X, y):
    X = np.array(X)
    y = np.array(y)
    Xy = np.hstack((X, y.reshape(-1, 1)))
    Xy = np.unique(Xy, axis=0)
    X = Xy[:, :-1]
    y = Xy[:, -1]
    return X, y

def _get_data_paths():
    return [
        os.path.dirname(path)
        for path in glob(GENERATED_IMAGES_PATH + "/" +DOWNLOAD_PATH+ "/**/*.npy", recursive=True)
    ]


def read_labels(path):
    with open(path, "r") as f:
        return ast.literal_eval(json.loads(f.read()))["bboxes"]


def read_data():
    paths = _get_data_paths()

    for path in tqdm(paths):
        for i in [1, 2]:
            label_path = os.path.join(path, f"{i}_{RESOLUTION}.mkv_bbox.json")
            video_path = os.path.join(path, f"{i}_{RESOLUTION}.mkv.npy")
            labels = read_labels(label_path)
            video = np.load(video_path)
            yield video, labels

def pickle1(o, name):
    with open(name, 'wb') as f:
        pickle.dump(o, f)

def unpickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def train_test_split(X, y, test_size, random_state, groupkey):
    X = np.array(X)
    y = np.array(y)
    # assume X is your feature data and y is your target data
    # groupkey is a list or array that identifies the group each sample belongs to
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Split the data into training and test sets
    train_indices, test_indices = next(gss.split(X, None, groupkey))
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def predict_video(video, model):
    predictions = []
    for frame in video:
        predictions.append(model.predict(frame))
    return np.array(predictions)


def preprocessing(video, image_enhancement=False):
    if image_enhancement:
        raise NotImplementedError
    return extract_features(video)

if __name__ == "__main__":

    print("Reading data...")
    videos = []
    labels = []
    for i, (video, labels_) in enumerate(read_data()):
        videos.append(video)
        labels.append(labels_)
        if i == 9:
            break

    print("size of videos", len(videos))
    print("size of labels", len(labels))

    print("Generating Samples...")

    # Generate samples for each image
    samples = []
    group_ids = []

    for i, (vid, label) in tqdm(enumerate(zip(videos, labels))):
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

    # X_train = unpickle("X_train.pkl")
    # y_train = unpickle("y_train.pkl")
    # X_test = unpickle("X_test.pkl")
    # y_test = unpickle("y_test.pkl")

    print("size of X_train", len(X_train))
    print("size of y_train", len(y_train))
    print("size of X_test", len(X_test))
    print("size of y_test", len(y_test))



    print("Extracting features...")
    extractor = cv2.SIFT_create()
    # Extract features from the training images using a keypoint descriptor
    X_train_features = list(extract_features(X_train, extractor=extractor))
    print("size of X_train_features", len(X_train_features))
    X_train_features = [e for e in X_train_features if len(e) != 0]
    X_treain_features_flat = [item for sublist in X_train_features for item in sublist]

    print("size of X_train_features", len(X_treain_features_flat))

    # Create a visual vocabulary using KMeans clustering
    kmeans = create_vocabulary(X_treain_features_flat, model1["num_clusters"])
    print("vocabulary created")
    # Create histograms of visual words for the training and testing images
    X_train = create_histograms(X_train, extractor, kmeans)
    X_test = create_histograms(X_test, extractor, kmeans)

    print("size of X_train", len(X_train))
    print("size of X_test", len(X_test))
    print("size of y_train", len(y_train))
    print("size of y_test", len(y_test))

    X_train, y_train = remove_duplicates(X_train, y_train)
    X_test, y_test = remove_duplicates(X_test, y_test)

    print("duplicates removed")
    print("size of X_train", len(X_train))
    print("size of X_test", len(X_test))
    print("size of y_train", len(y_train))
    print("size of y_test", len(y_test))


    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
])
    pipeline = pipeline.fit(X_train, y_train)

    predictions = pipeline.predict_proba(X_test)[:,1]

    precisions, recalls = compute_precision_recall(predictions, y_test,
                                                   iou_threshold=0.5)

    import pudb; pudb.set_trace()

    ap = compute_ap(precisions, recalls)


    # Check if the score is above the threshold
    if class_score > model1['score_threshold']:
        detections.append([x, y, class_score, class_idx])

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
