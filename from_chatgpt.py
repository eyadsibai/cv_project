import cv2
import numpy as np
from sklearn.cluster import KMeans



def sliding_window(image, window_size, step):
    for y in range(0, image.shape[0] - window_size[1], step):
        for x in range(0, image.shape[1] - window_size[0], step):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def create_histograms(images, extractor, kmeans):
    histograms = []
    for img in images:
        keypoints, descriptors = extractor.detectAndCompute(img, None)
        histogram = np.zeros(len(kmeans.cluster_centers_))
        for desc in descriptors:
            idx = kmeans.predict([desc])[0]
            histogram[idx] += 1
        histograms.append(histogram)
    return histograms

def create_vocabulary(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(features)
    return kmeans

def extract_features(images, extractor):
    features = []
    for img in images:
        keypoints, descriptors = extractor.detectAndCompute(img, None)
        features.extend(descriptors)
    return features



def object_detection(image, window_size, step, extractor, kmeans, scaler, classifier, score_threshold, nms_threshold):
    # Initialize detection results
    detections = []

    # Perform sliding window
    for x, y, patch in sliding_window(image, window_size, step):
        # Extract features from the patch
        extract_features([patch], extractor)

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
            detections.append((x, y, class_score, class_idx))

    # Perform Non-Maximum Suppression (NMS)
    detections = non_maximum_suppression(detections, nms_threshold)

    # Draw bounding boxes for the detected objects
    for (x, y, score, class_idx) in detections:
        cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
        cv2.putText(image, f"Class: {class_idx}, Score: {score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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

    return positive_samples, negative_samples


if __name__ == "__main__":
  # Load and preprocess the dataset
    images, annotations = load_dataset()

    # Generate samples for each image
    samples = []
    labels = []
    iou_threshold = 0.5
    window_size = (64, 64)
    step = 16

    for img, anns in zip(images, annotations):
        positive_samples, negative_samples = generate_samples(img, anns, window_size, step, iou_threshold)
        samples.extend(positive_samples + negative_samples)
        labels.extend([1] * len(positive_samples) + [0] * len(negative_samples))

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)

    # Extract features from the training images using a keypoint descriptor
    extractor = cv2.SIFT_create()
    features = extract_features([s[0] for s in X_train], extractor)

    # Create a visual vocabulary using KMeans clustering
    num_clusters = 50
    kmeans = create_vocabulary(features, num_clusters)

    # Create histograms of visual words for the training and testing images
    train_histograms = create_histograms([s[0] for s in X_train], extractor, kmeans)
    test_histograms = create_histograms([s[0] for s in X_test], extractor, kmeans)

    # Normalize the histograms using the StandardScaler
    scaler = StandardScaler()
    train_histograms_normalized = scaler.fit_transform(train_histograms)
    test_histograms_normalized = scaler.transform(test_histograms)

    # Train a classifier (e.g., SVM) on the training set histograms
    classifier = train_classifier(train_histograms_normalized, y_train)

    # Evaluate the classifier on the test set histograms
    accuracy, conf_matrix = evaluate_classifier(classifier, test_histograms_normalized, y_test)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
