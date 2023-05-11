import cv2
import numpy as np
from glob import glob
import os
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from train import train_classifier, evaluate_classifier, evaluate_object_detection
from utils import *
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    print("Loading Data ..")
    videos = []
    labels = []
    for i, (video, labels_) in enumerate(read_data()):
        videos.extend(video)
        labels.extend(labels_)
        # if i == 10:
        #     break

    images = videos
    annotations = labels
    # Generate samples for each image
    samples = []
    labels = []
    iou_threshold = 0.5
    window_size = (20, 45)
    num_clusters = 15
    step = 11

    print("Generating Samples ..")
    for img, anns in zip(images, annotations):
        positive_samples, negative_samples = generate_samples(img, anns, window_size, step, iou_threshold)
        samples.extend(positive_samples + negative_samples)
        labels.extend([1] * len(positive_samples) + [0] * len(negative_samples))

    print("Splitting data ..")
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)
    
    print("Creating Features ..")
    # Extract features from the training images using a keypoint descriptor
    extractor = cv2.SIFT_create()
    features = extract_features([s[0] for s in X_train], extractor)

    # Create a visual vocabulary using KMeans clustering
    print("Creating Vocabulary ..")
    kmeans = create_vocabulary(features, num_clusters)
    
    print("Creating Histograms ..")
    # Create histograms of visual words for the training and testing images
    train_histograms = create_histograms([s[0] for s in X_train], extractor, kmeans)
    test_histograms = create_histograms([s[0] for s in X_test], extractor, kmeans)
    
    print("Removing Duplicates ..")
    train_histograms, y_train = remove_duplicates(train_histograms, y_train)
    
    print("Normalizing data ..")
    # Normalize the histograms using the StandardScaler
    scaler = StandardScaler()
    train_histograms_normalized = scaler.fit_transform(train_histograms)
    test_histograms_normalized = scaler.transform(test_histograms)
    
    print("Training Classifier ..")
    # Train a classifier (e.g., SVM) on the training set histograms
    svc = SVC(kernel='linear', class_weight="balanced", probability=True)
    classifier = train_classifier(train_histograms_normalized, y_train, svc)

    # Evaluate the classifier on the test set histograms
    acc = evaluate_classifier(classifier, test_histograms_normalized, y_test)
    print("Test Accuracy of SVC:", acc)
    
    print("Calculating Confusion Matrix ..")
    cm_img = ConfusionMatrixDisplay.from_estimator(svc, test_histograms_normalized, y_test)
    plt.savefig("output/results/confusion_matrix.jpg")

    
    
    print("Classification Report ..")
    y_preds = svc.predict(test_histograms_normalized)
    print(classification_report(y_test, y_preds))
    
    print("Claculating PR Curve ..")
    y_probs = svc.predict_proba(test_histograms_normalized)
    display = PrecisionRecallDisplay.from_predictions(y_test, y_probs[:, 1], name="LinearSVC")
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    plt.savefig("output/results/PR Curve.jpg")
    
    print("Generating Sample output ..")
    for i in range(0, 100, 10):    
        # Evalutations
        input_image = images[i]
        input_label = annotations[i]
        input2_image = input_image.copy()
        score_threshold = 0.25
        nms_threshold = 0.1
        
        # Test on one image
        gt_image = plot_boxes(input_image, input_label)
        output_image = object_detection(input2_image, window_size, step, 
                                extractor, kmeans, scaler, 
                                classifier, score_threshold, 
                                nms_threshold)
        
        cv2.imwrite(f'output/results/gt_example{i}.jpg', gt_image)
        cv2.imwrite(f'output/results/output_example{i}.jpg', output_image)
        
    print("Done :)")

