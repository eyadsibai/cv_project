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
import ast

if __name__ == '__main__':
    
    # # Get path names
    # data_folder = 'output/data/england_epl/2016-2017/2016-08-20 - 19-30 Leicester 0 - 0 Arsenal'
    # video_path = glob(data_folder + '/*.npy')[0]
    # label_path = glob(data_folder + '/*.json')[0]
    
    # # Load and preprocess the dataset
    # video = np.load(video_path)
    # file = read_json(label_path)
    # labels = ast.literal_eval(file)
    # images, annotations = video, labels
    print("Loading Data ..")
    videos = []
    labels = []
    for i, (video, labels_) in enumerate(read_data()):
        videos.extend(video)
        labels.extend(labels_)
        # if i == 40:
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

    print("Splitting data")
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)
    
    print("Creating Features")
    # Extract features from the training images using a keypoint descriptor
    extractor = cv2.SIFT_create()
    features = extract_features([s[0] for s in X_train], extractor)

    # Create a visual vocabulary using KMeans clustering
    print("Creating Vocabulary")
    kmeans = create_vocabulary(features, num_clusters)
    
    print("Creating Histograms")
    # Create histograms of visual words for the training and testing images
    train_histograms = create_histograms([s[0] for s in X_train], extractor, kmeans)
    test_histograms = create_histograms([s[0] for s in X_test], extractor, kmeans)
    
    print("Normalizing data")
    # Normalize the histograms using the StandardScaler
    scaler = StandardScaler()
    train_histograms_normalized = scaler.fit_transform(train_histograms)
    test_histograms_normalized = scaler.transform(test_histograms)
    
    print("Training Classifier")
    # Train a classifier (e.g., SVM) on the training set histograms
    svc = SVC(kernel='linear',probability=True)
    classifier = train_classifier(train_histograms_normalized, y_train, svc)

    # Evaluate the classifier on the test set histograms
    acc = evaluate_classifier(classifier, test_histograms_normalized, y_test)
    print("Test Accuracy of SVC:", acc)
    
    print("Generating Sample output")
    for i in range(0, 100, 10):    
        # Evalutations
        input_image = images[i]
        input_label = annotations[i]
        input2_image = input_image.copy()
        score_threshold = 0.25
        nms_threshold = 0.2
        # TODO: Fix evaluate object detection
        # results = evaluate_object_detection(input2_image, input_label,
        #                           window_size, step, 
        #                           extractor, kmeans, scaler, 
        #                           classifier, score_threshold, 
        #                           nms_threshold)
        
        # Test on one image
        # TODO: Create function for displaying annotation
        ##
        output_image = object_detection(input2_image, window_size, step, 
                                extractor, kmeans, scaler, 
                                classifier, score_threshold, 
                                nms_threshold)
        cv2.imwrite(f'output/results/input_example{i}.jpg', input_image)
        cv2.imwrite(f'output/results/output_example{i}.jpg', output_image)

