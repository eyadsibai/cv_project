import time
import cv2
from utils import sliding_window, create_histograms, non_maximum_suppression
import numpy as np
from metrics import compute_precision_recall, compute_ap

def train_classifier(X_train, y_train, svc):
    '''
    Train SVC
    '''
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    return svc

def evaluate_classifier(svc, X_test, y_test):
    '''
    Evaluate SVC
    '''
    return round(svc.score(X_test, y_test), 4)

def evaluate_object_detection(image, annotations,
                              window_size, step, 
                              extractor, kmeans, scaler, 
                              classifier, score_threshold, 
                              nms_threshold):
    # Alter ground truth format
    # ground_truth = []
    # for x1, y1, _, _ in annotations:
    #     ground_truth.append(x1, y1, None, 1)
        
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
    
    # Format detection to bboxes
    detected_bboxes = []
    for x, y, class_score, _ in detections:
        detected_bboxes.append([x, y, x + window_size[0], y + window_size[1], class_score])
    

    precision, recall = compute_precision_recall(detected_bboxes, annotations,
                                                 iou_threshold=0.5)
    
    map = compute_ap(precision, recall, detected_bboxes)
    print(map)
    print(f'{precision=}')
    print(f'{recall=}')
    return image
    

