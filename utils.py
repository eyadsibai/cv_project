import cv2
import numpy as np

from feature_engineering import create_histograms


def sliding_window(image, window_size, step):
    for y in range(0, image.shape[0] - window_size[1], step):
        for x in range(0, image.shape[1] - window_size[0], step):
            yield (x, y, image[y : y + window_size[1], x : x + window_size[0]])


def object_detection(
    image,
    window_size,
    step,
    extractor,
    kmeans,
    model,
    score_threshold,
    nms_threshold,
):
    # Initialize detection results
    detections = []

    # Perform sliding window
    for x, y, patch in sliding_window(image, window_size, step):

        # Create a histogram of visual words
        patch_histogram = create_histograms([patch], extractor, kmeans)[0].reshape(1,-1)

        # Classify the patch
        probabilities = model.predict_proba(patch_histogram)[0]
        class_idx = np.argmax(probabilities)
        class_score = probabilities[class_idx]

        # Check if the score is above the threshold
        if class_score > score_threshold:
            detections.append([x, y, class_score, class_idx])

    # Perform Non-Maximum Suppression (NMS)
    detections = non_maximum_suppression(
        np.array(detections), nms_threshold, window_size
    )

    return detections


def draw_predictions(detections, window_size, image):

    # Draw bounding boxes for the detected objects
    for (x, y, score, class_idx) in detections:
        if class_idx == 1:
            image = cv2.rectangle(
                image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2
            )
            image = cv2.putText(
                image,
                f"Class: {class_idx}, Score: {score:.2f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
    return image


def generate_samples(video, list_bboxes, window_size, step, iou_threshold):
    samples = []

    for x, y, patch in sliding_window(video, window_size, step):
        patch_bbox = [x, y, x + window_size[0], y + window_size[1]]
        max_iou = 0

        for annotation in list_bboxes:
            max_iou = np.maximum(max_iou, compute_iou(patch_bbox, annotation))

        samples.append((patch, int(max_iou >= iou_threshold), patch_bbox))


    return samples


def compute_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Each bounding box is a list of [x1, y1, x2, y2] coordinates.
    """
    x1 = np.maximum(bbox1[0], bbox2[0])
    y1 = np.maximum(bbox1[1], bbox2[1])
    x2 = np.maximum(bbox1[2], bbox2[2])
    y2 = np.maximum(bbox1[3], bbox2[3])
    intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area

    return iou


def non_maximum_suppression(detections, iou_threshold, window_size):
    if len(detections) == 0:
        return []

    # Convert bounding boxes format to (x1, y1, x2, y2)
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
