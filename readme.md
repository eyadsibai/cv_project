- download_data.py
- converter.py
- main.py

#########
1- Input data pipeline (later)
2- Send correct input to non_maximum_suppression function (Done)
3- Evaluate model (PR Curve, mAP, Acc) (Partially Done)
4- Train on multiple videos (Done)
X- Multiple window sizes?
5- Create one output (Video)
6- Sample of input data (challenges section)
#########



# Steps for image enhancement and post-processing
Preprocessing: Start by preprocessing the soccer-net data, converting it into grayscale, and then applying Gaussian blur to remove any noise in the image.

Edge Detection: Use edge detection techniques such as Canny Edge detection to detect edges in the image. This will help in identifying the boundaries of the players.

Contour Detection: After edge detection, use contour detection techniques to identify and extract the contours of the players. These contours will represent the boundaries of the players in the image.

Bounding Box Detection: After detecting the contours, use the minimum bounding box technique to draw a rectangle around the players. The minimum bounding box is the smallest rectangle that can enclose the contour of the player.

Filtering False Positives: Finally, filter out any false positives by removing any bounding boxes that are too small or too large in size or any boxes that overlap with other boxes.



# Ideas to Feature Extractiob

HOG (Histogram of Oriented Gradients): HOG captures the distribution of gradient directions (how the colors change) in an image. It divides the image into small regions called cells, computes histograms of gradient directions within each cell, and then normalizes them. HOG features can be used with classifiers like SVM (Support Vector Machines) for object detection.

Haar-like features: Haar-like features are simple rectangular features that can capture the difference in intensity between adjacent regions in an image. They are often used in cascade classifiers, like the popular Viola-Jones face detection algorithm. The cascade classifier is trained using a large number of positive and negative samples, and then it can detect objects in new images.

Deep learning-based features: Convolutional Neural Networks (CNNs) can automatically learn features from images that are useful for object detection. These networks consist of multiple layers that can learn a hierarchy of features, from low-level patterns to high-level object parts. Popular deep learning-based object detection frameworks include R-CNN, Fast R-CNN, Faster R-CNN, YOLO (You Only Look Once), and SSD (Single Shot MultiBox Detector).

Template matching: Template matching is a simple technique that can be used for object detection in cases where the object appearance doesn't change much. It involves sliding a template (a small image of the object) over the input image and computing a similarity score at each location. The locations with the highest similarity scores indicate the presence of the object.

Feature matching: You can use feature matching techniques, like those based on SIFT, SURF, or ORB, to find correspondences between keypoints in the input image and a set of known object images. This can help in detecting objects by finding a consistent set of matches between the input image and the known objects.
