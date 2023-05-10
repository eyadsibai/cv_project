import time
import cv2
from utils import sliding_window, create_histograms, non_maximum_suppression

def train_classifier(X_train, y_train, svc):
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	return svc

def evaluate_classifier(svc, X_test, y_test):
    return round(svc.score(X_test, y_test), 4)

