# import xml.etree.ElementTree as ET
import json
from glob import glob
import numpy as np
import ast
import cv2

data_folder = 'output/data/england_epl/2016-2017/2016-08-20 - 19-30 Leicester 0 - 0 Arsenal'
video_path = glob(data_folder + '/*.npy')[0]
label_path = glob(data_folder + '/*.json')[0]

def read_json(path):
    with open(path, 'r') as f:
        return json.loads(f.read())

def crop_array(arr, coordinates_list):
    cropped_arrays = [arr[y1:y2, x1:x2] for x1,y1,x2,y2 in coordinates_list]
    return cropped_arrays

def get_players(video_path, label_path):
    '''
    Returns a list, each element contains list of boxes (players) in a frame
    '''
    video = np.load(video_path)
    file = read_json(label_path)
    labels = ast.literal_eval(file)
    
    one_game = []
    # crop all players on each frame and return a list of lists [frame1[cropped_players], frame2[...]]
    for frame, objects in zip(video, labels['bboxes']):
        # img_array = np.array(frame)
        cropped_arrays = crop_array(frame, objects)
        cropped_players = [cv2.cvtColor(array, cv2.COLOR_RGB2BGR) for array in cropped_arrays]
        one_game.append(cropped_players)
        
        # # for writing the images on a directory
        # for idx, img in enumerate(cropped_images):
        #     cv2.imwrite(f'results/output_{idx}.jpg', img)
    return one_game