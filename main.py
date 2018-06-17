import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
import pickle
import cv2
import glob
import time
import pickle
from sklearn.externals import joblib
from sklearn import ensemble
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from framework import extract_features_imgs,slide_window,bin_spatial,color_hist
from framework import get_hog_features, search_windows,draw_boxes,extract_feature
from framework import add_heat,apply_threshold,draw_labeled_bboxes

from find_car import *
from utils import *

if __name__=="__main__":
    car_obj=Find_Car('../../datasets/vehicle_detection')
    
    # Training
    car_obj.train()
    
    # Running the pipeline
    if True:
        # Write Video with given file
        write_video(car_obj, './project_video.mp4')

    if False:
        # Show the pipelined test_images to user
        test_imgList=glob.glob('./test_images/test*.jpg')

        for test_img in test_imgList:
            img=mpimg.imread(test_img)
            return_img=car_obj.pipeline(img)
            fig = plt.figure()
            plt.imshow(return_img)
            plt.show()
            
    print('The End.')