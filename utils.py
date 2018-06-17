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

def save_training_params(clf, scaler):
    """
    Saves the parameters clf and scaler to the dictionary dict.p
    """
    dict={}
    dict["clf"]=clf
    dict["scaler"]=scaler
    pickle.dump(dict, open("dict.p","wb"))
    return True

def load_training_params():
    """
    Load the parameters that are saved into dict.p
    Returns clf and scaler
    """
    load_dict = pickle.load(open( "dict.p", "rb" ) )
    clf=load_dict["clf"]
    scaler=load_dict["scaler"]
    return clf,scaler

def write_video(Find_Carobj, inputVideo):
    """
    Function to record video
    """
    test_out_file = 'written_video.mp4'
    input_video = VideoFileClip(inputVideo)
    clip_test_out = input_video.fl_image(Find_Carobj.pipeline)
    clip_test_out.write_videofile(test_out_file, audio=False)
    return