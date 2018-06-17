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

from utils import *

# Parameters for function calling
cspace = 'YCrCb'
orient = 8
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
size = (16, 16)
hist_bins = 32
hist_range = (0, 256)

class Find_Car(object):
    def __init__(self, folderToDataset):
        samples=8000
        self.car_images = glob.glob(folderToDataset + '/vehicles/**/*.png')[:samples]
        self.noncar_images = glob.glob(folderToDataset + '/non-vehicles/**/*.png')[:samples]
        self.clf=None
        self.rectList=[]
        
        # Analyse the input data
        print ("Input data")
        print ("Car Images:" + str(len(self.car_images)))
        print ("Non-Car Images:" + str(len(self.noncar_images)))

    ### TRAINING FUNCTIONS ###
    def train(self):
        """
        Main training function for coordinating the training
        """
        X_train, X_test, y_train, y_test=self.feature_extraction()
        self.train_classifier(X_train, y_train)
        self.predict(X_test, y_test)
        save_training_params(self.clf, self.scaler)
        return

    def feature_extraction(self):
        """
        Features will be extracted within class with given input and
        extract_features_imgs function
        """

        car_features = extract_features_imgs(self.car_images, cspace=cspace, orient=orient, 
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                hog_channel=hog_channel)
        notcar_features = extract_features_imgs(self.noncar_images, cspace=cspace, orient=orient, 
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                hog_channel=hog_channel)

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)  

        # Fit a per-column scaler - this will be necessary if combining different types of features (HOG + color_hist/bin_spatial)
        self.scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        X = self.scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=np.random.randint(0, 100))

        return X_train, X_test, y_train, y_test


    def train_classifier(self, X_train, y_train):
        """
        Train the classifier
        """
        # SVC as clf chosen
        self.clf = LinearSVC()
        
        # Train the clf
        self.clf.fit(X_train, y_train)

        return self.clf

    def predict(self, X_test, y_test):
        """
        Predict X_test
        """
        predictions=self.clf.predict(X_test)
        f1=f1_score(predictions,y_test)
        print('Test Accuracy of clf = ', round(f1, 4))
        

    ### PIPELINE FUNCTIONS ###
    def pipeline(self, img):
        """
        Overall pipeline to detect vehicles in the given img
        """
        rects=[]
        if self.clf==None:
            # Load clf and scaler from dict
            self.clf,self.scaler=load_training_params()

        # Sliding window size pairs
        pairs=[(350,500,0.8),(400,464,1.0),(416,480,1.0),(400,496,1.5),(432,496,1.5),(400,528,2.0),(400,528,2.0),(432,560,2.0),(400,596,3.5),(464,660,3.5),(464,660,4.0)]
        
        for pair in pairs:
            ystart,ystop,scale=pair
            rects.append(self.find_cars(img, ystart, ystop, scale, self.clf, self.scaler, orient, pix_per_cell, cell_per_block))
        
        rectangles = [item for sublist in rects for item in sublist] 
        
        heat = np.zeros_like(img[:,:,0]).astype(np.float)

        self.rectList.append(rectangles)
        for rectangles in self.rectList:
            heat=add_heat(heat,rectangles)
        
        # Remove first element
        if(len(self.rectList)>2):
            self.rectList = self.rectList[1:]
    
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,3)

        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)

        return draw_img
            
    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self,img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block):
        """
        Function that returns the rectangles for the detected cars.
        It is taken from the lessen and adapted.
        Here, the sliding window is realized (slide_window and search_windows is combined in this function)
        """
        rectangles = []
        
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient*cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
                # Extract the features on the subimg
                test_features=extract_feature(subimg)

                # Scale them with the same scaler from the input data
                test_features = self.scaler.transform(np.array([test_features]))
                
                test_prediction = clf.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    rect_tuple=tuple(( xbox_left, ytop_draw+ystart)) , tuple((xbox_left+win_draw,ytop_draw+win_draw+ystart))
                    rectangles.append(tuple(( rect_tuple  )) )
                    
        return rectangles