import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from find_car import *
from utils import *

if __name__=="__main__":
    car_obj=Find_Car('../../datasets/vehicle_detection')
    
    # Training
    #car_obj.train()
    
    # Running the pipeline
    if True:
        # Write Video with given file
        write_video(car_obj, './doc/project_video.mp4')

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