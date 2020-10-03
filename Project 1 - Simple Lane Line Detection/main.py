# Importing the required Libraries & Modules
import os
import sys
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from helper_functions import lane_line_pipeline, video_helper_function

def image_pipeline():
    """
    Function which calls the pipelines for detecting lane line in an image
    """
    paths = glob(os.path.join('data', 'images', '*.jpg'))

    for path in paths:
        image = mpimg.imread(path)

        lane_line_image = lane_line_pipeline(image)

        plt.imshow(lane_line_image)
        plt.show()    

def video_pipeline():
    """
    Function which calls the pipelines for detecting lane line in a video
    """
    paths = glob(os.path.join('data', 'videos', '*'))
    for vid_path in paths:
        video_helper_function(vid_path)

# Main Loop
if __name__ == "__main__":

    if sys.argv[1] == "-i":
        image_pipeline()
    
    elif sys.argv[1] == "-v":
        video_pipeline()