import numpy as np
import pandas as pd
import os
import cv2


from .rectangel import Rectangel
from .cv2_image import CV2Image

'''This class represents a video frame, along of the 
needed Gyro info, ground truth etc...'''
class VideoFrame():
    
    def __init__(self, video_file, frame_index: int, frame_image):
        
        self.video_file = video_file
        
        self.frame_index = frame_index
        self.image = CV2Image(frame_image)

        self.frame_tick = -1

        self.angular_velocity= []
        self.header = []
        self.camera_sensor_info = []

        self.ground_truth_rectangel: Rectangel = None