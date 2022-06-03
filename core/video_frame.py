import numpy as np

from rectangle import Rectangle
from cv_image import CVImage

'''This class represents a video frame, along of the 
needed Gyro info, ground truth etc...'''
class VideoFrame():
    
    def __init__(self, video_file, frame_index: int, frame_image):
        
        self.video_file = video_file
        
        self.frame_index = frame_index
        self.image = CVImage(frame_image)

        self.frame_tick = -1

        self.angular_velocity= []
        self.header = []
        self.camera_sensor_info = []

        self.ground_truth_rectangle: Rectangle = None