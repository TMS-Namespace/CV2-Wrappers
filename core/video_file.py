import numpy as np
import pandas as pd
import os
import cv2
import os.path

from .video_frame import VideoFrame

'''This class, represents a video file, with iterable frames, and other needed properties'''
class VideoFile():

    def __init__(self, path):
        self.video_file_path =  path
        
        # holds frame index during iteration
        self._current_frame_index = -1
        # will save a temproray list of all frame objects
        self._frames_buffer = []  
        
        # check if file exists
        if not os.path.isfile(path):
            raise Exception(f"------ (error)\nvideo file\n{path}\nnot found.")

        try:
            self._cv2_video = cv2.VideoCapture(path)
            print(f'------ (info)\nthe video file:\n{path}\nis loaded successfully.')
            # load video dimensions
            self.height = self.get_frame(0).image.height
            self.width = self.get_frame(0).image.width  
        except Exception as exp:
            print(f'------ (error)\ncould not load the video file:\n{path}\nbecause:{exp}')
                

    def __iter__(self):
        # reset frame index
        self._current_frame_index = -1
        # rest cv2 reader position
        self._cv2_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self) -> VideoFrame:
        self._current_frame_index += 1

        # check if the frame already in our list
        frame = list(filter(lambda m: m.frame_index == self._current_frame_index, self._frames_buffer))
        
        if (len(frame) != 1):
            success, frame_image = self._cv2_video.read()
            if success:
                # create frame object if none
                new_frame = VideoFrame(self, self._current_frame_index, frame_image)
                self._frames_buffer.append((new_frame))
                return new_frame
            else:
                self._current_frame_index -= 1
                raise StopIteration
        else:
            return frame[0]

    def get_frame(self, frame_index) -> VideoFrame:
        '''finds and returns frame object by its index in the current video file'''
        # check if the frame already in our list
        frame = list(filter(lambda m: m.frame_index == frame_index, self._frames_buffer))
        if (len(frame) != 1):
            self._current_frame_index = frame_index
            # sweep cv2 iterator to new position
            self._cv2_video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame_image = self._cv2_video.read()
            if success:
                # create frame object if none
                new_frame = VideoFrame(self, self._current_frame_index, frame_image)
                self._frames_buffer.append((new_frame))
                return new_frame
            else:
                self._current_frame_index = -1
                raise Exception('wrong frame index.')
               
        else:
            return frame[0]

    def get_frame_count(self) -> int:
        '''counts the number of frames in a video file.'''
        return int(self._cv2_video.get(cv2.CAP_PROP_FRAME_COUNT))

    def clean_resource(self) -> None:
        '''frees cv2 and internal resources'''
        self._cv2_video.release()
        self._frames_buffer.clear()
        cv2.destroyAllWindows()