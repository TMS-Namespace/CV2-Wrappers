from __future__ import annotations # this is needed to enable python to define return type aas current class

import cv2
import os

import numpy as np
from numpy.lib.type_check import imag

from .rectangel import Rectangel

'''A class that represents a cv2 image, along with multiple functions 
that can be applied on it, it takes image path, as well as image array 
data as an input.
It saves two copies of the image, one will be manipulated, second one 
will be saved uncached, so it will be possible to revert at any moment.
If compared to matlab, it should be always kept in mind that there is differences in colour space, and decompression algorithms between the two, for more see:
https://stackoverflow.com/questions/57889720/slight-differences-in-pixel-values-between-opencv-and-matlab'''
class CV2Image:
    
    def __init__(self, image) -> None:
        # this will hold original image copy, and no changes is done on it
        self.original_image_array = []
        # represents image array after the manipulations
        self.image_array = []
        self.image_path = ''
        
        if type(image) is str:
            # check if file exists
            if not os.path.isfile(image):
                raise Exception(f"------ (error)\nvideo file\n{image}\nnot found.")
        
            res = cv2.imread(image)
            if (len(res) == 0):
                raise Exception("------ (error)\nThe specified image can't be read.")
            else:
                self.original_image_array = res
                self.image_path = image
                print(f'------ (info)\nthe image:\n{self.image_path}\nloaded successfully.')
        else:
            self.original_image_array = image
            
        # make a copy, so this is image that will be manipulated
        self.image_array = self.original_image_array.copy()
        
        self._reset_size_info()
        
    def _reset_size_info(self):
        '''just a help function to recalculate image size'''
        self.size = self.image_array.shape # size includes channels
        
        # if we have a gray image, there is no channels
        if len(self.size) == 3:
            self.height, self.width, self.channel_count = self.size
        else:
            self.channel_count = 1
            self.height, self.width = self.size
            
        self.dimensions = (self.height, self.width) # image size excluding channels
        
    def to_gray(self) -> CV2Image:
        if self.image_array.ndim > 2:
            self.image_array =  cv2.cvtColor(self.image_array, cv2.COLOR_BGR2GRAY)
            # update image size info
            self._reset_size_info()
        return self
        
    def to_size(self, size) -> CV2Image:
        # good to know https://www.reddit.com/r/MachineLearning/comments/qsl5jj/d_opinion_the_recent_paper_on_buggy_resizing/
        self.image_array = cv2.resize(self.image_array, (size[0], size[1]) , interpolation  = cv2.INTER_LINEAR)
        # update image size info
        self._reset_size_info()
        return self
    
    def rescale(self, factor):
        self.to_size((int(self.width*factor), int(self.height*factor)))
        return self
    
    def to_square(self) -> CV2Image:
        size = max(self.height, self.width)
        self.to_size((size,size))
        return self
    
    def to_heat_map(self): 
        self.image_array = cv2.applyColorMap(self.image_array, cv2.COLORMAP_HOT)
        self._reset_size_info()
        return self
    
    def _crop_rectangle_to_image_boundaries(self, rect: Rectangel) -> Rectangel:
        corrected_rect = [x for x in rect.tuple_standart]
        #corrected_rect.append(rect.tuple_standart)
        
        if rect.up_left_corner[0] < 0:
            corrected_rect[0] = 0
        if rect.up_left_corner[0] > self.width - 1:
            return None # the rectangale is totally out of the image boundaries
        
        if rect.up_left_corner[1] < 0:
            corrected_rect[1] = 0
        if rect.up_left_corner[1] > self.height - 1:
            return None
                
        if rect.down_right_corner[0] < 0:
            return None
        if rect.down_right_corner[0] > self.width - 1:
            corrected_rect[2] = self.height - 1
            
        if rect.down_right_corner[1] < 0:
            return None
        if rect.down_right_corner[1] > self.height - 1:
            corrected_rect[3] = self.height - 1
        
        return Rectangel(corrected_rect, is_ROI=False)
    
    def crop_to(self, rect:Rectangel) -> CV2Image:
        
        corr_rect = self._crop_rectangle_to_image_boundaries(rect)
        
        if (corr_rect != None):        
            # python slicing dose not include last index, so we add 1 
            # to keep resultant image width/height equal to rectangle's
            self.image_array =  self.image_array[corr_rect.y1 : (corr_rect.y2 + 1) , corr_rect.x1 : (corr_rect.x2 + 1)]
            self._reset_size_info()
            return self
    
    def copy(self) -> CV2Image:
        img = CV2Image(self.image_array.copy())
        img.image_path = self.image_path
        return img
    
    def reset(self) -> CV2Image:
        '''resets image to original state, before any manipulations'''
        self.image_array = self.original_image_array.copy()
        self._reset_size_info()
        return self
    
    def save(self, path):
        cv2.imwrite(path, self.image_array)
        
    def show(self, pause = True):
        '''dispales the modified image'''
        cv2.imshow("frame", self.image_array)    
        if (pause):
            cv2.waitKey(0)

    def draw_polygon(self, coordinate, color = (107, 252, 3)) -> CV2Image:
        """Takes the coordinates X1, Y1 ... XN, YN and plots the corresponding polygon"""
        pts = np.array(list(zip(coordinate[0::2], coordinate[1::2])), np.int32)

        self.image_array = cv2.polylines(self.image_array, 
                                                    [pts.reshape((-1, 1, 2))],
                                                    True,
                                                    color,
                                                    2)
        return self

    def draw_rectangle(self, rect: Rectangel, color = (255, 0, 0, 0.6), thickness = 2) -> CV2Image:
        """Takes a rectangle to be 
        plotted on current frame, with alpha channel support"""
        # extract alpha and color
        alpha , color = self._get_color_alpha(color)
        #create overlay to implement alpha blending
        # see the folowing if you want to understand
        # https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
        overlay = self.image_array.copy()
        cv2.rectangle(overlay, 
                        rect.up_left_corner, 
                        rect.down_right_corner, 
                        color, 
                        thickness)
        output = self.image_array.copy()
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        self.image_array = output
        
        return self

    def _get_color_alpha(self, color):
        '''extract alpha channel information from color four-plet'''
        if np.size(color) == 4:
            return  (color[3], color[0:3])
        else:
            return (1, color)

    def draw_text(self, 
                text, 
                position, 
                font = cv2.FONT_HERSHEY_SIMPLEX, 
                font_size = 0.4, 
                color = (0, 0, 255, 0.8), 
                thickness = 2) -> CV2Image:
        """draws text on the current frame image, with alpha channel support"""

        alpha , color = self._get_color_alpha(color)
        
        # if no alpha is specified, or ==1, just draw text
        if alpha == 1:
            cv2.putText(self.image_array,
                                text,
                                position, 
                                font, 
                                font_size, 
                                color,
                                thickness)
        else:
            overlay = self.image_array.copy()
            cv2.putText(overlay,
                                text,
                                position, 
                                font, 
                                font_size, 
                                color,
                                thickness)
            
            output = self.image_array.copy()
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
            self.image_array = output
        
        return self

    def draw_sub_image(self, sub_image:CV2Image, location):
        rect = Rectangel((location[0], location[1], sub_image.width, sub_image.height), is_ROI=True)
        rect = self._crop_rectangle_to_image_boundaries(rect)
        if rect!=None:
            if self.image_array.ndim == sub_image.image_array.ndim:
                self.image_array[rect.y1:rect.y2, rect.x1:rect.x2] =  sub_image.image_array[0:(rect.height-1),0:(rect.width-1)]
            elif self.image_array.ndim == 3 and sub_image.image_array.ndim == 2:
                for i in range(self.image_array.shape[2]):
                    self.image_array[rect.y1:rect.y2, rect.x1:rect.x2, i] =  sub_image.image_array[0:(rect.height-1),0:(rect.width-1)]
            elif self.image_array.ndim == 2 and sub_image.image_array.ndim == 3:
                sub_image = sub_image.copy().to_gray()
                self.image_array[rect.y1:rect.y2, rect.x1:rect.x2] =  sub_image.image_array[0:(rect.height-1),0:(rect.width-1)]
                
            return self        

# --------- test
# Image(r'D:\System Folders\Desktop\photo_2021-09-08_12-35-14 (2).jpg')\
#     .to_square()\
#         .to_gray()\
#             .draw_rectangle((10,10,50,50), (0,0,0,0.5))\
#                 .draw_text('bla bla', (30, 30))\
#                     .show()