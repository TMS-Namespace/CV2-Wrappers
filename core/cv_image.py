from __future__ import annotations
from typing import Tuple # this is needed to enable python to define return type aas current class

import cv2 as cv
import numpy as np
from rectangle import Rectangle

'''A class that represents a cv2 image, along with multiple functions 
that can be applied on it, it takes image path, as well as image array 
data as an input.

It holds two copies of the image, one will be manipulated, second one 
will be left uncached, so there will be possible to reset changes at 
any moment.

If compared to matlab, it should be always kept in mind that there is 
differences in color space, and decompression algorithms between matlab 
and cv2, for more see:
https://stackoverflow.com/questions/57889720/slight-differences-in-pixel-values-between-opencv-and-matlab'''
class CVImage:
    
    def __init__(self, image_source) -> None:
        # this will hold original image copy, and no changes is done on it
        self.original_image_array : np.array = None
        # represents image array after the manipulations
        self.image_array : np.array = None
        self.image_path = ''
        
        if type(image_source) is str:
            import os
            # check if file exists
            if not os.path.isfile(image_source):
                raise Exception(f"------ (error)\nvideo file\n{image_source}\nnot found.")
        
            res = cv.imread(image_source)
            if (len(res) == 0):
                raise Exception("------ (error)\nThe specified image can't be read.")
            else:
                self.original_image_array = res
                self.image_path = image_source
                print(f'------ (info)\nthe image:\n{self.image_path}\nloaded successfully.')
        else:
            self.original_image_array = image_source
            
        # make a copy, since this image will be manipulated
        self.image_array = self.original_image_array.copy()
        
        self._reset_size_info()
        
    def _reset_size_info(self):
        '''A help function to recalculate image size'''
        self.size = self.image_array.shape # size includes channels
        
        # if we have a gray image, there is no channels
        if len(self.size) == 3:
            self.height, self.width, self.channel_count = self.size
        else:
            self.channel_count = 1
            self.height, self.width = self.size
            
        self.dimensions = (self.height, self.width) # image size excluding channels
        
    def to_grayscale(self) -> CVImage:
        '''Converts image to gray scale.'''
        if not self.is_grayscale():
            self.image_array =  cv.cvtColor(self.image_array, cv.COLOR_BGR2GRAY)
            # update image size info
            self._reset_size_info()
        return self
        
    def to_size(self, size : Tuple) -> CVImage:
        '''Resizes the images to a specified size.'''
        # good to know https://www.reddit.com/r/MachineLearning/comments/qsl5jj/d_opinion_the_recent_paper_on_buggy_resizing/
        self.image_array = cv.resize(self.image_array, (size[0], size[1]) , interpolation  = cv.INTER_LINEAR)
        # update image size info
        self._reset_size_info()
        return self
    
    def rescale(self, factor : Tuple):
        '''Rescales the image by a specified factor.'''
        self.to_size((int(self.width*factor), int(self.height*factor)))
        return self
    
    def to_square(self) -> CVImage:
        '''Resizes the image to a square, according to the biggest dimension.'''
        size = max(self.height, self.width)
        self.to_size((size,size))
        return self
    
    def to_heat_map(self):
        '''Converts the image to a heat map.'''
        self.image_array = cv.applyColorMap(self.image_array, cv.COLORMAP_HOT)
        self._reset_size_info()
        return self
    
    def _crop_rectangle_to_image_boundaries(self, rect: Rectangle) -> Rectangle:
        ''' A help function to keep a given rectangle within the image boundaries.'''
        corrected_rect = [x for x in rect.tuple_standard]
        
        if rect.up_left_corner[0] < 0:
            corrected_rect[0] = 0
        if rect.up_left_corner[0] > self.width - 1:
            return None # the rectangle is totally out of the image boundaries
        
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
        
        return Rectangle(corrected_rect, is_ROI=False)
    
    def crop_to(self, rect:Rectangle) -> CVImage:
        '''Crops image to a specified rectangle.'''
        corr_rect = self._crop_rectangle_to_image_boundaries(rect)
        
        if (corr_rect != None):        
            # python slicing dose not include last index, so we add 1 
            # to keep resultant image width/height equal to rectangle's
            self.image_array =  self.image_array[corr_rect.y1 : (corr_rect.y2 + 1) , corr_rect.x1 : (corr_rect.x2 + 1)]
            self._reset_size_info()
            return self
    
    def copy(self) -> CVImage:
        '''Creates a copy of the current object.'''
        img = CVImage(self.image_array.copy())
        img.image_path = self.image_path
        return img
    
    def reset(self) -> CVImage:
        '''Resets image to original state, before any manipulations'''
        self.image_array = self.original_image_array.copy()
        self._reset_size_info()
        return self
    
    def save(self, path):
        '''Saves the image to the specified path.'''
        cv.imwrite(path, self.image_array)
        
    def show(self, pause = True):
        '''Displays the current image in a new window.'''
        cv.imshow("frame", self.image_array)    
        if (pause):
            cv.waitKey(0)

    def draw_polygon(self, coordinates, color = (107, 252, 3)) -> CVImage:
        """Takes the coordinates X1, Y1 ... XN, YN and plots the corresponding polygon"""
        pts = np.array(list(zip(coordinates[0::2], coordinates[1::2])), np.int32)

        self.image_array = cv.polylines(self.image_array, 
                                                    [pts.reshape((-1, 1, 2))],
                                                    True,
                                                    color,
                                                    2)
        return self

    def draw_rectangle(self, rect: Rectangle, color = (255, 0, 0, 0.6), thickness = 2) -> CVImage:
        """Draws a rectangle on the image, with alpha channel support"""
        # extract alpha and color
        alpha , color = self._get_color_alpha(color)
        # create overlay to implement alpha blending
        # ref: https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
        overlay = self.image_array.copy()
        cv.rectangle(overlay, 
                        rect.up_left_corner, 
                        rect.down_right_corner, 
                        color, 
                        thickness)
        output = self.image_array.copy()
        cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        self.image_array = output
        
        return self

    def _get_color_alpha(self, color) -> Tuple:
        '''Extract alpha channel information from color four-plet'''
        if np.size(color) == 4:
            return  (color[3], color[0:3])
        else:
            return (1, color)

    def draw_text(self, 
                text, 
                position, 
                font = cv.FONT_HERSHEY_SIMPLEX, 
                font_size = 0.4, 
                color = (0, 0, 255, 0.8), 
                thickness = 2) -> CVImage:
        """Draws text on the current frame image, with alpha channel support"""

        alpha , color = self._get_color_alpha(color)
        
        # if no alpha is specified, or ==1, just draw text
        if alpha == 1:
            cv.putText(self.image_array,
                                text,
                                position, 
                                font, 
                                font_size, 
                                color,
                                thickness)
        else:
            overlay = self.image_array.copy()
            cv.putText(overlay,
                                text,
                                position, 
                                font, 
                                font_size, 
                                color,
                                thickness)
            
            output = self.image_array.copy()
            cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
            self.image_array = output
        
        return self

    def draw_sub_image(self, sub_image:CVImage, location:Tuple) -> CVImage:
        '''Draws one image over another, and fuses channels if there there count mismatches.'''
        rect = Rectangle((location[0], location[1], sub_image.width, sub_image.height), is_ROI=True)
        rect = self._crop_rectangle_to_image_boundaries(rect)
        
        if rect!=None:
            if self.is_grayscale() == sub_image.is_grayscale():
                self.image_array[rect.y1:rect.y2, rect.x1:rect.x2] =  sub_image.image_array[0:(rect.height-1),0:(rect.width-1)]
            
            elif self.image_array.ndim == 3 and sub_image.image_array.ndim == 2:
                for i in range(self.image_array.shape[2]):
                    self.image_array[rect.y1:rect.y2, rect.x1:rect.x2, i] =  sub_image.image_array[0:(rect.height-1),0:(rect.width-1)]
            
            elif self.image_array.ndim == 2 and sub_image.image_array.ndim == 3:
                sub_image = sub_image.copy().to_grayscale()
                self.image_array[rect.y1:rect.y2, rect.x1:rect.x2] =  sub_image.image_array[0:(rect.height-1),0:(rect.width-1)]
                
            return self      
    
    def is_grayscale(self) -> bool:
        '''Returns true if the image has channels, or just gray scaled two dimensional array.'''
        return self.image_array.ndim != 3
    
    def equalize_histogram(self) -> CVImage:
        '''Equalizes the image'''
        # ref: https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
        if self.is_grayscale():
            self.image_array = cv.equalizeHist(self.image_array)
        else:
            self.image_array = cv.cvtColor(self.image_array, cv.COLOR_BGR2YCrCb)
            # equalize the histogram of the Y channel
            self.image_array[:, :, 0] = cv.equalizeHist(self.image_array[:, :, 0])
            # convert back to RGB color-space from YCrCb
            self.image_array = cv.cvtColor(self.image_array, cv.COLOR_YCrCb2BGR)
            
        return self

    def invert_image(self) -> CVImage:
        self.image_array = cv.addWeighted(self.image_array,-1.,self.image_array, 0,255)
        return self

    def sharpen_image(self) -> CVImage:
        '''Applies the standard sharpening filter.'''
        kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
        
        self.image_array = cv.filter2D(src = self.image_array, ddepth = -1, kernel = kernel)  
        
        return self
    def get_RGB_array(self) -> np.array:
        '''Returns Image array with standard RGB channel sequence.'''
        return cv.cvtColor(self.image_array, cv.COLOR_BGR2RGB)
    
    def get_PIL_image(self):
        '''Returns standard PIL image.'''
        from PIL import Image
        return Image.fromarray(self.get_RGB_array())

    def plot(self) -> None:
        '''Plots the image via matplotlib library.'''
        import matplotlib.pyplot as plt
        plt.imshow(self.get_PIL_image())
        plt.axis('Off')
        plt.show()

    def unsharp(self, strength = 1.0, threshold = 0, kernel_size = (5, 5), sigma = 1.0) -> CVImage:
        """Return a sharpened version of the image, using an unsharp mask.
        
         strength: is the amount of sharpening. 
         threshold: is the threshold for the low-contrast mask, or the pixels 
                    for which the difference between the input and blurred images
                    is less than threshold will remain unchanged.         
        """
        #  https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencvef:
        blurred = cv.GaussianBlur(self.image_array, kernel_size, sigma)
        
        sharpened = float(strength + 1) * self.image_array - float(strength) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        
        if threshold > 0:
            low_contrast_mask = np.absolute(self.image_array - blurred) < threshold
            np.copyto(sharpened, self.image_array, where=low_contrast_mask)
        
        self.image_array = sharpened
        
        return self

# --------- test
# CVImage(r'E:\System Folders\Documents\20220402_123525.jpg')\
#     .to_square()\
#         .unsharp()\
#             .to_grayscale()\
#                 .draw_rectangle(Rectangle((10,10,50,50)), (0,0,0,0.5))\
#                     .draw_text('bla bla', (30, 30))\
#                         .plot()