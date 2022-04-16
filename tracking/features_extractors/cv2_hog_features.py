from __future__ import annotations
from typing import Tuple # this is needed to enable python to define return type aas current class

import cv2
import numpy as np

from core.cv2_image import CV2Image
from .features_extractor_informal_interface import FeaturesExtractorInformalInterface

'''Implements HOG feature extractor from OpenCV, that 
is, alike skikit library, dose include voting system. 
see https://stackoverflow.com/questions/28390614/opencv-hogdescripter-python?noredirect=1&lq=1
also, here we implement "over blocks, bin normalization"
Note: all sizes should be in (width x height), since cv2 uses this format'''
class CV2HOGFeatures(FeaturesExtractorInformalInterface):
    
    def __init__(self) -> None:
        
        self.cell_size_in_pixels = (8, 8)
        self.block_size_in_cells = (2, 2)
        self.stride_in_cells = (1, 1)
                
        self.orientations_bins_count = 9

        self.normalization_timming_threshold = 0.2
        self.gaussian_smoothing_sigma = -1
        self.padding = (0, 0)

        self._descriptor = None
        self.features = []

        self._block_size_in_pixels = (-1, -1)
        self._stride_in_pixels = (-1, -1)

        self._image_size_in_cells = (-1, -1)
        self._cropped_image_to_cells_size_in_pixels = (-1, -1)

    def _create_descriptor(self, image_array) -> None:
        '''calc the correct parameters for discriptor and creates 
        it, there is no need to calc it for every image.'''
        self._block_size_in_pixels = (self.block_size_in_cells[0] * self.cell_size_in_pixels[0],
                                      self.block_size_in_cells[1] * self.cell_size_in_pixels[1])

        self._stride_in_pixels = (self.stride_in_cells[0] * self.cell_size_in_pixels[0],
                                  self.stride_in_cells[1] * self.cell_size_in_pixels[1])

        # remember to use cv2 width x height
        self._image_size_in_cells = (image_array.shape[1] // self.cell_size_in_pixels[0],
                                     image_array.shape[0] // self.cell_size_in_pixels[1])

        self._cropped_image_to_cells_size_in_pixels = (self._image_size_in_cells[0] * self.cell_size_in_pixels[0],
                                                       self._image_size_in_cells[1] * self.cell_size_in_pixels[1])
        # see https://docs.opencv.org/4.5.3/d5/d33/structcv_1_1HOGDescriptor.html
        self._descriptor = cv2.HOGDescriptor(self._cropped_image_to_cells_size_in_pixels,
                                             self._block_size_in_pixels,
                                             self._stride_in_pixels,
                                             self.cell_size_in_pixels,
                                             self.orientations_bins_count,
                                             1, # = derivAperture 
                                             self.gaussian_smoothing_sigma)
        
    def _get_image_covering_blocks_count(self, dimension):
        '''calcs the number of overlaping blocks count that will cover 
        the image, for an arbitrary image, cell, block, and stride seizes.'''
        blocks = int(self._image_size_in_cells[dimension] / self.stride_in_cells[dimension])
        # this is a special case when stride is one
        if (blocks == self._image_size_in_cells[dimension]):
            blocks = blocks - self.block_size_in_cells[dimension] + 1
            
        return blocks
    
    def _get_per_block_features(self, image_array):
        
        if self._descriptor == None:
            self._create_descriptor(image_array)

        # this will get us per block, per cell our bins, so we need to reshape the histogram.
        # then transpose to make it of a stadart indexing, i.e. by rows 
        # see https://stackoverflow.com/questions/22373707/why-does-opencvs-hog-descriptor-return-so-many-values
        return self._descriptor\
                                .compute(image_array,  self._stride_in_pixels, self.padding)\
                                .reshape(self._get_image_covering_blocks_count(0),
                                        self._get_image_covering_blocks_count(1),
                                        self.block_size_in_cells[0],
                                        self.block_size_in_cells[1],
                                        self.orientations_bins_count)\
                                .transpose((1, 0, 2, 3, 4))

                                
    def get_normalized_cell_features(self, image_array) -> CV2HOGFeatures:
        '''- returns returns per block normalized cell histogram.
        - When called multiple times, it assumes that every call is done 
        for same image size, and class parameters.
        - If the provided image is not grey, OpenCV will make it grey automatically,
        but in a particular way, see:
        https://answers.opencv.org/question/194218/applying-hog-on-bgr-vs-applying-hog-on-rgb-images/'''
        
        # get block features    
        block_histogram = self._get_per_block_features(image_array)
        
            
        # we normalize the histogram over the blocks
        # for description, see https://learnopencv.com/histogram-of-oriented-gradients/
        
        cell_histogram = np.zeros((self._image_size_in_cells[1], 
                                    self._image_size_in_cells[0], 
                                    self.orientations_bins_count))

        # count cells according to how many times they are repeated within overlaping boxes
        cell_repetition_count = np.zeros((self._image_size_in_cells[1], 
                                            self._image_size_in_cells[0], 
                                            1))


        # this can be imagined as if we sliding two copies of the obtained above 
        # block histogram grids, by the size of the blocks, and summing the bins 
        # of the cells that coinside, in the same time, we count how many times 
        # each cell is repeated (which differs, due to block overlaping, 
        # especially for border cells), to finally divide the sum by this count.
        # inspired by https://rsdharra.com/blog/lesson/25.html
        for y_cell_block_index in range(self.block_size_in_cells[1]):
            for x_cell_block_index in range(self.block_size_in_cells[0]):
                
                last_block_cell_index_similar_to_y = self._image_size_in_cells[1] \
                                                    - self.block_size_in_cells[1] \
                                                    + y_cell_block_index \
                                                    + 1
                last_block_cell_index_similar_to_x = self._image_size_in_cells[0] \
                                                    - self.block_size_in_cells[0] \
                                                    + x_cell_block_index \
                                                    + 1
                
                cell_histogram[y_cell_block_index:last_block_cell_index_similar_to_y,
                                x_cell_block_index:last_block_cell_index_similar_to_x,
                                :] += block_histogram[:,
                                                        :,
                                                        y_cell_block_index,
                                                        x_cell_block_index,
                                                        :]
                    
                cell_repetition_count[y_cell_block_index:last_block_cell_index_similar_to_y,
                                        x_cell_block_index:last_block_cell_index_similar_to_x,
                                        0] += 1

        # find the average of cell's bins
        cell_histogram /= cell_repetition_count
        
        # the result is actually inverted along y axis, so we fix that
        self.features = np.flip(cell_histogram, axis=0)
        
        return self
        
    def extract(self, image:CV2Image) -> CV2HOGFeatures:
        '''a help function to unify feature extraction interface'''
        return self.get_normalized_cell_features(image.image_array)

    # def feature_space_size(self) -> Tuple[int]:
    #     '''a help function to unify features interface, returns cells count as (height x width)'''
    #     return (self._image_size_in_cells[1], self._image_size_in_cells[0])
    
    # def feature_space_size(self) -> int:
    #     '''a help function to unify features interface, returns cells count as (height x width)'''
    #     return 5555
    
    def show(self, orientation_bin_index = 3) -> None:
        import matplotlib.pyplot as plt

        # plot a map for a particular bin
        plt.title(f'cv2 HOG normalized feature histogram\nfor the bin = {orientation_bin_index}\nfeature space size is = {self.features.shape}')
        plt.pcolor(self.features[:, :, orientation_bin_index])
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.colorbar()
        plt.show()

# --------- test

# from image import Image

# image = Image(r'D:\System Folders\Desktop\photo_2021-09-08_12-35-14 (2).jpg')
# CV2HOGFeatures().extract(image).show()