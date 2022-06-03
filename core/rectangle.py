from __future__ import annotations # this is needed to enable python to define return type as current class

from typing import List, Tuple

'''represent a rectangle, that is defined by the coordinates 
of top-left coroner (x1,y1), and bottom-right corner (x2,y2)
or, if is_ROI is set to true, it assumes (x, y, width, height)'''
class Rectangle:
    
    def __init__(self, data:List[int], is_ROI = True) -> None:
        
        self.x1 = data[0]
        self.y1 = data[1]
            
        if is_ROI == False:        
            self.x2 = data[2]
            self.y2 = data[3]
        else:
            self.x2 = data[0] + data[2] - 1
            self.y2 = data[1] + data[3] - 1
        
        self._set_info()
    
    def _set_info(self):
        '''just a help function to set class properties'''
        
        # dimensions
        self.width = self.x2 - self.x1 + 1
        self.height = self.y2 - self.y1 + 1
        self.size = (self.height, self.width)
        
        # rectangle center
        self.center_x = int(self.x1 + self.width / 2)
        self.center_y = int(self.y1 + self.height / 2)
        self.center = (self.center_x, self.center_y)
        
        # corner coordinates
        self.up_left_corner = (self.x1, self.y1) 
        self.up_right_corner = (self.x2, self.y1)
        self.down_left_corner = (self.x1, self.y2)
        self.down_right_corner = (self.x2, self.y2)
        
        # ROI format (x, y, width, height)
        self.tuple_ROI = (self.x1, self.y1, self.width, self.height)
        
        # standard format (x1, y1, x2, y2)
        self.tuple_standard = (self.x1, self.y1, self.x2, self.y2)
        
    def shift_center_to(self, new_center_x: int, new_center_y: int) -> Rectangle:
        '''shifts rectangle center to a new point'''
        
        # calc center deltas
        delta_x = new_center_x - self.center_x
        delta_y = new_center_y - self.center_y
        
        # change corners coordinates
        self.x1 += delta_x
        self.y1 += delta_y
        
        self.x2 += delta_x
        self.y2 += delta_y
        
        # recalc info
        self._set_info()
        
        return self
       
    def rescale(self, factor: float) -> Rectangle:
        '''rescales the rectangle, relative to its center'''
        # do nothing for scale 1
        if factor != 1:
            # calc how much we need to shift each corner
            x_diff = int(self.width * (factor - 1) / 2) 
            y_diff = int(self.height * (factor - 1) / 2) 
            
            # shift corners
            self.x1 -= x_diff
            self.y1 -= y_diff
            
            self.x2 += x_diff
            self.y2 += y_diff
            
            # recalc info
            self._set_info()
            
        return self
        
    def copy(self) -> Rectangle:
        '''creates a copy of th rectangle object'''
        return Rectangle(self.tuple_ROI)
        