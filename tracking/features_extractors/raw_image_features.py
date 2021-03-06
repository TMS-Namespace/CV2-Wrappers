from __future__ import annotations
from typing import Tuple # this is needed to enable python to define return type aas current class

from core.cv_image import CVImage

class RawImageFeatures:
    
    def __init__(self) -> None:
        self.features = []
    
    def extract(self, image:CVImage) -> RawImageFeatures:
        # we just return the image as it is
        self.features = image.image_array
        return self

    def show() -> None:
        pass