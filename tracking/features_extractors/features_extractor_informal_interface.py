from __future__ import annotations
from typing import Tuple # this is needed to enable python to define return type aas current class

from core.cv2_image import CV2Image

class FeaturesExtractorInformalInterface:
    
    def __init__(self) -> None:
        self.features = []
    
    def extract(self, image:CV2Image) -> FeaturesExtractorInformalInterface:
        pass

    def show() -> None:
        pass