# OpenCV v.2 Python Wrappers

A couple of simple wrapping classes, that utilizes the Fluent API paradigm, and wraps and extend some of CV2 classes and functionalities.

Mainly made for a fast paste implementation and ease of use in another project that targeted object tracking in videos.

## Contents

- **CVImage**: Contains different functions for cropping, resizing, equalizing, sharpening, etc ... also drawing text and rectangles with alpha channel support.
- **Rectangle**: Represents a rectangle with different functionalities such as shifting, rescaling ect...
- **VideoFile & VideoFrame**: Represents a video file that one can iterate over its frames.
- **CVHOGFeatures**: Extends CV2 HOG features extractor, by adding a missing, but important, functionality, which is over block bin normalization.
