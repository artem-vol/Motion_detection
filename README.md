# Motion detection
![GitHub top language](https://img.shields.io/github/languages/top/artem-vol/Motion_detection)

A program for motion detection on video and the implementation of its own multiple object tracking.

## Installation
```
git clone git@github.com:artem-vol/Motion_detection.git
```
## Usage
- motion_detection.py runs motion detection using OpenCV library functions.
- custom_motion_detection.py runs motion detection using native OpenCV function implementations in pure python.
- Motion_tracker.py starts multiple tracking of objects that are previously detected by the algorithm. motion_detection.py.
  
To change the video to which you want to apply this algorithm, change the line: ```video = cv2.VideoCapture(0)``` on ```video = cv2.VideoCapture('path/to/your/video.mp4')```.

## Examples
https://github.com/user-attachments/assets/d8cc2572-1083-462b-bd81-f9be96ed4bd3

