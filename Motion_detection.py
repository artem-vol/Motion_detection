import cv2
import numpy as np

# Load the video and read first and thirst frames
video = cv2.VideoCapture('232-video.mp4')
_, frame1 = video.read()
_, _ = video.read()
_, frame2 = video.read()

while video.isOpened():
    # Creating a foreground mask, highlighting moving objects
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=1, sigmaY=1)
    adaptive = cv2.adaptiveThreshold(blur, 120, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 7)
    kernel = np.ones((5, 5), np.uint8)
    image_closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=8)

    # Detecting contours of shapes using the resulting mask
    contours, _ = cv2.findContours(image_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Determining x and y coordinates, width and height of each contour
        if cv2.contourArea(contour) < 5000:  # Condition for removing noise
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('cam', frame1)
    frame1 = frame2
    _, _ = video.read()
    ret, frame2 = video.read()

    # The condition of closing a window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()