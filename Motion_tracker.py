import cv2
import numpy as np


def scale_image(input_image: np.ndarray, desired_width: int) -> np.ndarray:
    """Scales the input image to the desired width while maintaining the aspect ratio.

    Args:
        input_image: The input image to resize.
        desired_width: A number representing the desired width of the output image.

    Returns:
        Scaled image with the desired width while preserving the aspect ratio.
    """
    aspect_ratio = desired_width / input_image.shape[1]
    desired_height = int(input_image.shape[0] * aspect_ratio)
    return cv2.resize(input_image, (desired_width, desired_height))


# Parameters for the Laplace Pyramid
lk_params = dict(winSize=(20, 20),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5))

video = cv2.VideoCapture('232-video.mp4')
frames = [video.read() for _ in range(3)]  # Write first three frames to list
frames = [(ret, scale_image(frame, 1200)) for ret, frame in frames]  # Scale frames
rectangles, contours, trajectories = [], [], []

# Main loop, reading continues until the video is open and the frame is read
while video.isOpened() and frames[-1][0]:
    img = frames[0][1].copy()

    # Creating a foreground mask, highlighting moving objects, between first and third frames
    diff = cv2.absdiff(frames[0][1], frames[2][1])
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=1, sigmaY=1)
    _, adaptive = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=7)

    # Detecting contours of shapes using the foreground mask
    f_contours, _ = cv2.findContours(image_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Variables that store the coordinates of squares and contours found in new frames
    new_contours, new_rectangles = [], []
    for contour in f_contours:
        x, y, w, h = cv2.boundingRect(contour)  # Determining x and y coordinates, width and height of each contour
        if cv2.contourArea(contour) < 5000:  # Condition for removing noise
            continue
        new_contours.append(contour)
        new_rectangles.append((x, y, w, h))

    # Detect new features
    frame_gray = cv2.cvtColor(frames[2][1], cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(frames[1][1], cv2.COLOR_BGR2GRAY)

    # If the list of old contours is empty - search for new trajectory points in the list of new contours
    if len(contours) != 0:
        track_rect, track_cont = rectangles, contours
    else:
        track_rect, track_cont = new_rectangles, new_contours
    for i in range(len(track_rect)):

        # Counting the number of end points of trajectories in a given contour
        p_in_rect = []
        x_rect, y_rect, w_rect, h_rect = track_rect[i]
        for trajectory in trajectories:
            x_t, y_t = trajectory[-1][0], trajectory[-1][1]
            if x_rect <= x_t <= x_rect + w_rect and y_rect <= y_t <= y_rect + h_rect:
                p_in_rect.append(trajectory)
        # If there are two - keep the one with the longer list length (assuming it is a dot, better for tracking)
        if len(p_in_rect) > 1:
            max_length = max([len(trajectory) for trajectory in p_in_rect])
            for trajectory in p_in_rect:
                if len(trajectory) < max_length:
                    trajectories.remove(trajectory)
                    p_in_rect.remove(trajectory)
                    # If there are any traced trajectories left in the contour, then delete the last one
                    if len(p_in_rect) > 1:
                        trajectories.remove(p_in_rect[-1])
                        p_in_rect.remove(p_in_rect[-1])
        # If there is one point, then there is no need to look for a new point
        if len(p_in_rect) != 0:
            continue

        mask = np.zeros_like(frame_gray)  # Region of interest where a new point needs to be found
        cv2.fillPoly(mask, track_cont[i], (255, 0, 0))

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, maxCorners=1, qualityLevel=0.1, minDistance=1, blockSize=5)
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, statuses, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 30

        # Get all the trajectories
        new_trajectories = []
        for trajectory, (x, y), good_flag, status in zip(trajectories, p1.reshape(-1, 2), good, statuses):

            # If the distance between two points and points is not found - skip
            if not good_flag or not status:
                continue
            trajectory.append((x, y))

            # Check maximum length of trajectory
            if len(trajectory) > 50:
                del trajectory[0]
            new_trajectories.append(trajectory)

            # Checking if a point is in a new rectangles
            point_in_new = False
            for n_rect in new_rectangles:
                if n_rect[0] <= x <= n_rect[0] + n_rect[2] and n_rect[1] <= y <= n_rect[1] + n_rect[3]:
                    point_in_new = True
                if point_in_new:
                    break
            # if it is not there in the new rectangle, we will look for it in the old and add it to the new one
            if not point_in_new:
                for i, rect in enumerate(rectangles):
                    if rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]:
                        new_rectangles.append(rect)
                        new_contours.append(contours[i])
                        break

            # Draw the newest detected point
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        # Assign new values
        rectangles = new_rectangles
        contours = new_contours
        trajectories = new_trajectories

        # Draw all detected rectangles
        for rect in rectangles:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)

        # Draw all the trajectories
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 0, 255))

    cv2.imshow('Camera', img)  # Show Results
    frames.pop(0)  # Delete the old frame
    frames.append(video.read())  # Read the new frame

    # Exit if frame is not read
    if not frames[-1][0]:
        break

    # The condition of closing a window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frames[-1] = (frames[-1][0], scale_image(frames[-1][1], frames[0][1].shape[1]))  # Scale new frame

cv2.destroyAllWindows()
video.release()
