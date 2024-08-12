from typing import Any
import numpy as np
import cv2
from numba import njit


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


def abs_difference(input_array1: np.ndarray, input_array2: np.ndarray) -> np.uint8:
    """Calculates the per-element absolute difference between two arrays.

    Args:
        input_array1: The first array.
        input_array2: The second array.

    Returns:
        Absolute difference between first and second array.
    """
    input_array1, input_array2 = input_array1.astype('int8'), input_array2.astype('int8')
    return np.abs(input_array1 - input_array2).astype('uint8')


@njit
def convert_to_grayscale(input_image: np.uint8) -> np.uint8:
    """Convert RGB channels image to grayscale.

    Args:
        input_image: Three-channel RGB image.

    Returns:
        Input image converted to grayscale.
    """
    input_image = input_image.copy().astype(np.int8)
    input_image = input_image[:, :, 0] * 0.0722 + input_image[:, :, 1] * 0.7152 + input_image[:, :, 2] * 0.2126
    return input_image.astype('uint8')


@njit(inline='always')
def add_black_padding(input_array: [np.ndarray, np.bool_], kernel_size: int) -> np.ndarray:
    """Adds black padding around the input array.

    Adds a black padding around the input array so that after applying a
    convolutional kernel with the given size and stride 1, the output array
    has the same shape as the input array.

    Args:
        input_array: The input array to which padding will be added.
        kernel_size: The size of the kernel for which padding needs to be added.

    Returns:
        Input array with black padding added around it.
    """
    padding_size = kernel_size // 2
    padded_array = np.zeros((input_array.shape[0] + padding_size * 2, input_array.shape[1] + padding_size * 2),
                            dtype=input_array.dtype)
    padded_array[padding_size:padding_size + input_array.shape[0], padding_size:padding_size + input_array.shape[1]] = (
        input_array)
    return padded_array


def fast_gaussian_conv(input_array: np.uint8, iterations) -> np.uint8:
    """Apply fast Gaussian convolution on an input numpy array for a specified number of iterations.

    The function is called "Fast Gaussian Convolution" because instead of
    performing the convolution directly (iteratively), it uses a weighted
    average of image slices that are offset from each other.

    Args:
        input_array: Array to which convolution will be added.
        iterations: Number of iterations to perform the Gaussian convolution.

    Returns:
        Array after applying fast Gaussian convolution for the specified number of iterations.
    """
    output_array = input_array.copy()
    for _ in range(iterations):
        output_array[1:-1] = output_array[1:-1] // 2 + output_array[:-2] // 4 + output_array[2:] // 4
        output_array[:, 1:-1] = output_array[:, 1:-1] // 2 + output_array[:, :-2] // 4 + output_array[:, 2:] // 4
    return output_array


def simple_thresh(input_array: np.uint8, thresh: int, max_val: int) -> np.uint8:
    """Applies a fixed-level threshold to each element of the array.

    If the pixel value is less than the threshold, it is set to 0,
    otherwise it is set to the maximum value.

    Args:
        input_array: Array to which simple threshold will be added.
        thresh: threshold value.
        max_val: value which will be set if value more than threshold

    Returns:
        The computed threshold value array
    """
    indexes = input_array > thresh
    return np.uint8(indexes * max_val)


def convert_to_binary(input_array: np.uint8) -> np.bool_ | np.uint8:
    """ Convert the input numpy array to a binary array"""
    if input_array.dtype != bool:
        input_array = np.bool_(input_array > 0)
    return input_array


@njit(cache=True, inline='always')
def erode(input_array: np.ndarray[Any, np.dtype[np.bool_]], kernel: np.ndarray) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Erode an input binary image using a given kernel.

    The kernel slides over the image (like in 2D convolution). A pixel in the original image (either 1 or 0)
    will be considered condition 1 only if all pixels under the original condition are 1, otherwise it is erased.

    Args:
        input_array: Input binary numpy array representing the image.
        kernel: Binary array representing the structuring element.

    Returns:
        Binary numpy array representing the eroded image after applying the erosion operation using the given kernel.
    """
    out_shape = input_array.shape
    input_array = add_black_padding(input_array, kernel.shape[0])  # Apply this to save input dimensions
    out_array = np.zeros(out_shape, dtype=np.bool_)
    for row in range(out_shape[0]):
        for col in range(out_shape[1]):
            match = True
            for k_row in range(kernel.shape[0]):
                for k_col in range(kernel.shape[1]):
                    # Check if all elements of the kernel match the elements of the input array
                    if input_array[row + k_row, col + k_col] != kernel[k_row, k_col]:
                        match = False
                        break
                if not match:
                    break
            out_array[row, col] = match
    return out_array


@njit(cache=True, inline='always')
def dilate(input_array: np.bool_, kernel: np.ndarray) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Dilate an input binary image using a given kernel.

    The kernel slides over the image (as in a 2D convolution). A pixel in the original image (either 1 or 0) will
    be considered condition 1 if at least one pixel in the original state is 1, otherwise it is erased.

    Args:
        input_array: Input binary numpy array representing the image.
        kernel: Binary numpy array representing the structuring element.

    Returns:
        Binary numpy array representing the dilated image after applying the dilation operation using the given kernel.
    """
    out_shape = input_array.shape
    input_array = add_black_padding(input_array, kernel.shape[0])  # Apply this to save input dimensions
    out_array = np.zeros(out_shape, dtype=np.bool_)
    for row in range(out_shape[0]):
        for col in range(out_shape[1]):
            found = False
            for kr in range(kernel.shape[0]):
                for kc in range(kernel.shape[1]):
                    # We check if there is at least one match between the kernel and the elements of the input array
                    if input_array[row + kr, col + kc] and kernel[kr, kc]:
                        found = True
                        break
                if found:
                    break
            out_array[row, col] = found
    return out_array


def close(input_binary_array: np.bool_, kernel: np.ndarray) -> np.ndarray:
    """Sequential application of dilation and erosion in a single image with a given kernel"""
    closed_array = dilate(input_binary_array, kernel)
    closed_array = erode(closed_array, kernel)
    return closed_array


def morphology_ex(input_array: np.bool_, op, kernel: np.ndarray, iterations: int) -> np.ndarray:
    """Applies the morphological operations.

    Applies the specified morphological operations 'op' to the input numpy array 'input_array'
    using the provided 'kernel' for the given number of 'iterations'.

    Args:
        input_array: The input numpy array to which the morphological operation will be applied.
        op: The morphological operation function to be applied.
        kernel: The kernel to be used for the operation.
        iterations: The number of iterations the operation will be applied to the input array.

    Returns:
        The resulting array after applying the morphological operation.
    """
    for _ in range(iterations):
        input_array = op(input_array, kernel)
    return input_array


@njit
def find_contours(input_image: np.ndarray) -> tuple[list, list]:
    """""Finds contours in a binary image and calculates their areas.

    This function takes a binary image and uses a traversal algorithm to detect edges.
    It uses the Moore algorithm to detect contours. The Moore neighborhood of a pixel P is a set of
    8 pixels that share a vertex or edge with this pixel.
    The edges are returned as a list of pixel coordinates, as well as a list of regions for each edge.

     Args:
        input_image: Input binary image, where true pixels represent objects and false pixels represent background.

    Returns:
        tuple[list, list]:
            - contours_list (list): List of contours, each of which
            is represented by a list of pixel coordinates (row, col).
            - area_list (list): List of areas of the corresponding contours.
            The area of each contour is determined as the number of pixels that make it up.
    """
    contours_list = []
    area_list = []
    input_image = add_black_padding(input_image, 3)  # Add padding to avoid array overflow
    #  Dictionary by calling the key of which the direction changes to the opposite
    back_dir = {'right': 'left', 'bottom': 'top', 'left': 'right', 'top': 'bottom'}
    #  Dictionary on call of the key of which, the direction changes to the right
    r_dir = {'right': 'bottom', 'bottom': 'left', 'left': 'top', 'top': 'right'}
    #  Dictionary, upon calling the key of which, a step is taken in the corresponding direction
    steps = {'right': (0, 1), 'bottom': (1, 0), 'left': (0, -1), 'top': (-1, 0)}

    # Find a pixel with a value of true and declare it the "start" pixel of the outline
    for row in range(1, input_image.shape[0] - 1):
        for col in range(1, input_image.shape[1] - 1):
            if input_image[row, col]:
                contour = []
                now_direction = 'right'  # The initial direction in which we hit the pixel
                row_now, col_now = row, col

                # Each time we hit a true pixel P, we devolve and go back, that is, to the false pixel where we were
                # standing before. Then we go around the pixel P clockwise, visiting each pixel in its Moore
                # neighborhood until we hit a black pixel. The algorithm finishes executing when we hit a pixel
                # from the outline for the second time, just as we hit it initially.
                while (row_now, col_now, now_direction) not in contour or len(contour) == 0:
                    contour.append((row_now, col_now, now_direction))
                    now_direction = back_dir[now_direction]
                    row_now, col_now = (row_now + steps[now_direction][0], col_now + steps[now_direction][1])
                    if input_image[row_now, col_now]:
                        continue
                    now_direction = r_dir[now_direction]
                    row_now, col_now = (row_now + steps[now_direction][0], col_now + steps[now_direction][1])
                    if input_image[row_now, col_now]:
                        continue
                    flag = True
                    for _ in range(3):
                        if not flag:
                            break
                        now_direction = r_dir[now_direction]
                        for _ in range(2):
                            row_now, col_now = (row_now + steps[now_direction][0], col_now + steps[now_direction][1])
                            if input_image[row_now, col_now]:
                                flag = False
                                break
                    if not flag:
                        continue
                    now_direction = r_dir[now_direction]
                    row_now, col_now = (row_now + steps[now_direction][0], col_now + steps[now_direction][1])
                    if input_image[row_now, col_now]:
                        continue
                contour = sorted(contour, key=lambda val: (val[0], val[1]))
                contours_list.append([(row - 1, col - 1) for row, col, _ in contour])

                # Exclude the contour area from the image to search for other contours.
                # Create a list containing the elements of the window from one line.
                # And between the minimum and maximum of them we paint the mask.
                area_list.append(0)
                group = [contour[0]]
                for ind in range(1, len(contour)):
                    if group[-1][0] == contour[ind][0]:
                        group.append(contour[ind])
                    else:
                        area_list[-1] += group[-1][1] + 1 - group[0][1]
                        input_image[group[0][0], group[0][1]:group[-1][1] + 1] = False
                        group = [contour[ind]]
    return contours_list, area_list


def bounding_rect(contour: list[tuple]) -> tuple:
    """Calculate the bounding rectangle for a given contour.

    Args:
        contour: A list of tuples representing the contour points.

    Returns:
        A tuple containing the x-coordinate, y-coordinate, width and height of the bounding rectangle.
    """
    y_rect = contour[0][0]
    h_rect = contour[-1][0] - y_rect
    x_rect = min(contour, key=lambda val: val[1])[1]
    w_rect = max(contour, key=lambda val: val[1])[1] - x_rect
    return x_rect, y_rect, w_rect, h_rect


def rectangle(image: np.ndarray, start_point: tuple, end_point: tuple, color: tuple):
    """Draw a rectangle on the image given the start, end points and the color line.

    Args:
        image: Array representing the image.
        start_point: A tuple containing the (x, y) coordinates of the top-left corner of the rectangle.
        end_point: A tuple containing the (x, y) coordinates of the bottom-right corner of the rectangle.
        color: A tuple representing the color values (e.g., (R, G, B)) to fill the rectangle width.

    Returns:
        None: The function modifies the image in place and returns nothing.
    """
    image[start_point[1], start_point[0]:end_point[0] + 1] = color
    image[start_point[1]:end_point[1] + 1, end_point[0]] = color
    image[end_point[1], start_point[0]:end_point[0] + 1] = color
    image[start_point[1]:end_point[1] + 1, start_point[0]] = color


video = cv2.VideoCapture('229-video.mp4')
# Read first and third frames and scale them to desired dimensions
_, frame1 = video.read()
frame1 = scale_image(frame1, 800)
_, _ = video.read()
ret, frame2 = video.read()
frame2 = scale_image(frame2, 800)

# Main loop to process video frames and draw contours
while video.isOpened() and ret:
    # Creating a foreground mask, highlighting moving objects
    diff = abs_difference(frame1, frame2)
    gray = convert_to_grayscale(diff)
    blur = fast_gaussian_conv(gray, 2)
    threshold = simple_thresh(blur, 10, 255)
    binary = convert_to_binary(threshold)
    morph = morphology_ex(binary, close, np.ones((5, 5), dtype=np.bool_), 8)

    contours, areas = find_contours(morph)  # Detecting contours of shapes using the resulting mask
    for i in range(len(contours)):
        if areas[i] < 1200:  # Condition for removing noise
            continue
        x, y, w, h = bounding_rect(contours[i])  # Determining x and y coordinates, width and height of each contour
        # Draw a rectangle on the frame
        rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0))

    cv2.imshow('1', frame1)  # Displaying the processed frame
    # Reading the next frame and updating the frame1 for the next iteration
    frame1 = frame2
    _, _ = video.read()
    ret, frame2 = video.read()
    frame2 = scale_image(frame2, 800)

    # The condition of closing a window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()
