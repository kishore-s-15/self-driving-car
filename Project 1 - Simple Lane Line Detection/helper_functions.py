# Importing the required Libraries
import cv2
import sys
import numpy as np

def rgb2grayscale(image):
    """
    Function which converts Image from RGB to Grayscale Colorspace
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_blur(image, kernel_size=5):
    """
    Function which blurs the image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Function which detects edges in the image
    """
    return cv2.Canny(image, low_threshold, high_threshold)

def get_roi(edges, vertices):
    """
    Function which gets the Region Of Interest(ROI)
    """
    # Mask for getting ROI
    mask = np.zeros_like(edges)
    mask_color = 255

    # Polygon  Vertices
    cv2.fillPoly(mask, vertices, mask_color)

    # Getting ROI
    return cv2.bitwise_and(edges, mask)

def hough_transform(image):
    """
    Function which uses Hough Transform to find lane lines
    """
    # Hough Transform Parameters
    rho = 2
    theta = 1 / 180
    threshold = 15
    min_line_length = 20
    max_line_gap = 15

    return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

def make_coordiantes(image, line):
    """
    Function which is used to get their coordinates from line
    """
    slope, intercept = line

    # Height(Bottom) of the image
    y1 = int(image.shape[0])

    # Slightly Lower than the middle of the image
    y2 = int(y1 * 3.5 / 5)

    # Calculating X from Y
    # Y = mX + C (or) Y = slope.X + intercept
    # X = (Y - intercept) / slope
    x1 = (y1 - intercept) / slope
    x2 = (y2 - intercept) / slope

    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    """
    Function which averages out the slopes and returns a smoothen slope
    """

    # Empty lists to latter append left and right lane lines
    left_fit = []
    right_fit = []

    # If there are no lines detected,
    # We are going to return None
    if lines is None:
        return None
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # Fitting a Single degree polynomial
        # Or Solving a line equation from the points
        # To get the slope and intercept for the corresponding line
        fit = np.polyfit((x1, x2), (y1, y2), 1)

        # Slope
        slope = fit[0]
        
        # Intercept
        intercept = fit[1]

        # Positive Slope : Two Variables are positively related
        # In our case as X increases, y increases or as y increases x increases
        # The right side lane has positive slope

        # Negative Slope : Two Variables are negatively related
        # In our case as X increases, y decreases or as y increases x decreases
        # The left side lane has negative slope

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # Smoothed out lines
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    # The coordinates of lines are obtained from make_coordiantes function
    left_line = make_coordiantes(image, left_fit_average)
    right_line = make_coordiantes(image, right_fit_average)

    averaged_lines = np.array([left_line, right_line], dtype=np.int32)

    return averaged_lines

def display_lines(image, lines):
    """
    Function which draws the lane lines on the image
    """
    line_image = np.zeros_like(image)

    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    
    return line_image

def lane_line_pipeline(image):
    """
    Pipeline for Finding Lane Lines
    """
    
    grayscaled_image = rgb2grayscale(image)

    blurred_image = gaussian_blur(grayscaled_image, kernel_size=5)

    edges = canny_edge_detection(blurred_image, low_threshold=50, high_threshold=150)

    vertices = np.array([[(405, 345), (575, 345), (image.shape[1] - 100, image.shape[0]), (100, image.shape[0])]], dtype=np.int32)

    roi_image = get_roi(edges, vertices)

    lines = hough_transform(roi_image)

    averaged_lines = average_slope_intercept(image, lines)

    line_image = display_lines(image, averaged_lines)

    return cv2.addWeighted(image, 0.8, line_image, 1, 1)

def video_helper_function(vid_path):
    """
    Function which gets the frames from video and pushes it into the
    lane detection pipeline
    """
    cap = cv2.VideoCapture(vid_path)

    slash = "\\" if sys.platform == 'win32' else "/"
    video_filename = vid_path.split(slash)[-1]

    if not cap.isOpened():       
        print(f"Error opening {video_filename}")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            lane_line_image = lane_line_pipeline(frame)
            cv2.imshow("Video", lane_line_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"Can't receive frame (stream end?). Exiting {video_filename}...")
            break

    cap.release()
    cv2.destroyAllWindows()