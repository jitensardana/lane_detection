import cv2
import numpy as np
import math


# Converts color of the image from RGB to GRAY
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def moving_average(avg, new_sample, N=20):
    if avg == 0:
        return new_sample
    avg -= avg/N
    avg += new_sample/N
    return avg


# Find edges using canny detection algorithm
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


# Smoothing using Gaussian Blur
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# Define the region of interest
def region_of_interest(img, vertices):
    # make a numpy array similar to shape and size as that of image and filled with 0
    mask = np.zeros_like(img)

    if len(img.shape) > 2 :
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(mask, img)

    return masked_image


# Returns the original image with hough lines drawn on it
def weighted_img(img, initial_img, alpha = 0.8, beta = 1., gamma=0.):
    '''
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * alpha + img * beta + gamma
        NOTE: initial_img and img must be the same shape!
    '''
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


# Draw lane lines using hough transform on the canny image
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def find_line_fit(slope_intercept):
    kept_slopes = []
    kept_intercepts = []

    print("Slope & intercept : ", slope_intercept)

    if len(slope_intercept) == 1 :
        return slope_intercept[0][0], slope_intercept[0][1]

    slopes = [pair[0] for pair in slope_intercept]
    mean_slope = np.mean(slopes)
    std_slope = np.std(slopes)

    for pair in slope_intercept:
        slope = pair[0]
        if slope - mean_slope < 1.5 * std_slope:
            kept_slopes.append(slope)
            kept_intercepts.append(pair[1])

    if not kept_slopes:
        kept_slopes = slopes
        kept_intercepts = [pair[1] for pair in slope_intercept]

    slope = np.mean(kept_slopes)
    intercept = np.mean(kept_intercepts)
    print("Slope: ", slope, "Intercept: ", intercept)
    return slope, intercept


def intersection_x(coef1, intercept1, coef2, intercept2):
    x = (intercept2 - intercept1)/(coef1 - coef2)
    return x


def draw_linear_regression_line(coef, intercept, intersection_x, img , imshape = [540,960], color = [255,0,0], thickness = 2):
    print("Coef: ", coef, " Intercept: ", intercept, " intersection_x: ", intersection_x)

    point_one = (int(intersection_x), int(intersection_x * coef + intercept))
    if coef > 0:
        point_two = (imshape[1], int(imshape[1] * coef + intercept))
    elif coef < 0:
        point_two = (0, int(0 * coef + intercept))

    print("Point one: ", point_one, "Point two: ", point_two)

    cv2.line(img, point_one, point_two, color, thickness)


# This function draws the lines by taking the coordinates from the Hough transform function in opencv
def draw_lines(img, lines, color=[255,0,0], thickness = 2):
    positive_slope_points = []
    negative_slope_points = []
    positive_slope_intercept = []
    negative_slope_intercept = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            if not math.isnan(slope):
                if length > 50:
                    if slope > 0:
                        positive_slope_points.append([x1, y1])
                        positive_slope_points.append([x2, y2])
                        positive_slope_intercept.append([slope, y1 - slope * x1])

                    elif slope < 0:
                        negative_slope_points.append([x1, y1])
                        negative_slope_points.append([x2, y2])
                        negative_slope_intercept.append([slope, y1 - slope * x1])


    # If either array is empty waive the length requirement

    if not positive_slope_points:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = (y1-y2)/(x1-x2)
                if slope > 0:
                    positive_slope_points.append([x1,y1])
                    positive_slope_points.append([x2,y2])
                    positive_slope_intercept.append([slope, y1 - slope * x1])

    if not negative_slope_points:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = (y1-y2)/(x1-x2)
                if slope < 0:
                    negative_slope_points.append([x1,y1])
                    negative_slope_points.append([x2,y2])
                    negative_slope_intercept.append([slope, y1 - slope * x1])

    if not positive_slope_points:
        print("positive_slope_points still empty")
    if not negative_slope_points:
        print("negative_slope_points still empty")

    # For debugging
    positive_slope_points = np.array(positive_slope_points)
    negative_slope_points = np.array(negative_slope_points)

    pos_coef, pos_intercept = find_line_fit(positive_slope_intercept)
    neg_coef, neg_intercept = find_line_fit(negative_slope_intercept)

    intersection_x_coord = intersection_x(pos_coef, pos_intercept, neg_coef, neg_intercept)

    draw_linear_regression_line(pos_coef, pos_intercept, intersection_x_coord, img, color=color, thickness=thickness)
    draw_linear_regression_line(neg_coef, neg_intercept, intersection_x_coord, img, color=color, thickness=thickness)