from helper_functions import *
import numpy as np
import cv2

'''
image = cv2.imread('test_images/solidWhiteRight.jpg')
cv2.imshow("op", image)
cv2.waitKey(0)
'''


def my_draw_lines2(img, lines, color=[255, 0, 0], thickness=6, debug=False):
    ysize = img.shape[0]
    try:
        # rightline and leftline cumlators
        rl = {'num': 0, 'slope': 0.0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
        ll = {'num': 0, 'slope': 0.0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = ((y2 - y1) / (x2 - x1))
                if slope > 0.5 and slope < 1.0:  # right
                    rl['num'] += 1
                    rl['slope'] += slope
                    rl['x1'] += x1
                    rl['y1'] += y1
                    rl['x2'] += x2
                    rl['y2'] += y2
                elif slope > -1.0 and slope < -0.5:  # left
                    ll['num'] += 1
                    ll['slope'] += slope
                    ll['x1'] += x1
                    ll['y1'] += y1
                    ll['x2'] += x2
                    ll['y2'] += y2

        if rl['num'] > 0 and ll['num'] > 0:
            # average/extrapolate all of the lines that makes the right line
            rslope = rl['slope'] / rl['num']
            rx1 = int(rl['x1'] / rl['num'])
            ry1 = int(rl['y1'] / rl['num'])
            rx2 = int(rl['x2'] / rl['num'])
            ry2 = int(rl['y2'] / rl['num'])

            # average/extrapolate all of the lines that makes the left line
            lslope = ll['slope'] / ll['num']
            lx1 = int(ll['x1'] / ll['num'])
            ly1 = int(ll['y1'] / ll['num'])
            lx2 = int(ll['x2'] / ll['num'])
            ly2 = int(ll['y2'] / ll['num'])

            # find the right and left line's intercept, which means solve the following two equations
            # rslope = ( yi - ry1 )/( xi - rx1)
            # lslope = ( yi = ly1 )/( xi - lx1)
            # solve for (xi, yi): the intercept of the left and right lines
            # which is:  xi = (ly2 - ry2 + rslope*rx2 - lslope*lx2)/(rslope-lslope)
            # and        yi = ry2 + rslope*(xi-rx2)
            xi = int((ly2 - ry2 + rslope * rx2 - lslope * lx2) / (rslope - lslope))
            yi = int(ry2 + rslope * (xi - rx2))

            # calculate backoff from intercept for right line
            if rslope > 0.5 and rslope < 1.0:  # right
                ry1 = yi + thickness * 3
                rx1 = int(rx2 - (ry2 - ry1) / rslope)
                ry2 = ysize - 1
                rx2 = int(rx1 + (ry2 - ry1) / rslope)
                cv2.line(img, (rx1, ry1), (rx2, ry2), [255, 0, 0], thickness)

            # calculate backoff from intercept for left line
            if lslope < -0.5 and lslope > -1.0:  # left
                ly1 = yi + thickness * 3
                lx1 = int(lx2 - (ly2 - ly1) / lslope)
                ly2 = ysize - 1
                lx2 = int(lx1 + (ly2 - ly1) / lslope)
                cv2.line(img, (lx1, ly1), (lx2, ly2), [255, 0, 0], thickness)

        return lslope + rslope, lslope, rslope, lx1, rx1
    except:
        return -1000, 0.0, 0.0, 0.0, 0, 0


def my_hough_lines2(img, rho, theta, threshold, min_line_len, max_line_gap, debug=False):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn using the new single line for left and right lane line method.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    masked_lines = np.zeros(img.shape, dtype=np.uint8)
    lane_info = my_draw_lines2(masked_lines, lines, debug=debug)

    # Create a "color" binary image to combine with original image
    ignore_color = np.copy(masked_lines) * 0  # creating a blank color channel for combining
    line_image = np.dstack((masked_lines, ignore_color, ignore_color))

    return line_image, lane_info


def my_draw_masked_area(img, areas, color=[128, 0, 128], thickness=2):
    for points in areas:
        for i in range(len(points) - 1):
            cv2.line(img, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]), color, thickness)
        cv2.line(img, (points[0][0], points[0][1]), (points[len(points) - 1][0], points[len(points) - 1][1]), color,
                 thickness)


def my_image_only_yellow_white_curve1(image):
    # setup inRange to mask off everything except white and yellow
    lower_yellow_white = np.array([192, 192, 32])
    upper_yellow_white = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower_yellow_white, upper_yellow_white)
    return cv2.bitwise_and(image, image, mask=mask)


def my_image_only_yellow_white_curve2(image):
    # setup inRange to mask off everything except white and yellow
    lower_yellow_white = np.array([140, 140, 64])
    upper_yellow_white = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower_yellow_white, upper_yellow_white)
    return cv2.bitwise_and(image, image, mask=mask)


def procimage2(image):
    # mask calculations
    imshape = image.shape
    xbottom1 = int(imshape[1] / 16)
    xbottom2 = int(imshape[1] * 15 / 16)
    xtop1 = int(imshape[1] * 14 / 32)
    xtop2 = int(imshape[1] * 18 / 32)
    ybottom1 = imshape[0]
    ybottom2 = imshape[0]
    ytopbox = int(imshape[0] * 9 / 16)

    # Read in the image and convert to grayscale
    gray = grayscale(my_image_only_yellow_white_curve1(image))

    # Define a kernel size for Gaussian smoothing / blurring
    kernel_size = 7  # Must be an odd number (3, 5, 7...)
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and run it
    low_threshold = 5
    high_threshold = 17
    edges = canny(blur_gray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    # We can lift the mask higher now, since the line drawing function is a bit smarter
    vertices = np.array([[(xbottom1, ybottom1), (xtop1, ytopbox), (xtop2, ytopbox), (xbottom2, ybottom2)]],
                        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 40  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 75  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "masked_lines" is a single channel mask
    ignore_color = np.zeros(masked_edges.shape, dtype=np.uint8)
    original_image = np.dstack((masked_edges, ignore_color, ignore_color))

    line_image, lane_info = my_hough_lines2(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

    # if first attempted failed - that means we are dealing with yellow and white on white.
    #   try second mask
    # if np.array_equal(line_image, original_image):
    if lane_info[0] == -1000:
        # Read in the image and convert to grayscale
        gray = grayscale(my_image_only_yellow_white_curve2(image))

        # Define a kernel size for Gaussian smoothing / blurring
        kernel_size = 13  # Must be an odd number (3, 5, 7...)
        blur_gray = gaussian_blur(gray, kernel_size)

        # Define our parameters for Canny and run it
        low_threshold = 10
        high_threshold = 36
        edges = canny(blur_gray, low_threshold, high_threshold)

        # This time we are defining a four sided polygon to mask
        # We can lift the mask higher now, since the line drawing function is a bit smarter
        masked_edges = region_of_interest(edges, vertices)

        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 16  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 130  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 75  # minimum number of pixels making up a line
        max_line_gap = 100  # maximum gap in pixels between connectable line segments

        # debugging
        # orignial_image = np.dstack((masked_edges, ignore_color, ignore_color))

        # Run Hough on edge detected image
        # Output "masked_lines" is a single channel mask

        line_image, lane_info = my_hough_lines2(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

        # debugging short
        # return weighted_img(line_image, original_image)

    # my_draw_masked_area(line_image, vertices)
    # line_image = weighted_img(line_image, original_image)
    # return line_image

    # return the merged red colored lines with the image as background
    return weighted_img(line_image, image)


# loop through and display all of the process images
'''
import os
import re

filepath = "test_images/"
pattern = re.compile("^.+processed.jpg$")
files = os.listdir(filepath)
for file in files:
    # import the image if it is not a saved output
    if not pattern.match(file):
        image_filepath = filepath + file
        image = mpimg.imread(image_filepath)

        # process image
        image_wlines = procimage2(image)  # attempt 2
        print('Image ', image_filepath, ' has dimensions: ', image.shape)

        # next image
        plt.figure()

        # plot the image
        plt.imshow(image_wlines)

        # writeout the image with "-processed" in the name so it will not be reprocessed.
        plt.savefig(image_filepath.replace(".jpg", "-processed.jpg"))
'''

image = cv2.imread('test_images/solidWhiteCurve.jpg')
cv2.imshow("output", procimage2(image))
cv2.waitKey(0)
