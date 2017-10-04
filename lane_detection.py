import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functions import grayscale, gaussian_blur, canny, my_hough_lines2, my_image_only_yellow_white_curve2, my_image_only_yellow_white_curve1, weighted_img,\
    region_of_interest, hough_lines
import sys


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
    #cv2.imshow("edges",edges)
    #cv2.waitKey()
    #print(edges.shape)

    # This time we are defining a four sided polygon to mask
    # We can lift the mask higher now, since the line drawing function is a bit smarter
    vertices = np.array([[(xbottom1, ybottom1), (xtop1, ytopbox), (xtop2, ytopbox), (xbottom2, ybottom2)]],
                        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    #cv2.imshow("masked", masked_edges)
    #cv2.waitKey()

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

    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # if first attempted failed - that means we are dealing with yellow and white on white.
    #   try second mask
    # if np.array_equal(line_image, original_image):
    """
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
        plt.imshow(masked_edges)
        #plt.show()

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
        #cv2.imshow("line image", line_image)
        #cv2.waitKey()

        # debugging short
        # return weighted_img(line_image, original_image)
    """
    # my_draw_masked_area(line_image, vertices)
    # line_image = weighted_img(line_image, original_image)
    # return line_image

    # return the merged red colored lines with the image as background
    return weighted_img(line_image, image)


# loop through and display all of the process images
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
        #plt.show()

        # writeout the image with "-processed" in the name so it will not be reprocessed.
        plt.savefig(image_filepath.replace(".jpg", "-processed.jpg"))