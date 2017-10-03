import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import functions
import sys

def draw_lane_lines(image = sys.argv[1]):
    image = mpimg.imread(image)
    imshape = image.shape
    grayscaled_img = functions.grayscale(image)
    plt.subplot(2,2,1)
    plt.imshow(grayscaled_img, cmap="gray")

    # Smoothing
    blurred_img = functions.gaussian_blur(grayscaled_img, 7)

    # Edge Detection
    edges_img = functions.canny(blurred_img, 50, 150)

    # Mask edges img

    vertices = np.array([[(0, imshape[0]), (465, 320), (475, 320), (imshape[1], imshape[0])]], dtype = np.int32)
    edges_img_with_mask = functions.region_of_interest(edges_img, vertices)

    plt.subplot(2,2,2)
    plt.imshow(edges_img_with_mask)

    rho = 2
    theta = np.pi/180
    threshold = 45
    min_line_len = 40
    max_line_gap = 100
    lines_img = functions.hough_lines(edges_img_with_mask, rho, theta, threshold, min_line_len, max_line_gap)

    hough_rgb_img = cv2.cvtColor(lines_img, cv2.COLOR_GRAY2BGR)
    plt.subplot(2,2,3)
    plt.imshow(hough_rgb_img)
    final_img = functions.weighted_img(hough_rgb_img, image)

    plt.subplot(2,2,4)
    plt.imshow(final_img)

    return final_img


if __name__ == "__main__":
    draw_lane_lines()


