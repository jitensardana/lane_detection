from __future__ import division
import cv2
import numpy as np
import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        print (channel_count)  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image






def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * a + img * b + c
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, c)



def draw_lines(image, lines, thickness):
    ll = {'num':0, 'slope':0.0, 'x1':0, 'y1':0, 'x2':0, 'y2':0}
    rl = {'num':0, 'slope':0.0, 'x1' : 0, 'y1' : 0, 'x2':0, 'y2':0}
    ysize = image.shape[0]

    for line in lines:

        for x1, y1, x2, y2 in line:
            slope = float((y2 - y1)/(x2 - x1))
            #print y2, y1, x2, x1
            #print slope
            if slope > 0.49 and slope < 1.0:
                rl['num'] += 1
                rl['slope'] += slope
                rl['x1'] += x1
                rl['y1'] += y1
                rl['x2'] += x2
                rl['y2'] += y2
            elif slope > -1.0 and slope < -0.49:
                ll['num'] += 1
                ll['slope'] += slope
                ll['x1'] += x1
                ll['y1'] += y1
                ll['x2'] += x2
                ll['y2'] += y2

    print(rl, ll)

    if rl['num'] > 0 and ll['num'] > 0:
        rslope = rl['slope']/rl['num']
        rx1 = int(rl['x1'] / rl['num'])
        rx2 = int(rl['x2']/ rl['num'])
        ry1 = int(rl['y1']/rl['num'])
        ry2 = int(rl['y2']/rl['num'])

        lslope = ll['slope']/ll['num']
        lx1 = int(ll['x1']/ll['num'])
        lx2 = int(ll['x2']/ll['num'])
        ly1 = int(ll['y1']/ll['num'])
        ly2 = int(ll['y2']/ll['num'])

        xi = int((ly2 - ry2 + rslope * rx2 - lslope * lx2) / (rslope - lslope))
        yi = int(ry2 + rslope * (xi - rx2))
        print ("loop")
        # calculate backoff from intercept for right line
        if rslope > 0.49 and rslope < 1:  # right
            print ("right")
            ry1 = yi + thickness * 3
            rx1 = int(rx2 - (ry2 - ry1) / rslope)
            ry2 = ysize - 1
            rx2 = int(rx1 + (ry2 - ry1) / rslope)
            cv2.line(image, (rx1, ry1), (rx2, ry2), [255, 0, 0], thickness)

            # calculate backoff from intercept for left line
        if lslope < -0.49 and lslope > -1:  # left
            ly1 = yi + thickness * 3
            lx1 = int(lx2 - (ly2 - ly1) / lslope)
            ly2 = ysize - 1
            lx2 = int(lx1 + (ly2 - ly1) / lslope)
            cv2.line(image, (lx1, ly1), (lx2, ly2), [255, 0, 0], thickness)
    return image


def hough(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)

    image = draw_lines(image, lines, 7)
    #print "hello"
    #cv2.imshow("hough lines", image)
    #cv2.waitKey()
    return image
