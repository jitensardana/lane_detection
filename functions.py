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


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# Python 3 has support for cool math symbols.

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


# seventh attempt, based on second attempt, second take
#   This version of draw_lines averages out all of the left and right lane lines
#   And just draw a single line for both
#
#   This attempt seems to work for case 1 and 2, most of the Optional Challenge! :)
#   It appears that if we add a yellow_white color filter using inRange and a range of
#   rho larger than 1 with higher thresholds (higher minimum votes) to eliminate extra lines.
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


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def movingAverage(avg, new_sample, N=20):
    if (avg == 0):
        return new_sample
    avg -= avg / N;
    avg += new_sample / N;
    return avg;


def draw_lines_orig(img, lines, color=[255, 0, 0], thickness=2):
    largestLeftLineSize = 0
    largestRightLineSize = 0
    largestLeftLine = (0, 0, 0, 0)
    largestRightLine = (0, 0, 0, 0)

    if lines is None:
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw right line
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            size = math.hypot(x2 - x1, y2 - y1)
            slope = ((y2 - y1) / (x2 - x1))
            # Filter slope based on incline and
            # find the most dominent segment based on length
            if (slope > 0.5):  # right
                if (size > largestRightLineSize):
                    largestRightLine = (x1, y1, x2, y2)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            elif (slope < -0.5):  # left
                if (size > largestLeftLineSize):
                    largestLeftLine = (x1, y1, x2, y2)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # Define an imaginary horizontal line in the center of the screen
    # and at the bottom of the image, to extrapolate determined segment
    imgHeight, imgWidth = (img.shape[0], img.shape[1])
    upLinePoint1 = np.array([0, int(imgHeight - (imgHeight / 3))])
    upLinePoint2 = np.array([int(imgWidth), int(imgHeight - (imgHeight / 3))])
    downLinePoint1 = np.array([0, int(imgHeight)])
    downLinePoint2 = np.array([int(imgWidth), int(imgHeight)])

    # Find the intersection of dominant lane with an imaginary horizontal line
    # in the middle of the image and at the bottom of the image.
    p3 = np.array([largestLeftLine[0], largestLeftLine[1]])
    p4 = np.array([largestLeftLine[2], largestLeftLine[3]])
    upLeftPoint = seg_intersect(upLinePoint1, upLinePoint2, p3, p4)
    downLeftPoint = seg_intersect(downLinePoint1, downLinePoint2, p3, p4)
    if (math.isnan(upLeftPoint[0]) or math.isnan(downLeftPoint[0])):
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw right line
        return
    cv2.line(img, (int(upLeftPoint[0]), int(upLeftPoint[1])), (int(downLeftPoint[0]), int(downLeftPoint[1])),
             [0, 0, 255], 8)  # draw left line

    # Calculate the average position of detected left lane over multiple video frames and draw
    global avgLeft
    avgx1, avgy1, avgx2, avgy2 = avgLeft
    avgLeft = (
        movingAverage(avgx1, upLeftPoint[0]), movingAverage(avgy1, upLeftPoint[1]),
        movingAverage(avgx2, downLeftPoint[0]),
        movingAverage(avgy2, downLeftPoint[1]))
    avgx1, avgy1, avgx2, avgy2 = avgLeft
    cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line

    # Find the intersection of dominant lane with an imaginary horizontal line
    # in the middle of the image and at the bottom of the image.
    p5 = np.array([largestRightLine[0], largestRightLine[1]])
    p6 = np.array([largestRightLine[2], largestRightLine[3]])
    upRightPoint = seg_intersect(upLinePoint1, upLinePoint2, p5, p6)
    downRightPoint = seg_intersect(downLinePoint1, downLinePoint2, p5, p6)
    if (math.isnan(upRightPoint[0]) or math.isnan(downRightPoint[0])):
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw right line
        return
    cv2.line(img, (int(upRightPoint[0]), int(upRightPoint[1])), (int(downRightPoint[0]), int(downRightPoint[1])),
             [0, 0, 255], 8)  # draw left line

    # Calculate the average position of detected right lane over multiple video frames and draw
    global avgRight
    avgx1, avgy1, avgx2, avgy2 = avgRight
    avgRight = (movingAverage(avgx1, upRightPoint[0]), movingAverage(avgy1, upRightPoint[1]),
                movingAverage(avgx2, downRightPoint[0]), movingAverage(avgy2, downRightPoint[1]))
    avgx1, avgy1, avgx2, avgy2 = avgRight
    cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255, 255, 255], 12)  # draw left line


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    #print img.shape
    line_img = np.zeros(img.shape+(3,), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
