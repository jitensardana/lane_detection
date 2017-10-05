import cv2
import numpy as np
import functions
import sys



def process_image(image=sys.argv[1]):

    # variables
    image = cv2.imread(image)
    imshape = image.shape
    print imshape
    xbottom1 = int(imshape[1] / 16)
    xbottom2 = int(imshape[1] * 15 / 16)
    xtop1 = int(imshape[1] * 14 / 32)
    xtop2 = int(imshape[1] * 18 / 32)
    ybottom1 = imshape[0]
    ybottom2 = imshape[0]
    ytopbox = int(imshape[0] * 9 / 16)

    vertices = np.array([[(xbottom1, ybottom1), (xtop1, ytopbox), (xtop2, ytopbox), (xbottom2, ybottom2)]], dtype=np.int32)


    #gray image
    gray = functions.grayscale(image)
    cv2.imshow("gray", gray)
    cv2.waitKey()

    kernel_size = 7

    #blurred image
    blurred = functions.gaussian_blur(gray, kernel_size)
    cv2.imshow("blurred", blurred)
    cv2.waitKey()

    # region of interest
    masked_image = functions.region_of_interest(blurred, vertices)
    cv2.imshow("masked_image", masked_image)
    cv2.waitKey()

    # edge image
    low_threshold = 100
    high_threshold = 200
    edge_image = functions.canny(masked_image, low_threshold, high_threshold)
    cv2.imshow("edge image", edge_image)
    cv2.waitKey()

    # hough_line image
    rho = 2
    theta = np.pi/180
    min_line_len = 40
    max_line_gap = 50
    threshold = 40
    cv2.line(edge_image, (xbottom1, ybottom1), (xtop1, ytopbox), [0,0,0], thickness=2)
    cv2.line(edge_image, (xtop2, ytopbox), (xbottom2, ybottom2), [0,0,0], thickness=2)
    cv2.line(edge_image, (xtop1, ytopbox), (xtop2, ytopbox), [0,0,0], thickness=2)
    line_img = functions.hough(edge_image, rho, theta, threshold, min_line_len, max_line_gap)
    ignore_color = np.copy(line_img)*0
    line_img = np.dstack((ignore_color, ignore_color, line_img))
    cv2.imshow("line_img",line_img)
    cv2.waitKey()

    image = functions.weighted_img(image, line_img)
    cv2.imshow("final", image)
    cv2.waitKey()


if __name__ == '__main__':
    process_image(sys.argv[1])