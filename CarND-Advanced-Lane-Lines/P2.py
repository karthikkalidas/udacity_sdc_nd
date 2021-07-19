#!/usr/bin/env python
# coding: utf-8
# Self-Driving Car Engineer Nanodegree
# Project: **Finding Lane Lines on the Road**
# Author: Karthik Kalidas

# Import packages
import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

TEST_DIR = './test_images/'
SAVE_DIR = TEST_DIR + 'undistorted/'

def camera_cal():
    # prepare object points, like (0,0,0), (1,0,0), ...., (8,5,0)
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image space

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object and image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display corners
            # img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    ## DISTORTION CORRECTION
    images = glob.glob(TEST_DIR + '*.jpg')
    filename = os.path.basename(fname)
    for fname in images:
        img = cv2.imread(fname)
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        undistorted_fname = SAVE_DIR + os.path.splitext(os.path.basename(fname))[0] + '_undistorted.jpg'
        cv2.imwrite(undistorted_fname, undistorted)

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output

def dir_thresh(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    theta = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(theta)
    binary_output[(theta >= thresh[0]) & (theta <= thresh[1])] = 1
    return binary_output

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S >= thresh[0]) & (S <= thresh[1])] = 1
    return binary_output

def binary_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ksize = 3
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    # grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(0, 100))
    # mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(20, 100))
    # dir_binary = dir_thresh(gray, sobel_kernel=ksize, thresh=(0, np.pi/2))
    hls_binary = hls_select(img, thresh=(150, 255))
    combined = np.zeros_like(hls_binary)
    # combined[(gradx == 1) & (mag_binary == 1) & (dir_binary == 1) & (hls_binary == 1)] = 1
    combined[(gradx == 1) | (hls_binary == 1)] = 1
    # cv2.imshow("", hls_binary)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # plt.imshow(combined, cmap='gray')
    # plt.show()
    return combined

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def measure_curvature_real(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ally)

        # Calculation of R_curve (radius of curvature)
        self.radius_of_curvature = ((1 + (2*self.current_fit[0]*y_eval*ym_per_pix + self.current_fit[1])**2)**1.5) \
                                    / np.absolute(2*self.current_fit[0])

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Create a left Line object
    left_line = Line()
    left_line.detected = True
    left_line.current_fit = left_fit
    left_line.allx = left_fitx
    left_line.ally = ploty
    left_line.bestx = np.mean(left_line.allx)

    # Create a right Line object
    right_line = Line()
    right_line.detected = True
    right_line.current_fit = right_fit
    right_line.allx = right_fitx
    right_line.ally = ploty
    right_line.bestx = np.mean(right_line.allx)

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_line, right_line

def find_lanelines():
    ## CAMERA CALIBRATION
    # camera_cal()

    ## LOADING TEST IMAGES/VIDEOS
    img = cv2.imread(SAVE_DIR + 'test1_undistorted.jpg')

    ## IMAGE BINARY THRESHOLDING
    binary_combined = binary_threshold(img)

    ## PERSPECTIVE TRANSFORM
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[250,670],[600,445],[740,445],[1130,670]])
    dst = np.float32([[250,660],[250,0],[1150,0],[1150,660]])
    M = cv2.getPerspectiveTransform(src, dst)
    binary_warped = cv2.warpPerspective(binary_combined, M, img_size, flags=cv2.INTER_LINEAR)

    ## POLYNOMIAL FITTING
    out_img, left_line, right_line = fit_polynomial(binary_warped)

    ## CURVATURE ESTIMATION
    left_line.measure_curvature_real()
    right_line.measure_curvature_real()
    print(left_line.radius_of_curvature, right_line.radius_of_curvature)

    ## POSITION ESTIMATION
    vehicle_center = ((left_line.bestx + right_line.bestx)/2) - (img.shape[1]/2)
    print(vehicle_center)

    ## FINAL RESULT WARPED BACK
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.allx, left_line.ally]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.allx, right_line.ally])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(dst, src)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size)
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__=='__main__':
    find_lanelines()
