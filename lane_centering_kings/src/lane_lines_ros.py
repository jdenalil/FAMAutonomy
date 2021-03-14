#!/usr/bin/env python3
#### This code is an implementation of Udacity's Self-Driving Car ND project by Awad Abdelhalim
#### Based on Jeremy Shannon's solution https://github.com/jeremy-shannon/CarND-Advanced-Lane-Lines

import numpy as np
import cv2
import glob
import argparse

#ros
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys

from numpy import *
import math
import pickle
from scipy.optimize import *
#boost yellows and
def boost(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0,tileGridSize=(10,10))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    return lab


# undistort image using camera calibration matrix from above
def undistort(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return boost(undist)


#source and destination points
mtx= np.float64([[2.79122914e+03, 0.00000000e+00, 1.08414922e+03],
				[0.00000000e+00, 2.80771705e+03, 2.83677514e+02],
				[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist=  np.float64([[-0.07850763, -0.3227967 , -0.02015479,  0.00711071,  1.19477813]])

h= 720
w= 1280

src = np.float32([(475,290),
                  (825,290),
                  (60,710),
                  (1280,710)])

dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])




def unwarp(img, src, dst):
    h,w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv


def sobel(img, thresh=(50, 255)):
    #30 for normal vids, 85 is good for courses
    #sobel = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    sobel= cv2.GaussianBlur(img,(23,23),0)
    sobel= cv2.Sobel(sobel,cv2.CV_16U,1,0,ksize=7)
    sobel = sobel[:,:,0]
    sobel = sobel*(255./np.max(sobel[350:,55:1280]))
    # 2) Apply a threshold to the Sobel
    binary_output = np.zeros_like(sobel)
    binary_output[(sobel > thresh[0]) & (sobel <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


# Define a function that thresholds the L-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=) 185 for night
# Use 100 day, 190 night, 220 courses
def lab_lthresh(img, thresh=(160, 255)):
    # 1) Convert to LAB color space
    #lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_l = img[:,:,0]
    lab_l = lab_l*(255./np.max(lab_l[350:,55:1280]))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_l)
    binary_output[(lab_l > thresh[0]) & (lab_l <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


# Define a function that thresholds the B-channel of LAB
# Use exclusive lower bound (>) and inclusive upper (<=), OR the results of the thresholds (B channel should capture
# yellows)
# Use 175
def lab_bthresh(img, thresh=(205,255)):
    # 1) Convert to LAB color space
    #lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = img[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 180:
        lab_b = lab_b*(255./np.max(lab_b[350:,55:1280]))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output


# Define the complete image processing pipeline, reads raw image and returns binary image with lane lines identified
def pipeline(img):
    # Undistort
    #img_undistort = undistort(img)
    # Perspective Transform
    #img_unwarp, M, Minv = unwarp(img_undistort, src, dst)

    #img_sobel = sobel(img_undistort)

    #img_LThresh = lab_lthresh(img_undistort)
    #img_BThresh = lab_bthresh(img_undistort)

    img_sobel = sobel(img)

    img_LThresh = lab_lthresh(img)
    img_BThresh = lab_bthresh(img)

    # Combine LAB L and Lab B channel thresholds
    combined = np.zeros_like(img_LThresh)
    combined[(img_LThresh == 1) | (img_BThresh == 1)] = 1
    combined[(img_sobel != 1)] = 0

    combined, M, Minv = unwarp(combined, src, dst)

    return combined, Minv


# Define method to fit polynomial to binary image with lines extracted, using sliding window
def sliding_window_polyfit(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    quarter_point = np.int(midpoint//2)
    # Previously the left/right base was the max of the left/right half of the histogram
    # this changes it so that only a quarter of the histogram (directly to the left/right) is considered
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint

    #print('base pts:', leftx_base, rightx_base)

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 150
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    visualization_data = (rectangle_data, histogram)

    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data


# Define method to fit polynomial to binary image based upon a previous fit (chronologically speaking);
# this assumes that the fit will not change significantly from one video frame to the next
def polyfit_using_prev_fit(binary_warped, left_fit_prev, right_fit_prev):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 50
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) &
                      (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - margin)) &
                       (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)
    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds


# Method to determine radius of curvature and distance from lane center
# based on binary image, polynomial fit, and L and R lane pixel indices
def calc_curv_rad_and_center_dist(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3.048/130 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7/370.26 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    left_curverad, right_curverad, center_dist = (0, 0, 0)
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = bin_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds]
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]

    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters

    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts
    if r_fit is not None and l_fit is not None:
        car_position = bin_img.shape[1]/2
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix

    return left_curverad, right_curverad, center_dist


def draw_lane(original_img, binary_img, l_fit, r_fit, Minv):
    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

    ym_per_pix = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7/378 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters

    wheelbase=2.420/ym_per_pix
    x_ref=w/2
    y_ref=h+wheelbase
    x_ref_int=int(x_ref)
    y_ref_int=int(y_ref)
    ld_meters=10
    ld=720-(ld_meters*(100/3.048))
    ld_int=int(ld)
    centerLaneCoef=((l_fit+r_fit)/2)


    #Solving for goal point using look-ahead circle and centerline equations
    def myFunction(goalPoint):

        x=goalPoint[0]
        y=goalPoint[1]

        SOE=empty(2)
        SOE[0]=pow((x_ref-x),2)+pow((y_ref-y),2)-pow(ld,2)
        SOE[1]=centerLaneCoef[0]*pow(x,2)+centerLaneCoef[1]*x+centerLaneCoef[2]-(h-y)
        return SOE

    intGoalPoint=empty(2)
    goalPoint_guess=[x_ref,ld]
    goalPoint=fsolve(myFunction,goalPoint_guess)

    #Converting some parameters for plotting and using in pure pursuint
    intGoalPoint_x=int(goalPoint[0])
    intGoalPoint_y=int(goalPoint[1])
    x_Goal_meters=intGoalPoint_x*xm_per_pix
    y_Goal_meters=intGoalPoint_y*ym_per_pix
    crossTrackError=(goalPoint[0]-x_ref)*xm_per_pix

    #Pure Pursuit
    numer=2*wheelbase*crossTrackError
    denom=ld_meters**2
    delta_degrees=math.degrees(math.atan(numer/denom))
    print("Reference Point: ", x_ref, y_ref)
    print("Goal Point: ",goalPoint)
    print("Look Ahead Distance (meters): ",ld_meters)
    print("Cross Track Error (meters): ", crossTrackError )
    print("Wheel angle change: ", delta_degrees)
    #Evans Changes End

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)
    cv2.circle(color_warp,(x_ref_int,y_ref_int), ld_int, (0,0,255), 30)
    cv2.circle(color_warp,(intGoalPoint_x,intGoalPoint_y), 5, (255,0,0), 30)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result


def draw_data(original_img, curv_rad, center_dist):
    new_img = np.copy(original_img)
    h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(new_img, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    return new_img


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
        self.current_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #number of detected pixels
        self.px_count = None
    def add_fit(self, fit, inds):
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
            if (self.diffs[0] > 0.001 or                self.diffs[1] > 1.0 or                self.diffs[2] > 100.) and                len(self.current_fit) > 0:
                # bad fit! abort! abort! ... well, unless there are no fits in the current_fit queue, then we'll take it
                self.detected = False
            else:
                self.detected = True
                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit)-5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # or remove one from the history, if not found
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)

class image_converter:
    def __init__(self):
        self.offset_pub = rospy.Publisher("lane_offset",Float64,queue_size=20)
        self.l_rad_pub = rospy.Publisher("l_rad",Float64,queue_size=20)
        self.r_rad_pub = rospy.Publisher("r_rad",Float64,queue_size=20)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/zed/zed_node/left_raw/image_raw_color", Image, self.detect)

    def detect(self,data):
        try:
          img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)

        img= cv2.resize(img,(1280,720))
        new_img = np.copy(img)
        img_bin, Minv = pipeline(new_img)

        l_line = Line()
        r_line = Line()


        # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
        if not l_line.detected or not r_line.detected:
            l_fit, r_fit, l_lane_inds, r_lane_inds, _ = sliding_window_polyfit(img_bin)
        else:
            l_fit, r_fit, l_lane_inds, r_lane_inds = polyfit_using_prev_fit(img_bin, l_line.best_fit, r_line.best_fit)

        # invalidate both fits if the difference in their x-intercepts isn't around 250 px (+/- 100 px)
        if l_fit is not None and r_fit is not None:
            # calculate x-intercept (bottom of image, x=image_height) for fits
            h = img.shape[0]
            l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
            r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
            x_int_diff = abs(r_fit_x_int-l_fit_x_int)
            if abs(300 - x_int_diff) > 100:
                l_fit = None
                r_fit = None

        l_line.add_fit(l_fit, l_lane_inds)
        r_line.add_fit(r_fit, r_lane_inds)

        # draw the current best fit if it exists
        if l_line.best_fit is not None and r_line.best_fit is not None:
            rad_l, rad_r, d_center = calc_curv_rad_and_center_dist(img_bin, l_line.best_fit, r_line.best_fit,
                                                                   l_lane_inds, r_lane_inds)
            self.offset_pub.publish(d_center)
            self.l_rad_pub.publish(rad_l)
            self.r_rad_pub.publish(rad_r)
        else:
            print("no lanes detected")

        # img_out = draw_lane(new_img, img_bin, l_line.best_fit, r_line.best_fit, Minv)

def main():
  ic = image_converter()
  rospy.init_node('lane_detection_node', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

main()


#if __name__ == '__main__':
#    try:
#        images= (glob.glob('*.png'))
#    	for image in images:
#    		l_line = Line()
#    		r_line = Line()
#    		img= cv2.imread(image)
#    		cv2.imwrite('output_%s' % image, process_image(img))
#    except rospy.ROSInterruptException:
#        pass


####################################################################################################################################
"""
parser = argparse.ArgumentParser(description="Draw detected lane lines into image.")
parser.add_argument('input_image', help='Input file where lane lines are to be detected.')
args = parser.parse_args()

if args.input_image == 'all':
	print('finding images')
	images= (glob.glob('*.png'))
	print('got em')
	print(images)
	n = 0
	for image in images:
		l_line = Line()
		r_line = Line()
		img= cv2.imread(image)
		print('saving image ' + str(n))
		cv2.imwrite('output_%s' % image, process_image(img))
		n += 1
else:
	l_line = Line()
	r_line = Line()
	image= cv2.imread(args.input_image)
	output= process_image(image)
	cv2.imwrite('output_%s' % args.input_image, output)
"""
####################################################################################################################################
