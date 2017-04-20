from helpers import add_recent_centers
from helpers import draw_window_box
from helpers import moving_average_scale
from helpers import draw_lines_weighted
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import cv2
import numpy as np
import pickle as pickle
import pandas as pd
import glob
import settings


class LaneLineFinder():
    """
    This class performs the individual calculations on a single lane line
    """

    def __init__(self, img_size, x_pixels_per_meter, y_pixels_per_meter, kind):
        self.img_size = img_size
        self.x_pixels_per_meter = x_pixels_per_meter
        self.y_pixels_per_meter = y_pixels_per_meter
        self.smooth_factor = 15
        self.found = False
        self.first = True
        self.kind = kind
        self.line = np.zeros((img_size[1], img_size[0], 3), dtype = np.uint8)
        self.line_mask = np.ones((img_size[1], img_size[0]), dtype=np.uint8) # create 2D line mask
        self.img_width = img_size[0]
        self.img_height = img_size[1]
        self.first_coeffs = np.array([], dtype = np.float64)
        self.next_coeffs = np.array([], dtype = np.float64)
        self.out_img = np.zeros_like(self.line)
        self.firstMargin = 50
        self.nextMargin = 50

    def find_lane_line(self, mask):

        if self.first:
            self.get_first_coeffs(mask, self.kind)
            fitx, ploty = self.get_line_pts(self.first_coeffs)
            self.first = False

        # if we are not on the first image
        if not self.first:
            self.get_next_coeffs(mask, self.kind)
            fitx, ploty = self.get_line_pts(self.next_coeffs)

        self.line = self.draw_lines(mask, fitx, ploty)


    def get_next_coeffs(self, mask, kind):
        nonzero = mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Should this be first coeffs or previous coeffs
        lane_inds = ((nonzerox > self.first_coeffs[0] * (nonzeroy **2)
                     + self.first_coeffs[1] * nonzeroy + self.first_coeffs[2] - self.nextMargin)) & \
                    (nonzerox < self.first_coeffs[0] * (nonzeroy ** 2) + self.first_coeffs[1] * nonzeroy + self.first_coeffs[2] + self.nextMargin)

        # TODO: If count > 1 should be self.prev_coeffs instead of self.first_coeffs

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        self.next_coeffs = np.polyfit(y, x, 2)

    def get_line_pts(self, coeffs):
        ploty = np.linspace(0, self.img_height - 1, self.img_height)
        fitx = coeffs[0] * ploty ** 2 + coeffs[1] * ploty + coeffs[2]
        return fitx, ploty

    def draw_lines(self, mask, fitx, ploty):
        out_img = np.dstack((mask, mask, mask)) * 255
        window_img = np.zeros_like(out_img)

        line_window1 = np.array([np.transpose(np.vstack([fitx - self.nextMargin, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx + self.nextMargin, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))

        lane = np.array(list(zip(fitx, ploty)), np.int32)
        # draw the lane
        cv2.fillPoly(window_img, np.int_([line_pts]), (0, 255, 0))
        cv2.fillPoly(out_img, [lane], (0, 255, 0))
        return out_img

    def get_first_coeffs(self, mask, kind):
        histogram = np.sum(mask[int(mask.shape[0]/2):,:], axis = 0)
        self.out_img = np.dstack((mask, mask, mask)) * 255
        midpoint = np.int(histogram.shape[0]/2)

        if kind == 'LEFT':
            x_base = np.argmax(histogram[:midpoint])
        if kind == 'RIGHT':
            x_base = np.argmax(histogram[midpoint:]) + midpoint

        n_windows = 9
        window_height = np.int(self.img_height // n_windows)

        nonzero = mask.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]

        x_current = x_base
        margin = 50
        minpix = 50
        lane_inds = []

        for level in range(n_windows):
            win_y_low = self.img_height - (level + 1) * window_height
            win_y_high = self.img_height - (level * window_height)
            win_x_low = x_current - self.firstMargin
            win_x_high = x_current + self.firstMargin

            # draw rectangles
            cv2.rectangle(self.out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0))

            # Identify good indices (numbers type)
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low)
                         & (nonzerox < win_x_high)).nonzero()[0]

            lane_inds.append(good_inds)

            # recenter onto the mean position if we found > minpix
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # extract x pixel positions:
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        # Fit a second order polynomial
        self.first_coeffs = np.polyfit(y, x, 2)
        #
        # # Visualize
        # ploty = np.linspace(0, self.img_height - 1, self.img_height)
        # first_fitx = self.first_coeffs[0] * ploty ** 2 + self.first_coeffs[1] * ploty + self.first_coeffs[2]
        # self.out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [255, 0, 0]
        # # plt.imshow(self.out_img)
        # # plt.plot(first_fitx, ploty, color='blue')
        # # plt.show()
        #
        # return first_fitx

        # TODO: use previous line points as an anchor for next line x_base
        # TODO: use previous line points as an anchor for next x_current


    def get_curvature(self):
        pass
