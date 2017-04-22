from helpers import add_recent_centers
from helpers import moving_average_scale
import matplotlib.pyplot as plt
import cv2
import numpy as np

# TODO: Add averaging method to average / smooth the most recent polynomial coefficients

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
        self.previous_line = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        self.line_mask = np.ones((img_size[1], img_size[0]), dtype=np.uint8) # create 2D line mask
        self.img_width = img_size[0]
        self.img_height = img_size[1]
        self.initial_coeffs = np.array([], dtype = np.float64)
        self.next_coeffs = np.array([], dtype = np.float64)
        self.out_img = np.zeros_like(self.line)
        self.firstMargin = 50
        self.nextMargin = 15
        self.recent_coefficients = []
        self.deviations = []# TODO: Remove later
        self.curvature = None

    def find_lane_line(self, mask, reset = False):

        if self.first:
            self.get_initial_coeffs(mask, self.kind)
            fitx, ploty = self.get_line_pts(self.initial_coeffs)
            self.get_next_coeffs(mask, self.initial_coeffs, self.kind)
            self.first = False


        if not self.first:
            # Append recent coefficients
            self.recent_coefficients.append(self.next_coeffs)

            self.get_next_coeffs(mask, self.next_coeffs, self.kind)


            fitx, ploty = self.get_line_pts(self.next_coeffs)

        self.get_curvature(fitx, ploty)
        self.line = self.draw_lines(mask, fitx, ploty)

        if self.kind == 'LEFT' and self.found:
            self.previous_line = self.line
        if self.kind == 'RIGHT' and self.found:
            self.previous_line = self.line

        if reset:
            self.reset_lane_line()

        # TODO: Find a way to determine whether a line has been found or not

    def reset_lane_line(self):
        self.found = False
        self.next_coeffs = np.zeros_like(self.next_coeffs)
        self.line = np.zeros((img_size[1], img_size[0], 3), dtype = np.uint8)
        self.first = True


    def get_next_coeffs(self, mask, coeffs, kind):
        nonzero = mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Should this be first coeffs or previous coeffs
        lane_inds = ((nonzerox > coeffs[0] * (nonzeroy **2)
                     + coeffs[1] * nonzeroy + coeffs[2] - self.nextMargin)) & \
                    (nonzerox < coeffs[0] * (nonzeroy ** 2) + coeffs[1] * nonzeroy + coeffs[2] + self.nextMargin)


        # print('inside next: ', len(lane_inds))
        # TODO: If count > 1 should be self.prev_coeffs instead of self.first_coeffs
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        self.next_coeffs = np.polyfit(y, x, 2)

        if len(self.recent_coefficients) > 0:
            to_check = np.mean(np.array(self.recent_coefficients[-25:]), axis = 0)
            deviation = np.abs(np.subtract(to_check, self.next_coeffs))

            # Average the deviations
            deviation = np.average(deviation)

            if (deviation > 4):
                # If deviation is too high use previous line
                self.found = False
            else:
                self.found = True

    def get_line_pts(self, coeffs):
        ploty = np.linspace(0, self.img_height - 1, self.img_height)
        fitx = coeffs[0] * ploty ** 2 + coeffs[1] * ploty + coeffs[2]
        return fitx, ploty

    def draw_pw(self, img, pts, color):
        pts = np.int_(pts)
        for i in range(len(pts) - 1):
            x1 = pts[i][0]
            y1 = pts[i][1]
            x2 = pts[i+1][0]
            y2 = pts[i+1][1]
            cv2.line(img, (x1, y1), (x2, y2), color, 15)
        return img

    def draw_lines(self, mask, fitx, ploty):
        if self.kind == 'LEFT': color = (20, 200, 100)
        if self.kind == 'RIGHT': color = (200, 100, 20)

        out_img = np.dstack((mask, mask, mask)) * 255
        window_img = np.zeros_like(out_img)

        line_window1 = np.array([np.transpose(np.vstack([fitx - self.nextMargin, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx + self.nextMargin, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))

        lane_points = np.array(list(zip(fitx, ploty)), np.int32)
        # draw the lane
        cv2.fillPoly(window_img, np.int_([line_pts]), (0, 20, 200))
        out_img = self.draw_pw(out_img, lane_points, color)
        return out_img


    def get_initial_coeffs(self, mask, kind):
        histogram = np.sum(mask[int(mask.shape[0] / 2):, :], axis=0)
        self.out_img = np.dstack((mask, mask, mask)) * 255
        midpoint = np.int(histogram.shape[0] / 2)

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

            # lane_inds.append(good_inds)
            # print('length good inds: ', len(good_inds))
            # recenter onto the mean position if we found > minpix
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

            # If we get more than 5 good indices then we append
            if len(good_inds) > 5:
                lane_inds.append(good_inds)

            # print('inside first: ', len(good_inds))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # extract x pixel positions:
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        # Fit a second order polynomial
        self.initial_coeffs = np.polyfit(y, x, 2)

    def get_curvature(self, fitx, ploty):
        """
        Calculates the curvature using r = (a + (dy/dx)^2]^3/2 / |d^2y/dx^2|
        adds curvature to self.curvature, 
        :return: curvature appended as self.curvature
        """
        ym_per_pix = 1 / self.y_pixels_per_meter
        xm_per_pix = 1 / self.x_pixels_per_meter

        y_eval = np.max(ploty)
        fit_cr = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)


        self.curvature =  ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        print('self.curvature: ', self.curvature, 'for: ', self.kind)

