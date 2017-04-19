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

    def __init__(self, img_size, x_pixels_per_meter, y_pixels_per_meter, kind='none'):
        self.img_size = img_size
        self.x_pixels_per_meter = x_pixels_per_meter
        self.y_pixels_per_meter = y_pixels_per_meter
        self.kind = kind  # specify left or right lane
        self.line_mask = np.ones((img_size[1], img_size[0]), dtype=np.uint8)  # create 2D line mask
        self.recent_centers = []
        self.smooth_factor = 15
        self.prev_left_center = 185
        self.prev_right_center = 370
        self.left_recent_centers_averaged = []
        self.right_recent_centers_averaged = []
        self.recent_centers_averaged = []
        self.count = 0
        self.found = False
        self.usePrevLeft = False
        self.usePrevRight = False


    def find_lane_line(self, mask, FLAG):
        """
        Perform convolutional sliding window
        Identify points of interest and save them
        Then pass the saved points to fit_lane_line where you apply polynomial fit
        input: mask (shape [500, 600])
        """
        # TODO: Create conditional in case not enough centroids are found, to use previous centroids



        img_width = self.img_size[0]  # 500
        img_height = self.img_size[1]  # 600

        self.window_width = 40
        self.window_height = 70

        window_centroids = self.find_window_centroids(mask,
                                                      window_width=self.window_width,
                                                      window_height=self.window_height,
                                                      margin_width=35,
                                                      #                                                  margin_height = 35,
                                                      margin_height=25,
                                                      FLAG=FLAG)

        #         print('window_centroids in find_lane_lines: ', window_centroids)
        [left_pts, right_pts], [leftx, rightx] = self.get_line_pts(mask, window_centroids)

        # To check accuracy of window centroids
        weighted, mask = draw_lines_weighted(left_pts, right_pts, mask)

        left_lane, right_lane, left_fitx, right_fitx, y_values = self.fit_lines(window_centroids,
                                                                                leftx,
                                                                                rightx,
                                                                                self.window_width,
                                                                                self.window_height)

        road_lines = self.draw_lines(left_lane, right_lane, left_fitx, right_fitx, y_values)
        # points now is an array containing [left_points, right_points]
        return road_lines, weighted, mask

    #         return road_lines, weighted, mask


    def find_window_centroids(self, mask,
                              window_width,
                              window_height,
                              margin_width,
                              margin_height,
                              FLAG):
        self.count += 1
        if self.count > 20:
            self.count = 0
            self.prev_left_center = settings.RESET_LEFT_CENTER
            self.prev_right_center = settings.RESET_RIGHT_CENTER
        img_width = self.img_size[0]  # 500
        img_height = self.img_size[1]  # 600

        window = np.ones((window_width))
        window_centroids = []

        left_x_vals = []
        right_x_vals = []

        window_vertical_start = int((3 / 4) * img_height) + 7
        window_horizontal_start = int(img_width / 2) - 62

        #         print('window_vertical_start: ', window_vertical_start) # 450
        #         print('window_horizontal_start: ', window_horizontal_start) # 250

        left_sum = np.sum(mask[window_vertical_start:, :window_horizontal_start], axis=0)
        left_signal = np.convolve(window, left_sum)
        left_max = np.argmax(left_signal)

        right_sum = np.sum(mask[window_vertical_start:, window_horizontal_start:], axis=0)
        right_signal = np.convolve(window, right_sum)
        right_max = np.argmax(right_signal) + window_horizontal_start

        # from scipy.stats import describe

        #         print('type of left_signal: ', type(left_signal))
        #         print('left signal shape: ', left_signal.shape)
        #         print('mean value of left signal: ', np.mean(left_signal))
        #         print('left signal describe ', describe(left_signal))
        # print('right_signal shape: ', right_signal.shape)
        # print('describe right signal: ', describe(right_signal))
        good_rights = right_signal[right_signal > 100]
        good_lefts = left_signal[left_signal > 100]

        print('good_rights: ', len(good_rights))
        print('good_lefts: ', len(good_lefts))


        if (len(good_rights)) < 30:
            self.usePrevRight = True

            # TODO: Take this out
            return self.recent_centers_averaged
        if (len(good_lefts)) < 30:
            self.usePrevLeft = True

            # reset line somehow
            print('we have bad rights')

            self.right_line_isbad = True
        print('good_rights: ', good_rights)
        #         print('left_signal: ', left_signal)
        #         good_left_signals = left_signal[np.where(left_signal < 650)] = 0


        #         print('good left signals: ', good_left_signals)
        # #         print('length good left signals: ', len(good_left_signals))
        #         print('type good left signals: ', type(good_left_signals))

        #         print('length right_signal :', len(right_signal))
        #         print('right_signal: ', right_signal)

        #         good_right_signals = right_signal[np.where(right_signal < 650)] = 0
        #         print('good right signals: ', good_right_signals)
        #         print('length good right signals: ', len(good_right_signals)
        #         print('type good right signals: ', type(good_right_signals)


        #         print('left_max: ', left_max)
        #         print('right_max: ', right_max)

        # TODO: After ~30 runs, reset the prev_right_center back to default

        # conditionals to check first centroid
        if abs(self.prev_right_center - right_max) < 20:
            self.prev_right_center = right_max
            right_x_vals.append(right_max)
        else:
            right_x_vals.append(self.prev_right_center)

        if abs(self.prev_left_center - left_max) < 20:
            self.prev_left_center = left_max
            left_x_vals.append(left_max)
        else:
            left_x_vals.append(self.prev_left_center)
            # append the previous left center

        for level in range(1, int(img_height / window_height)):
            window_start_vertical = img_height - (level + 1) * window_height
            window_end_vertical = img_height - (level * window_height)

            image_layer = np.sum(mask[window_start_vertical: window_end_vertical, :], axis=0)
            conv_signal = np.convolve(window, image_layer)

            left_min_idx = int(max(left_max - margin_width, 0))
            left_max_idx = int(min(left_max + margin_width, img_width))

            right_min_idx = int(max(right_max - margin_width, 0))
            right_max_idx = int(min(right_max + margin_width, img_width))

            left_center = np.argmax(conv_signal[left_min_idx:left_max_idx]) + left_min_idx
            right_center = np.argmax(conv_signal[right_min_idx:right_max_idx]) + right_min_idx

            # add in something about whether to collect the centroid or not!
            if abs(self.prev_right_center - right_center) < 20:
                self.prev_right_center = right_center
                right_x_vals.append(right_center)

            if abs(self.prev_left_center - left_center) < 20:
                self.prev_left_center = left_center
                left_x_vals.append(left_center)

                #         print('left_x_vals: ', left_x_vals)
                #         print('right_x_vals: ', right_x_vals)

        left_centers = moving_average_scale(left_x_vals)
        right_centers = moving_average_scale(right_x_vals)

        if len(right_centers) > len(left_centers):
            # add left centers
            left_centers = add_recent_centers(len(right_centers), left_centers, self.left_recent_centers_averaged,
                                                      self.smooth_factor)
        elif len(left_centers) > len(right_centers):
            # add right centers
            right_centers = add_recent_centers(len(left_centers), right_centers, self.right_recent_centers_averaged,
                                                       self.smooth_factor)

        # print('left_centers after add_recent_centers: ', left_centers)
        #         print('right_centers after add_recent_centers: ', right_centers)
        #         print('self.right_recent_centers averaged: ', self.right_recent_centers_averaged)

        # once left and right centers are squared away!
        if len(left_centers) == len(right_centers):
            centroids = [[a, b] for a, b in zip(left_centers, right_centers)]
            for item in centroids:
                self.recent_centers.append(item)

        left_recent_centers = []
        right_recent_centers = []
        for i in reversed(range(len(self.recent_centers))):
            if len(left_recent_centers) < self.smooth_factor:
                left_recent_centers.append(self.recent_centers[i][0])
                right_recent_centers.append(self.recent_centers[i][1])

        # TODO: Fix: The recent centers just get reassigned each time, not added
        self.left_recent_centers_averaged = moving_average_scale(left_recent_centers)
        self.right_recent_centers_averaged = moving_average_scale(right_recent_centers)

        #         print('left_recent_centers: ', left_recent_centers)
        #         print('right_recent_centers: ', right_recent_centers)

        #         print('self.left_recent_centers_averaged: ', self.left_recent_centers_averaged)
        #         print('self.right_recent_centers_averaged: ', self.right_recent_centers_averaged)

        # TODO: If we have any flags for either left or right lane lines then roll:
        if self.usePrevLeft == True:
            print('usePrevLeft triggered')
            print('left_recent_centers averaged length: ', len(self.left_recent_centers_averaged))
            self.left_recent_centers_averaged = np.roll(self.left_recent_centers_averaged, len(left_recent_centers))
        if self.usePrevRight == True:
            print('usePrevRight triggered')
            print('right_recent_centers averaged length: ', len(self.right_recent_centers_averaged))
            print('self.right_recent_centers_averaged: ', self.right_recent_centers_averaged)
            self.right_recent_centers_averaged = np.roll(self.right_recent_centers_averaged, len(right_recent_centers))


        # TODO: Split this and send back recent_lefts_averaged and recent_rights_averaged
        # Then draw the lines individually
        # this way you can separate concerns for different lines
        if (len(self.left_recent_centers_averaged) & len(self.right_recent_centers_averaged)) < self.smooth_factor:
            self.recent_centers_averaged = [[a, b] for a, b in list(
                zip(self.left_recent_centers_averaged, self.right_recent_centers_averaged))]
        else:
            self.recent_centers_averaged = [[a, b] for a, b in
                                            list(zip(self.left_recent_centers_averaged[-self.smooth_factor:],
                                                     self.right_recent_centers_averaged[-self.smooth_factor:]))]

        print('RECENT_CENTERS_AVERAGED: ', self.recent_centers_averaged)

        return self.recent_centers_averaged


    def fit_lines(self, window_centroids, leftx, rightx, window_width, window_height):

        # TODO: uneven lengths for y values
        if len(leftx) == len(rightx):
            vert_start = self.img_size[1] - window_height / 2
            vert_stop = window_height
            y_values = np.linspace(vert_start, vert_stop, len(leftx), dtype=np.float32)

        # fit to a polynomial (ax^2 + bx + c)
        left_fit = np.polyfit(y_values, leftx, 2)
        left_fitx = left_fit[0] * (y_values ** 2) + left_fit[1] * y_values + left_fit[2]
        left_fitx = np.array(left_fitx, np.int32)

        right_fit = np.polyfit(y_values, rightx, 2)
        right_fitx = right_fit[0] * (y_values ** 2) + right_fit[1] * y_values + right_fit[2]
        right_fitx = np.array(right_fitx, np.int32)
        shift = 0

        left_lane = np.array(list(zip(
            np.concatenate((left_fitx - window_width / 10 + shift, left_fitx[::-1] + window_width / 10 + shift),
                           axis=0),
            np.concatenate((y_values, y_values[::-1]), axis=0))), dtype=np.int32)

        right_lane = np.array(list(zip(
            np.concatenate((right_fitx - window_width / 10 + shift, right_fitx[::-1] + window_width / 10 + shift),
                           axis=0),
            np.concatenate((y_values, y_values[::-1]), axis=0))), dtype=np.int32)

        return left_lane, right_lane, left_fitx, right_fitx, y_values

    def draw_lines(self, left_lane, right_lane, left_fitx, right_fitx, y_values):
        lanes = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        cv2.fillPoly(lanes, [left_lane], color=[255, 0, 0])
        cv2.fillPoly(lanes, [right_lane], color=[0, 255, 0])
        #         ym_per_pix = 1/self.y_pixels_per_meter
        #         xm_per_pix = 1/self.x_pixels_per_meter


        #         ym_per_pix = self.y_pixels_per_meter
        #         xm_per_pix = self.x_pixels_per_meter

        #         curve_fit_cr = np.polyfit(np.array(y_values, np.float32)
        #                                   * ym_per_pix, np.array(left_fitx, np.float32) * xm_per_pix, 2)
        #         curverad = ((1 + (2*curve_fit_cr[0] * y_values[-1] * ym_per_pix + curve_fit_cr[1]) **2) ** 1.5) / np.absolute(2 * curve_fit_cr[0])


        #         camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
        #         center_diff = (camera_center - self.img_size[0]/2) * xm_per_pix

        #         side_pos = 'left'
        #         if center_diff <= 0:
        #             side_pos = 'right'


        #         cv2.putText(lanes, 'Radius of curvature = ' + str(round(curverad, 3)) + '(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #         cv2.putText(lanes, 'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return lanes

    def get_line_pts(self, mask, window_centroids):
        """
        Here we find the points that will be used to draw the left and right windows
        """
        # Points we will use for drawing on each level
        left_points = np.zeros_like(mask)
        right_points = np.zeros_like(left_points)

        left_x = []
        right_x = []

        #         print('inside get_line_pts window_centroids are :', window_centroids)
        #         print('inside get_line_pts window_centroids type :', type(window_centroids))


        for level in range(0, len(window_centroids)):
            # window_mask if a function to draw window boxes
            left_x.append(window_centroids[level][0])
            right_x.append(window_centroids[level][1])

            left_mask = draw_window_box(mask, self.window_width, self.window_height,
                                        window_centroids[level][0],
                                        level)
            right_mask = draw_window_box(mask, self.window_width, self.window_height,
                                         window_centroids[level][1],
                                         level)

            left_points[(left_points == 255) | ((left_mask == 1))] = 255
            right_points[(right_points == 255) | ((right_mask == 1))] = 255

        return [left_points, right_points], [left_x, right_x]

