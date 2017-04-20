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
        self.left_recent_centers_averaged = []
        self.right_recent_centers_averaged = []
        self.recent_left_centers = []
        self.recent_right_centers = []
        self.recent_centers_averaged = []
        self.first = True
        self.leftFound = False
        self.rightFound = False
        self.usePrevLeft = False
        self.usePrevRight = False
        self.leftAnchor = None
        self.rightAnchor = None
        self.previous_right_lane = None
        self.previous_left_lane = None
        self.previous_y_values_left = None
        self.previous_y_values_right = None


    def reset_lane_line(self):
        self.reset = True
        self.first = True
        self.leftFound = False
        self.rightFound = False
        # TODO: Add self.polynomial coefficients to be false

    def find_lane_line(self, mask, reset = False):
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

        left_centers, right_centers = self.find_window_centroids(mask,
                                                      window_width=self.window_width,
                                                      window_height=self.window_height,
                                                      margin_width=15,
                                                      margin_height=25,
                                                      reset = False)

        # Fit lines and return lanes from polyfit
        left_lane, right_lane, y_values_left, y_values_right = self.fit_lines(left_centers,
                                                                              right_centers,
                                                                              self.window_width,
                                                                              self.window_height)

        # draw lines using polyfill
        road_lines = self.draw_lines(left_lane, right_lane, y_values_left, y_values_right)

        return road_lines


    def find_window_centroids(self, mask,
                              window_width,
                              window_height,
                              margin_width,
                              margin_height,
                              reset):
        img_width = self.img_size[0]  # 500
        img_height = self.img_size[1]  # 600

        window = np.ones((window_width))
        left_x_vals = []
        right_x_vals = []

        window_vertical_start = int((3 / 4) * img_height)
        window_horizontal_start = int(img_width / 2)

        left_sum = np.sum(mask[window_vertical_start:, :window_horizontal_start], axis=0)
        left_signal = np.convolve(window, left_sum)
        left_max = np.argmax(left_signal) - window_width/2

        right_sum = np.sum(mask[window_vertical_start:, window_horizontal_start:], axis=0)
        right_signal = np.convolve(window, right_sum)
        right_max = np.argmax(right_signal) + window_horizontal_start - window_width/2

        good_rights = right_signal[right_signal > 200]
        good_lefts = left_signal[left_signal > 200]

        length_good_rights = len(good_rights)
        length_good_lefts = len(good_lefts)

        if length_good_lefts > 20:
            self.rightFound = True
        if length_good_rights > 20:
            self.leftFound = True

        # If we are on the first image set self.first equal to True
        if self.first:
            self.leftAnchor = left_max
            self.rightAnchor = right_max
            left_x_vals.append(left_max)
            right_x_vals.append(right_max)

        if reset:
            self.leftAnchor = left_max
            self.rightAnchor = right_max

        if not self.first:
            for level in range(1, int(img_height / window_height)):
                window_start_vertical = img_height - (level + 1) * (window_height)
                window_end_vertical = img_height - (level * window_height)

                image_layer = np.sum(mask[window_start_vertical: window_end_vertical, :], axis=0)
                conv_signal = np.convolve(window, image_layer)

                offset = int(window_width / 2)
                l_min_index = int(max(left_max - (offset + margin_width), 0))
                l_max_index = int(min(left_max + (offset + margin_width), img_width))
                l_max_arg = np.argmax(conv_signal[l_min_index:l_max_index])

                # try this one or the next
                l_center = np.argmax(conv_signal[l_min_index: l_max_index]) + l_min_index

                r_min_index = int(max(right_max - (offset + margin_width), 0))
                r_max_index = int(min(right_max + (offset + margin_width), img_width))

                # Try this one or the next
                r_center = np.argmax(conv_signal[r_min_index: r_max_index]) + r_min_index
                # r_center = np.argmax(conv_signal[r_min_index: r_max_index]) + (r_min_index - offset)

                if abs(self.leftAnchor - l_center) < 20:
                    left_x_vals.append(l_center)
                if abs(self.rightAnchor - r_center) < 20:
                    right_x_vals.append(r_center)

        elif self.first:
            for level in range(1, int(img_height / window_height)):
                window_start_vertical = img_height - (level + 1) * (window_height)
                window_end_vertical = img_height - (level * window_height)

                image_layer = np.sum(mask[window_start_vertical: window_end_vertical, :], axis=0)
                conv_signal = np.convolve(window, image_layer)

                offset = int(window_width / 2)
                l_min_index = int(max(left_max - (offset + margin_width) , 0))
                l_max_index = int(min(left_max + (offset + margin_width), img_width))
                l_max_arg = np.argmax(conv_signal[l_min_index:l_max_index])

                # try this one or the next
                l_center = np.argmax(conv_signal[l_min_index: l_max_index]) + l_min_index

                r_min_index = int(max(right_max - (offset + margin_width), 0))
                r_max_index = int(min(right_max + (offset + margin_width), img_width))

                # Try this one or the next
                r_center = np.argmax(conv_signal[r_min_index: r_max_index]) + r_min_index
                # r_center = np.argmax(conv_signal[r_min_index: r_max_index]) + (r_min_index - offset)

                left_x_vals.append(l_center)
                right_x_vals.append(r_center)




        left_centers = moving_average_scale(left_x_vals)
        right_centers = moving_average_scale(right_x_vals)


        self.recent_left_centers = np.concatenate((self.recent_left_centers, left_centers), axis = 0)
        self.recent_right_centers = np.concatenate((self.recent_right_centers, right_centers), axis = 0)

        if not self.rightFound:
            self.recent_right_centers = np.roll(self.recent_right_centers, len(right_centers))
        if not self.leftFound:
            self.recent_left_centers = np.roll(self.recent_left_centers, len(left_centers))

        return self.recent_left_centers, self.recent_right_centers


    def fit_lines(self, left_centers, right_centers, window_width, window_height):
        vert_start = self.img_size[1] - window_height / 2
        vert_stop = window_height
        y_values_left = np.linspace(vert_start, vert_stop, len(left_centers), dtype=np.float32)
        y_values_right = np.linspace(vert_start, vert_stop, len(right_centers), dtype=np.float32)

        if len(y_values_right) > 2 and len(y_values_left) > 2:
            # fit to a polynomial (ax^2 + bx + c)
            left_fit = np.polyfit(y_values_left, left_centers, 2)
            left_fitx = left_fit[0] * (y_values_left ** 2) + left_fit[1] * y_values_left + left_fit[2]
            left_fitx = np.array(left_fitx, np.int32)

            right_fit = np.polyfit(y_values_right, right_centers, 2)
            right_fitx = right_fit[0] * (y_values_right ** 2) + right_fit[1] * y_values_right + right_fit[2]
            right_fitx = np.array(right_fitx, np.int32)

            shift = 0
            left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width / 10 + shift, left_fitx[::-1] + window_width / 10 + shift),axis=0),np.concatenate((y_values_left, y_values_left[::-1]), axis=0))), dtype=np.int32)
            right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width / 10 + shift, right_fitx[::-1] + window_width / 10 + shift),axis=0), np.concatenate((y_values_right, y_values_right[::-1]), axis=0))), dtype=np.int32)

            self.previous_right_lane = right_lane
            self.previous_left_lane = left_lane
            self.previous_y_values_left = y_values_left
            self.previous_y_values_right = y_values_right
        else:
            right_lane = self.previous_right_lane
            left_lane = self.previous_left_lane
            y_values_left = self.previous_y_values_left
            y_values_right = self.previous_y_values_right


        return left_lane, right_lane, y_values_left, y_values_right


    #
    #
    # def fit_lines(self, window_centroids, leftx, rightx, window_width, window_height):
    #     # TODO: uneven lengths for y values
    #     if len(leftx) == len(rightx):
    #         vert_start = self.img_size[1] - window_height / 2
    #         vert_stop = window_height
    #         y_values_left = np.linspace(vert_start, vert_stop, len(leftx), dtype=np.float32)
    #         y_values_right = np.linspace(vert_start, vert_stop, len(rightx), dtype=np.float32)
    #     # fit to a polynomial (ax^2 + bx + c)
    #     left_fit = np.polyfit(y_values_left, leftx, 2)
    #     left_fitx = left_fit[0] * (y_values_left ** 2) + left_fit[1] * y_values_left + left_fit[2]
    #     left_fitx = np.array(left_fitx, np.int32)
    #
    #     right_fit = np.polyfit(y_values_right, rightx, 2)
    #     right_fitx = right_fit[0] * (y_values_right ** 2) + right_fit[1] * y_values_right + right_fit[2]
    #     right_fitx = np.array(right_fitx, np.int32)
    #     # TODO: Remove this concatenate piece
    #     shift = 0
    #     left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width / 10 + shift, left_fitx[::-1] + window_width / 10 + shift),axis=0),np.concatenate((y_values_left, y_values_left[::-1]), axis=0))), dtype=np.int32)
    #     right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width / 10 + shift, right_fitx[::-1] + window_width / 10 + shift),axis=0), np.concatenate((y_values_right, y_values_right[::-1]), axis=0))), dtype=np.int32)
    #     return left_lane, right_lane, left_fitx, right_fitx, y_values_left, y_values_right

    def draw_lines(self, left_lane, right_lane, y_values_left, y_values_right):
        lanes = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        cv2.fillPoly(lanes, [left_lane], color=[255, 0, 0])
        cv2.fillPoly(lanes, [right_lane], color=[0, 255, 0])

        return lanes
    #
    # def get_line_pts(self, mask, window_centroids):
    #     """
    #     Here we find the points that will be used to draw the left and right windows
    #     """
    #     # Points we will use for drawing on each level
    #     left_points = np.zeros_like(mask)
    #     right_points = np.zeros_like(left_points)
    #     left_x = []
    #     right_x = []
    #     #         print('inside get_line_pts window_centroids are :', window_centroids)
    #     #         print('inside get_line_pts window_centroids type :', type(window_centroids))
    #     for level in range(0, len(window_centroids)):
    #         # window_mask if a function to draw window boxes
    #         left_x.append(window_centroids[level][0])
    #         right_x.append(window_centroids[level][1])
    #         left_mask = draw_window_box(mask, self.window_width, self.window_height,
    #                                     window_centroids[level][0],
    #                                     level)
    #         right_mask = draw_window_box(mask, self.window_width, self.window_height,
    #                                      window_centroids[level][1],
    #                                      level)
    #         left_points[(left_points == 255) | ((left_mask == 1))] = 255
    #         right_points[(right_points == 255) | ((right_mask == 1))] = 255
    #     return [left_points, right_points], [left_x, right_x]

