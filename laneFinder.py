import cv2
import numpy as np
from laneLineFinder import LaneLineFinder

class LaneFinder():
    """
    This class manages the lane finding operation.
    Methods: 
    process_image: takes in an image (RGB) and finds the lane lines 
    then returns that image with lane lines drawn on it.
    """
    def __init__(self,
                 img_size,
                 warped_size,
                 camera_matrix,
                 dist_coeffs,
                 transform_matrix,
                 x_pixels_per_meter, y_pixels_per_meter):
        self.img_size = img_size
        self.warped_size = warped_size
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.transform_matrix = transform_matrix
        self.x_pixels_per_meter = x_pixels_per_meter
        self.y_pixels_per_meter = y_pixels_per_meter
        self.roi_mask = np.ones((self.warped_size[1], self.warped_size[0], 3), dtype=np.uint8)
        self.mask = np.zeros((self.warped_size[1], self.warped_size[0], 3), dtype=np.uint8)
        self.real_mask = np.zeros((self.warped_size[1], self.warped_size[0], 3), dtype=np.uint8)
        self.left_line = LaneLineFinder(warped_size, x_pixels_per_meter, y_pixels_per_meter, kind='LEFT')
        self.right_line = LaneLineFinder(warped_size, self.x_pixels_per_meter, self.y_pixels_per_meter, kind='RIGHT')
        self.previous_lanes = []
        self.count = 0
        self.center_diff = None
        self.previous_inner_lane = np.zeros((self.warped_size[1], self.warped_size[0], 3), dtype = np.uint8)

    def warp(self, img):
        return cv2.warpPerspective(img, self.transform_matrix, self.warped_size,
                                   flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_NEAREST)
    def unwarp(self, img):
        return cv2.warpPerspective(img, self.transform_matrix, self.img_size,
                                   flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
    def undistort(self, img):
        return cv2.undistort(img, self.camera_matrix, self.dist_coeffs)

    def add_weighted(self, base, lines):
        return cv2.addWeighted(base, 1.0, lines, 2, 0.0)

    def process_image(self, image):
        """
        Process image full pipeline applied to video stream
        input: image: original image
        output: original image with lane lines overlayed
        """
        self.find_lane(image)
        if not self.left_line.found:
            left = self.left_line.previous_line
            curve_left = self.left_line.previous_curvature

        if not self.right_line.found:
            right = self.right_line.previous_line # should be an instance
            curve_right = self.right_line.previous_curvature

        if self.left_line.found:
            left = self.left_line.line
            curve_left = self.left_line.curvature
        if self.right_line.found:
            right = self.right_line.line
            curve_right = self.right_line.curvature

        curve = curve_left or curve_right

        inner, inner_found = self.isGet_inner_lane()
        if inner_found is False:
            inner = self.previous_inner_lane
        lanes = left + right + inner

        # find center
        if self.left_line.found and self.right_line.found:
            camera_center = (self.left_line.last_fitx + self.right_line.last_fitx)/2
            self.center_diff = (camera_center - self.warped_size[0]/2) * (1 / self.x_pixels_per_meter)

        side_pos = 'left'
        if self.center_diff <= 0:
            side_pos = 'right'

        unwarp_lanes = self.unwarp(lanes)
        original_weighted = self.add_weighted(image, unwarp_lanes)
        original_weighted = self.add_curvature_and_center(original_weighted, curve, self.center_diff, side_pos)
        return original_weighted

    def add_curvature_and_center(self, img, curve, center_diff, side_pos):
        cv2.putText(img, 'Radius of curvature = ' + str(curve)[:-7] + '(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, 'Vehicle is: ' + str(center_diff)[:-9] + '(m) ' + str(side_pos) + 'of center', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    def isGet_inner_lane(self):
        if self.left_line.found and self.right_line.found:
            left_fitx = self.left_line.fitx
            ploty = self.left_line.ploty

            right_fitx = self.right_line.fitx
            inner_lane = np.array(list(zip(
                np.concatenate((left_fitx, right_fitx[::-1]), axis = 0), \
                np.concatenate((ploty, ploty[::-1]), axis = 0))), np.int32)
            img = np.zeros((self.warped_size[1], self.warped_size[0], 3), dtype=np.uint8)
            cv2.fillPoly(img, [inner_lane], color=[0, 255, 0])
            self.previous_inner_lane = img
            return img, True
        else:
            return None, False

    def find_lane(self, image, distorted=True, reset = False):
        """
        Pipeline:
        1) Undistort
        2) Perspective Transform
        3) Blur
        4) Convert to HLS and LAB and use the Luminance channel to identify yellow lines
        5) Use adaptive thresholding and morphological operations to reduce noise
        6) Execute laneFinder's find_lane_line method to find the lane lines
        """
        if reset == True:
            self.left_line.reset_lane_line()
            self.right_line.reset_lane_line()

        # 1) Undistort the image
        img = self.undistort(image)

        # 2) Apply perspective transform
        warped = self.warp(img)

        # 3) Blur
        blur_kernel = 5
        img_hls = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
        img_lab = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB)

        img_hls = cv2.medianBlur(img_hls, blur_kernel)
        img_lab = cv2.medianBlur(img_lab, blur_kernel)

        # Get structuring element for morph transforms
        # note: Select structuring element to be large enough so that it won't fit inside the objects to be removed
        large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        # The road is dark, so extract bright regions out of the image
        # If L in HLS is greater than 190 then it is bright
        # Also filter out low saturation < 50
        hls_filter = cv2.inRange(img_hls, (0, 0, 50), (30, 192, 255))

        yellow = hls_filter & (img_lab[:, :, 2].astype(np.uint8) > 127)

        # Logical not means find inverse because later on we will combine this mask with the self.mask
        roi_mask = np.logical_not(yellow).astype(np.uint8)
        # cut out the bright stuff
        roi_mask = (roi_mask & (img_hls[:, :, 1] < 245)).astype(np.uint8)

        # perform OPEN morphology (erosion + dilation) to reduce noise
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, small_kernel)

        # roi_mask is a binary mask
        # perform Dilation morphology for enhancement on larger features
        roi_mask = cv2.dilate(roi_mask, large_kernel)

        self.roi_mask[:, :, 0] = (self.left_line.line_mask | self.right_line.line_mask) & roi_mask
        self.roi_mask[:, :, 1] = self.roi_mask[:, :, 0]
        self.roi_mask[:, :, 2] = self.roi_mask[:, :, 0]

        # perform tophat (original - opening)
        tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
        black = cv2.morphologyEx(img_lab[:, :, 0], cv2.MORPH_TOPHAT, tophat_kernel)
        lanes = cv2.morphologyEx(img_hls[:, :, 1], cv2.MORPH_TOPHAT, tophat_kernel)

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        yellow_lanes = cv2.morphologyEx(img_lab[:, :, 2], cv2.MORPH_TOPHAT, rect_kernel)

        # Adaptive thresholding
        self.mask[:, :, 0] = cv2.adaptiveThreshold(black, 50, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -6)
        self.mask[:, :, 1] = cv2.adaptiveThreshold(lanes, 60, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -4)
        self.mask[:, :, 2] = cv2.adaptiveThreshold(yellow_lanes, 60, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,
                                                   -1.5)

        diff_mask = self.mask * self.roi_mask
        small_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # grab any values that are nonzero
        self.total_mask = np.any(diff_mask, axis=2).astype(np.uint8)
        # erosion on total_mask to reduce noise
        self.total_mask = cv2.morphologyEx(self.total_mask, cv2.MORPH_ERODE, small_ellipse)

        self.left_line.find_lane_line(self.total_mask)
        self.right_line.find_lane_line(self.total_mask)