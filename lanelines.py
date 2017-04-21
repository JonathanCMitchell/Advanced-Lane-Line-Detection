import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import cv2
import numpy as np
import pickle as pickle
import pandas as pd
import glob
import settings
from laneFinder import LaneFinder


data = pickle.load( open( "camera_calibration.p", "rb" ) )
camera_matrix = data['mtx']
dist_coeffs = data['dist']

perspective_transform_data = pickle.load(open("perspective.p", 'rb'))
x_pixels_per_meter = perspective_transform_data['x_pixels_per_meter']
y_pixels_per_meter = perspective_transform_data['y_pixels_per_meter']
M = perspective_transform_data['homography_matrix']
src_pts = perspective_transform_data['source_points']

# read in dataframe
df = pd.read_csv('./data/driving.csv')


lf = LaneFinder(settings.ORIGINAL_SIZE, settings.UNWARPED_SIZE, camera_matrix, dist_coeffs,
                        M, x_pixels_per_meter, y_pixels_per_meter)

# impath = df.iloc[[17]]['image_path'].values[0]
# img = mpimg.imread(impath)
# processed_img = lf.process_image(img)

# Process single individual image
# impath1 = './test_images/test2.jpg'
# impath2 = './test_images/test3.jpg'
# img1 = mpimg.imread(impath1)
# img2 = mpimg.imread(impath2)
#
# lf.find_lane(img1)
# lf.find_lane(img2)

# returned = lf.left_line.find_lane_line(lf.total_mask)
# plt.imshow(returned)
# plt.title('test5')
# plt.show()


# PROCESS MULTIPLE IMAGES
# for i in range(950, 1005):
#     print('COUNT: ', i)
#     row = df.iloc[[i]]
#     impath = df.iloc[[i]]['image_path'].values[0]
#     img = mpimg.imread(impath)
#     image = lf.process_image(img)
#     cv2.imwrite('./results/' + str(i) + 'drawn_on' + '.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

 # MOVIEPY
from moviepy.editor import VideoFileClip

test_output = 'project_video_output2.mp4'
clip1 = VideoFileClip("project_video.mp4")
lf = LaneFinder(settings.ORIGINAL_SIZE, settings.UNWARPED_SIZE, camera_matrix, dist_coeffs,
                        M, x_pixels_per_meter, y_pixels_per_meter)
white_clip = clip1.fl_image(lf.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(test_output, audio=False)

# warped = lf.warp(img)
# plt.plot(returned)
# plt.imshow(mask)
# plt.show()
#
# # Process many images from pandas directory
# for i in range(90, 95):
#     print('COUNT: ', i)
#     row = df.iloc[[i]]
#     impath = df.iloc[[i]]['image_path'].values[0]
#     img = mpimg.imread(impath)
#
#     road_lines = lf.find_lane(img)
#     # TODO: Uncomment below for testing
#     # road_lines, weighted, mask  = lf.find_lane(img)
#     # cv2.imwrite('./results/' + str(i) + 'road_lines' + '.jpg', cv2.cvtColor(road_lines, cv2.COLOR_BGR2RGB))
#     # cv2.imwrite('./results/' + str(i) + 'weighted' + '.jpg', cv2.cvtColor(weighted, cv2.COLOR_BGR2RGB))
#     # cv2.imwrite('./results/' + str(i) + 'mask' + '.jpg', mask)
#     warped = lf.warp(img)
#     drawn_on = lf.add_weighted(warped, road_lines)
#     cv2.imwrite('./results/' + str(i) + 'drawn_on' + '.jpg', cv2.cvtColor(drawn_on, cv2.COLOR_BGR2RGB))
#


# for i in range(10):
#     print('COUNT: ', i)
#     row = df.iloc[[i]]
#     impath = df.iloc[[i]]['image_path'].values[0]
#     img = mpimg.imread(impath)
#     road_lines, weighted, mask  = lf.find_lane(img)
#     plt.imshow(mask)
#     print('mask shape: ', mask.shape)
#     cv2.imwrite('./results/' + str(i) + 'road_lines' + '.jpg', cv2.cvtColor(road_lines, cv2.COLOR_BGR2RGB))
#     cv2.imwrite('./results/' + str(i) + 'weighted' + '.jpg', cv2.cvtColor(weighted, cv2.COLOR_BGR2RGB))
#     cv2.imwrite('./results/' + str(i) + 'mask' + '.jpg', mask)

# plt.imshow(processed_img)
# plt.show()

