import pickle as pickle
import pandas as pd
import settings
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

# PROCESS MULTIPLE IMAGES
# for i in range(0, 20):
#     print('COUNT: ', i)
#     row = df.iloc[[i]]
#     impath = df.iloc[[i]]['image_path'].values[0]
#     img = mpimg.imread(impath)
#     process_image = lf.process_image(img)
#     image = process_image
#     cv2.imwrite('./results/' + str(i) + 'drawn_on' + '.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#  # MOVIEPY project video
from moviepy.editor import VideoFileClip

test_output = 'project_video_output_averaged_new.mp4'
clip1 = VideoFileClip("project_video.mp4")
lf = LaneFinder(settings.ORIGINAL_SIZE, settings.UNWARPED_SIZE, camera_matrix, dist_coeffs,
                        M, x_pixels_per_meter, y_pixels_per_meter)
white_clip = clip1.fl_image(lf.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(test_output, audio=False)

# from moviepy.editor import VideoFileClip
#
# test_output = 'challenge_video_output4_new.mp4'
# clip1 = VideoFileClip("harder_challenge_video.mp4")
# lf = LaneFinder(settings.ORIGINAL_SIZE, settings.UNWARPED_SIZE, camera_matrix, dist_coeffs,
#                         M, x_pixels_per_meter, y_pixels_per_meter)
# white_clip = clip1.fl_image(lf.process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(test_output, audio=False)