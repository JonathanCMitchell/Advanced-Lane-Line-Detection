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

# Process many images from pandas directory
for i in range(200):
    print('COUNT: ', i)
    row = df.iloc[[i]]
    impath = df.iloc[[i]]['image_path'].values[0]
    img = mpimg.imread(impath)
    result = lf.process_image(img)
    plt.imshow(result)
    cv2.imwrite('./results/' + str(i) + 'processed' + '.jpg', cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


# plt.imshow(processed_img)
# plt.show()

