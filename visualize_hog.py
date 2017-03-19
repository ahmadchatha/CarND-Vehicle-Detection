import glob
import cv2
import numpy as np
from helpers import get_hog_features
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# HOG params
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

cars = glob.glob('vehicles/**/*.png')
noncars = glob.glob('non-vehicles/**/*.png')

print('Total cars: ' +  str(len(cars)))
print('Total noncars: '+ str(len(noncars)))
sample_size = 1
car = cars[0:sample_size]
notcar = noncars[0:sample_size]

car_img = mpimg.imread(car[0])
notcar_img = mpimg.imread(notcar[0])
gray = cv2.cvtColor(car_img, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(notcar_img, cv2.COLOR_RGB2GRAY)
features, hog_car_image = get_hog_features(gray, orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

features, hog_notcar_image = get_hog_features(gray2, orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)
plt.imshow(car_img)
plt.savefig('output_images/car_img_orig.png')
plt.imshow(notcar_img)
plt.savefig('output_images/notcar_img_orig.png')
plt.imshow(hog_car_image, cmap='gray')
plt.savefig('output_images/car_img_hog.png')
plt.imshow(hog_notcar_image, cmap='gray')
plt.savefig('output_images/notcar_img_hog.png')