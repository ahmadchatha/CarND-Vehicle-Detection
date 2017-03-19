import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import cv2
import glob
import time 

from helpers import *
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# switch between training and test
training_mode = False
video_mode = True

# parameters.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
prev_rects = []


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space, heatmap):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    boxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return boxes

# processing video frames. only 1 scale was used for faster processing.
def process_image(img):
	global prev_rects
	box_1= find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space,True)
	#box_2 = find_cars(img, 400, 464, 1.0, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space,True)
	#box_3 = find_cars(img, 432, 560, 2.0, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space,True)
	boxes = box_1#+box_2

	heatmap_img = np.zeros_like(img[:,:,0])
	heatmap_img = add_heat(heatmap_img, boxes)
	prev_rects.append(heatmap_img)
	if len(prev_rects) > 35:
		prev_rects = prev_rects[len(prev_rects)-35:]
	averaged_heatmap = sum(prev_rects)
	heatmap_img = apply_threshold(averaged_heatmap, 45)

	labels = label(heatmap_img)
	draw_img = draw_labeled_bboxes(img, labels)
	return draw_img

# Training mode is on
if training_mode:
	# grab car and non car images
	cars = glob.glob('vehicles/**/*.png')
	noncars = glob.glob('non-vehicles/**/*.png')

	print('Total cars: ' +  str(len(cars)))
	print('Total noncars: '+ str(len(noncars)))
	sample_size = 5000
	cars = cars[0:sample_size]
	notcars = noncars[0:sample_size]

	# extract features
	t=time.time()
	car_features = extract_features(cars, color_space=color_space, 
									orient=orient, 
	                                pix_per_cell=pix_per_cell, 
	                                cell_per_block=cell_per_block, 
	                        	    hog_channel=hog_channel)
	notcar_features = extract_features(noncars, color_space=color_space, 
									   orient=orient, 
									   pix_per_cell=pix_per_cell, 
									   cell_per_block=cell_per_block, 
	                        		   hog_channel=hog_channel)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to extract HOG features...')
	# Create an array stack of feature vectors
	X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)

	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
	    scaled_X, y, test_size=0.2, random_state=rand_state)

	print('Using:',orient,'orientations',pix_per_cell,
	    'pixels per cell and', cell_per_block,'cells per block')
	print('Feature vector length:', len(X_train[0]))
	# Use a linear SVC 
	svc = LinearSVC()
	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
	# Check the prediction time for a single sample
	t=time.time()
	n_predict = 10
	print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
	print('For these',n_predict, 'labels: ', y_test[0:n_predict])
	t2 = time.time()
	print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

	# Saving classifier for later
	svc_file = "svc_pickle.p"

	dist_pickle = {}
	dist_pickle["svc"]= svc
	dist_pickle["scaler"]= X_scaler
	dist_pickle["orient"]= orient
	dist_pickle["pix_per_cell"]= pix_per_cell
	dist_pickle["cell_per_block"]=cell_per_block

	pickle.dump(dist_pickle,  open(svc_file, 'wb'))


dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = (32, 32)
hist_bins = 32
ystart = 400
ystop = 656
scale = 1.5


# for test images
if not video_mode and not training_mode:
	for image in glob.glob("test_images/*.jpg"):
		print(image)
		filename = image.split('/')[1]
		filename = filename.split('.')[0]
		img = mpimg.imread(image)

		# testing various scales for test images
		box_1= find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space,True)
		box_2 = find_cars(img, ystart, ystop, 1.0, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space,True)
		box_3 = find_cars(img, ystart, ystop, 2.0, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space,True)
		boxes = box_1+box_2+box_3

		img_copy1 = np.copy(img)
		for box in boxes:
			cv2.rectangle(img_copy1,(box[0][0], box[0][1]),(box[1][0],box[1][1]),(0,0,255),6)
		plt.imshow(img_copy1)
		plt.savefig('output_images/' + filename + '_all_boxes.png')

		heatmap_img = np.zeros_like(img[:,:,0])
		heatmap_img = add_heat(heatmap_img, boxes)
		plt.imshow(heatmap_img, cmap='hot')
		plt.savefig('output_images/' + filename + '_heatmap.png')

		heatmap_img = apply_threshold(heatmap_img, 4)
		plt.imshow(heatmap_img, cmap='hot')
		plt.savefig('output_images/' + filename + '_heatmap.png')

		labels = label(heatmap_img)
		plt.imshow(labels[0], cmap='gray')
		plt.savefig('output_images/' + filename + '_labels.png')

		draw_img = draw_labeled_bboxes(img, labels)

		plt.imshow(draw_img)
		plt.savefig('output_images/' + filename + '.png')

# for video processing
else:
	video = VideoFileClip("project_video.mp4")
	output_clip= video.fl_image(process_image)
	output_clip.write_videofile("solution_video.mp4", audio=False)







