import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from feature_functions import *
import pickle
from skimage.feature import hog
import glob

def window_search(img, clf, scaler, window_scale=(1, 1), color_space='RGB', x_start_stop=[None, None],
                  y_start_stop=[None, None], spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                  cell_per_block=2, hog_channel='All', spatial_feat=True, hist_feat=True, hog_feat=True):
    # If x and y start/stop positions are not defined, set them as image dimensions
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Remove unnecessary parts of the image
    cropped_img = img[y_start_stop[0]:y_start_stop[1], x_start_stop[0]:x_start_stop[1]]

    # Put the image in the right colorspace
    color_space_img = color_convert(cropped_img, color_space)

    # Training images are 64x64, so we look at 64x64 windows.
    # Different window scales are accomplished by rescaling the image
    resized = cv2.resize(color_space_img,
                         (np.int(color_space_img.shape[1] / window_scale[1]),
                          np.int(color_space_img.shape[0] / window_scale[0])))

    #TODO don't calculate all chanels when we don't need to
    # Compute the HOG features for the entire image just once
    hog1 = get_hog_features(resized[:, :, 0], orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(resized[:, :, 1], orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(resized[:, :, 2], orient, pix_per_cell, cell_per_block, feature_vec=False)

    blocks_per_window = np.int(64 / pix_per_cell - cell_per_block + 1)  # How many HOG blocks each window contains

    nx_blocks = hog1.shape[1]
    ny_blocks = hog1.shape[0]

    nx_windows = nx_blocks - blocks_per_window + 1
    ny_windows = ny_blocks - blocks_per_window + 1

    windows = []

    for xs, ys in np.ndindex((nx_windows, ny_windows)):
        if hog_channel==1:
            hog_features = hog1[ys:ys+blocks_per_window, xs:xs+blocks_per_window, :, :, :].ravel()
        elif hog_channel==2:
            hog_features = hog2[ys:ys + blocks_per_window, xs:xs + blocks_per_window, :, :, :].ravel()
        elif hog_channel==3:
            hog_features = hog3[ys:ys + blocks_per_window, xs:xs + blocks_per_window, :, :, :].ravel()
        else: # All channels
            hog_features = []
            hog_features.extend(hog1[ys:ys + blocks_per_window, xs:xs + blocks_per_window, :, :, :].ravel())
            hog_features.extend(hog2[ys:ys + blocks_per_window, xs:xs + blocks_per_window, :, :, :].ravel())
            hog_features.extend(hog3[ys:ys + blocks_per_window, xs:xs + blocks_per_window, :, :, :].ravel())

        subimg = resized[pix_per_cell*ys:pix_per_cell*ys+64, pix_per_cell*xs:pix_per_cell*xs+64, :]

        # Get color features
        spatial_features = cv2.resize(img, spatial_size).ravel()
        hist_features = color_hist(subimg, nbins=hist_bins)

        # Scale features and make a prediction
        test_features = scaler.transform(
            np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))


        test_prediction = clf.predict(test_features)

        window_left = int((pix_per_cell * xs * window_scale[0] + x_start_stop[0]))
        window_top = int((pix_per_cell * ys * window_scale[1] + y_start_stop[0]))
        window_right = int(((pix_per_cell * xs + 64)* window_scale[0] + x_start_stop[0]))
        window_bottom = int(((pix_per_cell * ys + 64) * window_scale[1] + y_start_stop[0]))

        if test_prediction:
            windows.append(((window_left, window_top), (window_right, window_bottom)))

    return windows



def draw_boxes(img, bboxes, color=(0, 0, 1), thickness=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thickness)
    return imcopy

def find_cars(img):
    dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    color_space = dist_pickle["color_space"]

    small_windows = window_search(img, clf=svc, scaler=X_scaler, color_space=color_space, spatial_size=spatial_size,
                                  window_scale=(1, 1), orient=orient, pix_per_cell=pix_per_cell,
                                  cell_per_block=cell_per_block, hog_channel='All', hist_bins=hist_bins,
                                  y_start_stop=[400, 500])

    med_windows = window_search(img, clf=svc, scaler=X_scaler, color_space=color_space, spatial_size=spatial_size,
                                window_scale=(1.2, 1.2), orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block, hog_channel='All', hist_bins=hist_bins,
                                y_start_stop=[400, 550])

    large_windows = window_search(img, clf=svc, scaler=X_scaler, color_space=color_space, spatial_size=spatial_size,
                                  window_scale=(1.5, 1.5), orient=orient, pix_per_cell=pix_per_cell,
                                  cell_per_block=cell_per_block, hog_channel='All', hist_bins=hist_bins,
                                  y_start_stop=[400, None])

    huge_windows = window_search(img, clf=svc, scaler=X_scaler, color_space=color_space, spatial_size=spatial_size,
                                 window_scale=(1.8, 1.8), orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block, hog_channel='All', hist_bins=hist_bins,
                                 y_start_stop=[400, None])

    windows =  small_windows + med_windows + large_windows + huge_windows

    return windows

def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    positive_windows = []

    color_cvt = color_convert(img, color_space)

    for window in windows:
        feature_image = cv2.resize(color_cvt[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        img_features = []
        if spatial_feat:
            spatial_features = cv2.resize(feature_image, spatial_size)
            img_features.append(spatial_features)
        if hist_feat:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            img_features.append(hist_features)
        if hog_feat:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):

                    hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            img_features.append(hog_features)
        features = np.concatenate(img_features)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            positive_windows.append(window)
    return positive_windows

if __name__ == '__main__':
    for image in glob.glob("test_images/*.jpg"):
        img = mpimg.imread(image)
        img = img.astype(np.float32) / 255

        windows = find_cars(img)

        pickle.dump(windows, open("windows.p", 'wb'))
        aaa = draw_boxes(img, windows)
        plt.imshow(aaa)
        plt.show()



