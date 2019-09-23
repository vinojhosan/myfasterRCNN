import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
from tensorflow import keras as k
from tensorflow.keras import layers as kl
import tensorflow as tf
import cv2 as cv
from skimage.feature import peak_local_max
from scipy.signal import argrelextrema

np.random.seed(4)
from synthetic_dataset import ShapeDataset
import losses

IMAGE_SHAPE = [224, 224, 3]
GRID_SHAPE = [28, 28]
ANCHOR_SIZE = [32, 64, 128]
ANCHOR_ASPECT_RATIO = [0.5, 1, 2]
STRIDE = [IMAGE_SHAPE[0] // GRID_SHAPE[0],
          IMAGE_SHAPE[1] // GRID_SHAPE[1]]

grid_ratio = IMAGE_SHAPE[0]/GRID_SHAPE[0]
iou_threshold = 0.5
feature_size = 256
num_class = ShapeDataset.n_shapes

num_aspect_ratio = 3
pyramid_size = 3
channels_per_class = 5
channels_per_aspect_ratio = num_aspect_ratio * channels_per_class
channels_per_pyramid = num_aspect_ratio * num_class * channels_per_class

"""*****************************Data generation************************************************"""


def create_anchors(image_size, grid_size, anchor_size=[32, 64, 128], aspect_ratios=[0.5, 1, 2]):
    h_stride = image_size[0] // grid_size[0]
    w_stride = image_size[1] // grid_size[1]

    def get_anchors():
        anchor_list = []
        anc_id = 0
        for anc in anchor_size:
            for asp in aspect_ratios:
                h_sz = anc * 1
                w_sz = anc * asp

                bbox = [-h_sz * 0.5, -w_sz * 0.5, +h_sz * 0.5, +w_sz * 0.5,
                        anc_id, anc, asp]  # x1, y1, x2, y2, anc_id, anchor_size, aspect ratio
                anchor_list.append(bbox)
                anc_id += 1

        return np.array(anchor_list)

    def remove_boundary_outliers(anch_list):
        cond = lambda x: x[0] >= 0 and x[1] >= 0 and x[2] <= image_size[0] and x[3] <= image_size[1]
        corrected_list = list(filter(cond, anch_list))
        return corrected_list

    ideal_anchors = get_anchors()
    total_anchors = None
    for r in range(0, grid_size[0]):
        for c in range(0, grid_size[1]):

            r_centre = (r + 0.5) * h_stride
            c_centre = (c + 0.5) * w_stride

            current_anchors = np.zeros((len(anchor_size) * len(aspect_ratios), 9))
            current_anchors[:, 0] = ideal_anchors[:, 0] + r_centre
            current_anchors[:, 1] = ideal_anchors[:, 1] + c_centre
            current_anchors[:, 2] = ideal_anchors[:, 2] + r_centre
            current_anchors[:, 3] = ideal_anchors[:, 3] + c_centre
            current_anchors[:, 4] = r # Gird row position
            current_anchors[:, 5] = c # Gird column position
            current_anchors[:, 6] = ideal_anchors[:, 4] # Anchor ID
            current_anchors[:, 7] = ideal_anchors[:, 5]  # anchor size
            current_anchors[:, 8] = ideal_anchors[:, 6]  # aspect ratio

            if total_anchors is None:
                total_anchors = current_anchors
            else:
                total_anchors = np.vstack((total_anchors, current_anchors))

    # Segregate with respect to anchor size
    total_anchors_dict = {}
    for itr_anc in total_anchors:
        anc_sz = itr_anc[7]
        if anc_sz not in total_anchors_dict:
            total_anchors_dict[anc_sz] = itr_anc
        else:
            total_anchors_dict[anc_sz] = np.vstack((total_anchors_dict[anc_sz], itr_anc))

    final_anchors = None
    for k in anchor_size:
        current_anchors = total_anchors_dict[k]
        if final_anchors is None:
            final_anchors = current_anchors
        else:
            final_anchors = np.vstack((final_anchors, current_anchors))
    # corrected_list = remove_boundary_outliers(total_anchors)
    return np.array(final_anchors)


def choose_size_and_aspect_ratio(roi):
    size_grids = [2, 4] # No. of grid present inside
    width = roi[2] - roi[0]
    height = roi[3] - roi[1]

    h = height / IMAGE_SHAPE[0]
    w = width / IMAGE_SHAPE[1]

    h_grid = h // STRIDE[0]
    w_grid = h // STRIDE[1]

    a_size = 0
    if max(h_grid, w_grid) < size_grids[0]:
        a_size = 0
    elif max(h_grid, w_grid) < size_grids[1]:
        a_size = 1

    asp_ratio = 0
    if h_grid > w_grid:
        asp_ratio = 1
    elif h_grid < w_grid:
        asp_ratio = 2

    return a_size, asp_ratio


def activate_grids(target, class_index, roi):
    rad = [2, 1]
    roi_grid = roi[0] // STRIDE[0], roi[1] // STRIDE[1], \
               roi[2] // STRIDE[0], roi[3] // STRIDE[1]
    height = roi[3] - roi[1]
    width = roi[2] - roi[0]

    # Confidence assignment
    confidence = target[:, :, class_index * channels_per_class]
    center_point = roi[1] + height // 2, roi[0] + width // 2

    center = center_point[0] // STRIDE[0], center_point[1] // STRIDE[1]
    x = confidence[roi_grid[1]:roi_grid[3], roi_grid[0]:roi_grid[2]]
    x = np.select([x < 0.25], [0.25])
    confidence[roi_grid[1]:roi_grid[3],roi_grid[0]:roi_grid[2]] = x
    for r in range(center[0] - rad[0], center[0] + rad[0]):
        for c in range(center[1] - rad[0], center[1] + rad[0]):
            if 0 < r < GRID_SHAPE[0] and 0 < c < GRID_SHAPE[1]:
                confidence[r, c] = 0.5

    for r in range(center[0] - rad[1], center[0] + rad[1]):
        for c in range(center[1] - rad[1], center[1] + rad[1]):
            if 0 < r < GRID_SHAPE[0] and 0 < c < GRID_SHAPE[1]:
                confidence[r, c] = 0.75

    confidence[center[0], center[1]] = 1.0

    # Bounding box calculation
    r = (center_point[0] - center[0] * STRIDE[0])/STRIDE[0]
    c = (center_point[1] - center[1] * STRIDE[1])/STRIDE[1]
    h = height / IMAGE_SHAPE[0]
    w = width / IMAGE_SHAPE[1]

    bbox = [r, c, h, w]
    target[:, :, class_index * channels_per_class] = confidence
    target[center[0], center[1], class_index * channels_per_class + 1] = r
    target[center[0], center[1], class_index * channels_per_class + 2] = c
    target[center[0], center[1], class_index * channels_per_class + 3] = h
    target[center[0], center[1], class_index * channels_per_class + 4] = w

    # print('center[0], center[1]: ', center[0], center[1])
    # print('original ROI:', roi)
    # print('bbox: ', bbox)
    return target


def assign_roi(batch_itr, class_index, roi, target_all):
    a_size, asp_ratio = choose_size_and_aspect_ratio(roi)

    asp_channel_index = [asp_ratio * channels_per_pyramid,
                         (asp_ratio+1) * channels_per_pyramid]

    target = target_all[batch_itr, a_size, :, :, asp_channel_index[0]:asp_channel_index[1]]
    target = activate_grids(target, class_index, roi)
    target_all[batch_itr, a_size, :, :, asp_channel_index[0]:asp_channel_index[1]] = target
    return target_all


def data_generator(f_batch_size):
    shapes = ShapeDataset()

    while True:
        image_batch = np.zeros([f_batch_size, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3], np.float32)

        target = np.zeros([f_batch_size, pyramid_size, GRID_SHAPE[0], GRID_SHAPE[1], channels_per_pyramid], np.float32)

        for itr in range(0, f_batch_size):

            img, rects = shapes.generate_image(IMAGE_SHAPE, 3)
            image_batch[itr, ::] = shapes.preprocessing_img(img)
            print("original image:", itr)
            for class_rect in rects:
                class_index = class_rect[0]
                c_rect = class_rect[1]
                print("original c_rect:", c_rect)
                target = assign_roi(itr, class_index, c_rect, target)

        """
        Target shape:
        pyramidal_shape X batch X GRID_SHAPE[0] X GRID_SHAPE[1] X (n_aspect_ratio x n_class x channels_per_class)
        3x1x28x28x45
        """
        yield image_batch, [target[:, 0,::], target[:, 1,::], target[:, 2,::]]


def prediction2bbox(img_data, pred_data, f_batch_size):
    for batch_itr in range(f_batch_size):
        image = img_data[batch_itr, ::] *255 + 128
        out_pos = []
        target = np.array(pred_data)

        for pyr_itr in range(pyramid_size): # iterate for pyrmidal size
            for asp_itr in range(num_aspect_ratio):
                for cls_itr in range(num_class):
                    channel_pos_index = asp_itr*channels_per_aspect_ratio + cls_itr * channels_per_class
                    conf = np.array(target[pyr_itr, batch_itr, :,:, channel_pos_index])
                    bbox = np.array(target
                                    [pyr_itr, batch_itr, :, :,channel_pos_index+1:channel_pos_index + channels_per_class])

                    # cord_pair_list = peak_local_max(conf, min_distance=1, threshold_rel=0.45, indices=True)
                    cord_pair_list = argrelextrema(conf, np.greater, axis=1)
                    for cord_pair in zip(cord_pair_list[0],cord_pair_list[1]):

                        bbox_list = bbox[cord_pair[0], cord_pair[1], :]

                        r = bbox_list[0]
                        c = bbox_list[1]
                        h = bbox_list[2]
                        w = bbox_list[3]

                        # Bounding box calculation
                        y_centre = (cord_pair[0] + r) * STRIDE[0]
                        x_centre = (cord_pair[1] + c) * STRIDE[1]
                        height = h * IMAGE_SHAPE[0]
                        width = w * IMAGE_SHAPE[1]

                        width_by2 = int(np.round(width / 2))
                        height_by2 = int(np.round(height / 2))

                        x1 = int(x_centre - width_by2)
                        y1 = int(y_centre - height_by2)
                        x2 = int(x_centre + width_by2)
                        y2 = int(y_centre + height_by2)

                        [x1, y1, x2, y2] = ShapeDataset.get_restricted_rect([x1, y1, x2, y2], IMAGE_SHAPE)
                        out_pos.append([x1, y1, x2, y2, cls_itr])

        print("image_index: ", batch_itr)
        for p in out_pos:
            print(p)
            p1 = (p[0], p[1])
            p2 = (p[2], p[3])
            cls_index = p[4]

            if cls_index == 0:
                color = [0, 0, 255]
            elif cls_index == 1:
                color = [255, 0, 0]
            elif cls_index == 2:
                color = [0, 255, 0]

            cv.rectangle(image, p1, p2, color, 1)
        cv.imwrite('./output/out_' + str(batch_itr).zfill(3) + '.png', image)


def test():
    # rpn_model = create_fpn_model()
    # rpn_model.load_weights("models/trained_model_7x7.hdf5")

    os.makedirs('./output', exist_ok=True)

    f_batch_size = 74
    for data in data_generator(f_batch_size):
        # out = rpn_model.predict(data[0])
        break

    prediction2bbox(data[0], data[1], f_batch_size)



if __name__ == '__main__':

    test()
    # test()
