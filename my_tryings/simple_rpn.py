import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import keras as k
import keras.layers as kl
import tensorflow as tf
import cv2 as cv

np.random.seed(4)
from my_tryings.synthetic_dataset import ShapeDataset
def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]

# anchor_boxes = generate_anchors([32, 64, 128], [0.5, 1, 2], [20, 20], 1, 1)
# print(anchor_boxes)

IMAGE_SHAPE = [224, 224, 3]
GRID_SHAPE = [16, 16]
ANCHOR_SIZE = [32, 64, 128]
ANCHOR_ASPECT_RATIO = [1, 2]
STRIDE = [IMAGE_SHAPE[0] // GRID_SHAPE[0],
          IMAGE_SHAPE[1] // GRID_SHAPE[1]]

grid_ratio = IMAGE_SHAPE[0]/GRID_SHAPE[0]
output_channel = 6
iou_threshold = 0.5

def create_rpn_model():

    input_image = kl.Input(shape=IMAGE_SHAPE)

    feature_map = k.applications.ResNet50(include_top=False,
                                        weights='imagenet')(input_image)

    shared = kl.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=(1, 1),
                       name='rpn_conv_shared')(feature_map)

    """ *****************Confidence branch*********************"""
    x = kl.Conv2D(1, (1, 1), padding='valid', activation='linear')(shared)
    # rpn_class_logits = kl.Reshape([GRID_SHAPE[0] * GRID_SHAPE[1], 2])(x) # Reshape to [batch, anchors, 2]
    rpn_confidence = kl.Activation("softmax", name="rpn_confidence")(x)

    """ *****************BBox branch*********************"""
    rpn_bbox = kl.Conv2D(4, (1, 1), padding="valid", activation='sigmoid')(shared)
    # rpn_bbox = kl.Reshape([GRID_SHAPE[0] * GRID_SHAPE[1], 4], name='rpn_bbox_pred')(x) # Reshape to [batch, anchors, 4]

    model = k.models.Model(input_image, [rpn_confidence, rpn_bbox])
    model.summary()

    return model


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
                        anc_id, anc, asp]  # x1, y1, x2, y2, anc_id
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
        for c in range(0, grid_size[0]):

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
            current_anchors[:, 7] = ideal_anchors[:, 5]  # Anchor ID
            current_anchors[:, 8] = ideal_anchors[:, 6]  # Anchor ID

            if total_anchors is None:
                total_anchors = current_anchors
            else:
                total_anchors = np.vstack((total_anchors, current_anchors))

    # corrected_list = remove_boundary_outliers(total_anchors)
    return np.array(total_anchors)


def batch_iou(box, anchor_list, epsilon=1e-5):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        box:          (numpy array) one vector containing [x1,y1,x2,y2] coordinates
        anchor_list:  (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding
        boxes.
    """
    # COORDINATES OF THE INTERSECTION BOXES
    x1 = np.maximum(box[0], anchor_list[:, 0])
    y1 = np.maximum(box[1], anchor_list[:, 1])
    x2 = np.minimum(box[2], anchor_list[:, 2])
    y2 = np.minimum(box[3], anchor_list[:, 3])

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (anchor_list[:, 2] - anchor_list[:, 0]) * (anchor_list[:, 3] - anchor_list[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def data_generator(f_batch_size):
    shapes = ShapeDataset()
    anchors_per_gird = len(ANCHOR_SIZE)*len(ANCHOR_ASPECT_RATIO)
    anchor_list = create_anchors(IMAGE_SHAPE, GRID_SHAPE, ANCHOR_SIZE, ANCHOR_ASPECT_RATIO)


    while True:
        image_batch = np.zeros([f_batch_size, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3], np.float32)

        target_confidence = np.zeros([f_batch_size, GRID_SHAPE[0] * GRID_SHAPE[1] * anchors_per_gird, 1], np.float32)
        target_bbox = np.zeros([f_batch_size, GRID_SHAPE[0] * GRID_SHAPE[1] * anchors_per_gird, 4], np.float32)

        for itr in range(0, f_batch_size):

            img, rects = shapes.generate_image(IMAGE_SHAPE, 1)
            image_batch[itr, ::] = shapes.preprocessing_img(img)

            for class_rect in rects:
                class_index = class_rect[0]
                c_rect = class_rect[1]
                # print('original:', c_rect)
                iou = batch_iou(c_rect, anchor_list)

                print(len(iou[iou >= iou_threshold]))
                # print('above:', iou[iou >= 0.1])
                # print(anchor_list[iou >= 0.1])
                target_confidence[itr, iou >= iou_threshold, 0] = 1
                # target_confidence[itr, iou >= 0.7, 0] = 1

                target_bbox[itr, iou >= iou_threshold, :] = anchor_list[iou >= iou_threshold, 0:4]

                width = c_rect[2] - c_rect[0]
                height = c_rect[3] - c_rect[1]

                x_centre = c_rect[0] + width / 2
                y_centre = c_rect[1] + height / 2
                grid_x = int(x_centre / STRIDE[1])
                grid_y = int(y_centre / STRIDE[0])

                tx = (x_centre - grid_x * STRIDE[1]) / STRIDE[1]
                ty = (y_centre - grid_y * STRIDE[0]) / STRIDE[0]
                th = height / STRIDE[0]
                tw = width / STRIDE[1]

                target_bbox[itr, iou >= iou_threshold, :] = [tx, ty, tw, th]

        yield image_batch, [target_confidence, target_bbox]


def rpn_loss(y_true, y_pred):

    """Allocation of channels"""
    true_mask = kl.Lambda(lambda x: x[:, :, :, 0:1])(y_true)

    true_confidence = kl.Lambda(lambda x: x[:, :, :, 0:2])(y_true)
    bbox_true = kl.Lambda(lambda x: x[:, :, :, 2:6])(y_true)

    pred_confidence = kl.Lambda(lambda x: x[:, :, :, 0:2])(y_pred)
    bbox_pred = kl.Lambda(lambda x: x[:, :, :, 2:6])(y_pred)

    confidence_loss = K.mean(categorical_crossentropy(true_confidence, pred_confidence))

def train():

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    rpn_model = create_rpn_model()
    adam = k.optimizers.Adam(lr=0.0001)

    rpn_model.compile(adam, loss=[k.losses.binary_crossentropy, k.losses.mean_squared_error], metrics=['acc', 'mse'])

    # Model saving and checkpoint callbacks
    rpn_model.save('models/empty_model_7x7.hdf5')
    filepath = "models/weights.best_7x7.hdf5"
    checkpoint = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                                 mode='min')
    tensorboard = k.callbacks.TensorBoard(log_dir="./logs", write_graph=True, write_images=True)
    callbacks_list = [checkpoint, tensorboard]

    batch_size = 8

    rpn_model.fit_generator(data_generator(batch_size),
                             steps_per_epoch=int(1000/batch_size),
                             epochs=100,
                             callbacks=callbacks_list)

    rpn_model.save('models/trained_model_7x7.hdf5')

def test():
    # rpn_model = k.models.load_model("models/trained_model_7x7.hdf5")
    anchor_list = create_anchors(IMAGE_SHAPE, GRID_SHAPE)
    f_batch_size = 1
    for data in data_generator(f_batch_size):
        # out = rpn_model.predict(data[0])
        break

    for itr in range(f_batch_size):
        image = data[0][itr, ::] *255 + 128
        cv.imwrite('input.png', image)

        out = data[1]
        conf = np.array(out[0][itr])
        bbox = np.array(out[1][itr])

        out_pos = []
        for n in range(0, conf.shape[0]):
            if conf[n, 0] > iou_threshold:
                rect = bbox[n]
                tx = rect[0]
                ty = rect[1]
                tw = rect[2]
                th = rect[3]

                print('tx ty tw th:', tx, ty, tw, th)

                respective_anchor = anchor_list[n]
                grid_x = respective_anchor[4] # Grid row
                grid_y = respective_anchor[5] # Grid col

                x_centre = int(np.round((tx + grid_x) * STRIDE[1]))
                y_centre = int(np.round((ty + grid_y) * STRIDE[0]))
                width_by2 = int(np.round(tw * STRIDE[1]/2))
                height_by2 = int(np.round(th * STRIDE[0]/2))

                x1 = x_centre - width_by2
                y1 = y_centre - height_by2
                x2 = x_centre + width_by2
                y2 = y_centre + height_by2

                [x1, y1, x2, y2] = ShapeDataset.get_restricted_rect([x1, y1, x2, y2], IMAGE_SHAPE)

                out_pos.append([x1, y1, x2, y2])

        for p in out_pos:
            print(p)
            p1 = (p[0], p[1])
            p2 = (p[2], p[3])
            cv.rectangle(image, p1, p2, [0, 255, 0], 1)

        cv.imwrite(str(itr) + 'out.png', image)


if __name__ == '__main__':
    # create_rpn_model()
    test()
