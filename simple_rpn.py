import os
import numpy as np
import keras as k
import keras.layers as kl
import tensorflow as tf

from synthetic_dataset import ShapeDataset
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

IMAGE_SHAPE = [228, 228, 3]
GRID_SHAPE = [7, 7]
grid_ratio = IMAGE_SHAPE[0]/GRID_SHAPE[0]
output_channel = 6

def create_rpn_model():

    input_image = kl.Input(shape=IMAGE_SHAPE)

    feature_map = k.applications.ResNet50(include_top=False,
                                        weights='imagenet')(input_image)

    shared = kl.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=(1, 1),
                       name='rpn_conv_shared')(feature_map)

    """ *****************Confidence branch*********************"""
    x = kl.Conv2D(2, (1, 1), padding='valid', activation='linear')(shared)
    # Reshape to [batch, anchors, 2]
    rpn_class_logits = kl.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)
    rpn_confidence = kl.Activation("softmax", name="rpn_confidence")(rpn_class_logits)

    """ *****************BBox branch*********************"""
    x = kl.Conv2D(4, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred')(shared)
    # Reshape to [batch, anchors, 4]
    rpn_bbox = kl.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)


    model = k.models.Model(input_image, [rpn_confidence, rpn_bbox])
    model.summary()

    return model

def data_generator(f_batch_size):
    shapes = ShapeDataset()
    while True:
        image_batch = np.zeros([f_batch_size, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3], np.float32)
        target_batch = np.zeros([f_batch_size, GRID_SHAPE[0], GRID_SHAPE[1], output_channel], np.float32)

        target_confidence = np.zeros([f_batch_size, GRID_SHAPE[0] * GRID_SHAPE[1], 2], np.float32)
        target_bbox = np.zeros([f_batch_size, GRID_SHAPE[0] * GRID_SHAPE[1], 4], np.float32)

        target_batch[:, :, :, 1] = 1

        for itr in range(0, f_batch_size):

            img, rects = shapes.generate_image(IMAGE_SHAPE, 5)
            image_batch[itr, ::] = shapes.preprocessing_img(img)
            for rect in rects:
                class_index = rect[0]
                c_rect = rect[1]

                width = c_rect[2] - c_rect[0]
                height = c_rect[3] - c_rect[1]

                x_centre = c_rect[0] + width / 2
                y_centre = c_rect[1] + height / 2

                grid_x = int(x_centre / grid_ratio)
                grid_y = int(y_centre / grid_ratio)

                tx = (x_centre - grid_x * grid_ratio) / grid_ratio
                ty = (y_centre - grid_y * grid_ratio) / grid_ratio
                tw = width/IMAGE_SHAPE[0]
                th = height/IMAGE_SHAPE[1]


                target_batch[itr, grid_y, grid_x, 0] = 1  # confidence
                target_batch[itr, grid_y, grid_x, 1] = 0
                target_batch[itr, grid_y, grid_x, 2] = tx
                target_batch[itr, grid_y, grid_x, 3] = ty
                target_batch[itr, grid_y, grid_x, 4] = tw
                target_batch[itr, grid_y, grid_x, 5] = th

            target_confidence[itr, ::] = np.reshape(target_batch[itr, :, :, 0:2], [GRID_SHAPE[0] * GRID_SHAPE[1], 2])
            target_bbox[itr, ::] = np.reshape(target_batch[itr, :, :, 2:6], [GRID_SHAPE[0] * GRID_SHAPE[1], 4])

        yield image_batch, [target_confidence, target_bbox]

# for d in data_generator(1):
#     print(d[0].shape)
#     print(d[1][0].shape)
#     print(d[1][1].shape)
#     print(d[1][1])
#     break


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


train()