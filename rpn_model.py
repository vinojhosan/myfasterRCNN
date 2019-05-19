import keras as k
import keras.layers as KL
import rpn
import tensorflow as tf
import numpy as np
import math

IMAGE_SHAPE = [400, 400, 3]
total_anchors = 17100

def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])

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

def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)

def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return k.models.Model([input_feature_map], outputs, name="rpn_model")

def train():
    # Inputs
    input_image = KL.Input(
        shape=[None, None, IMAGE_SHAPE[2]], name="input_image")
    input_image_meta = KL.Input(shape=[IMAGE_SHAPE],
                                name="input_image_meta")
    # RPN GT
    input_rpn_match = KL.Input(
        shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
    input_rpn_bbox = KL.Input(
        shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

    # Detection GT (class IDs, bounding boxes, and masks)
    # 1. GT Class IDs (zero padded)
    input_gt_class_ids = KL.Input(
        shape=[None], name="input_gt_class_ids", dtype=tf.int32)
    # 2. GT Boxes in pixels (zero padded)
    # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
    input_gt_boxes = KL.Input(
        shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
    # Normalize coordinates
    gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
        x, K.shape(input_image)[1:3]))(input_gt_boxes)
    # 3. GT Masks (zero padded)
    # [batch, height, width, MAX_GT_INSTANCES]

    input_gt_masks = KL.Input(shape=[IMAGE_SHAPE[0], IMAGE_SHAPE[1], None],
                              name="input_gt_masks", dtype=bool)

    # RPN Model
    rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                          len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)


rpn_boxes = generate_anchors([32, 64, 128], )


























exit(0)
image_layer = k.layers.Input(image_sz)

resnet = k.applications.ResNet50(include_top=False,
             weights='imagenet',input_tensor=image_layer)

activation_49 = resnet.get_layer('activation_49').output

conv_256d = k.layers.Conv2D(256, 3)(activation_49)
conv_256d = k.layers.Flatten()(conv_256d)
objectness_2k = k.layers.Dense(2*total_anchors, activation='sigmoid')(conv_256d)

bbox_reg_2k = k.layers.Dense(4*total_anchors, activation='sigmoid')(conv_256d)

objectness_2k = k.layers.Reshape(target_shape=[total_anchors, 2])(objectness_2k)
bbox_reg_2k = k.layers.Reshape(target_shape=[total_anchors, 4])(bbox_reg_2k)

rpn_model = k.models.Model(image_layer, [objectness_2k, bbox_reg_2k])
rpn_model.summary()


def train():

    adam = k.optimizers.Adam(lr=0.001, decay=0.9, beta_1=0.9)

    rpn_model.compile(optimizer=adam, loss=[ 'binary_crossentropy','mean_squared_error'])

    # Model saving and checkpoint callbacks
    rpn_model.save('models/empty_model.hdf5')
    filepath = "models/weights.best.hdf5"
    checkpoint = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                                 mode='min')
    tensorboard = k.callbacks.TensorBoard(log_dir="./logs", write_graph=True, write_images=True)
    callbacks_list = [checkpoint, tensorboard]

    rpn_model.fit_generator(rpn.rpn_generator(),
                             steps_per_epoch=1000,
                             epochs=50,
                             callbacks=callbacks_list)

    rpn_model.save('models/trained_model_with_loss.hdf5')


train()