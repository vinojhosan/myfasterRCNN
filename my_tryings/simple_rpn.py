import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
from tensorflow import keras as k
from tensorflow.keras import layers as kl
import tensorflow as tf
import cv2 as cv

np.random.seed(4)
from synthetic_dataset import ShapeDataset

IMAGE_SHAPE = [224, 224, 3]
GRID_SHAPE = [28, 28]
ANCHOR_SIZE = [32, 64, 128]
ANCHOR_ASPECT_RATIO = [0.5, 1, 2]
STRIDE = [IMAGE_SHAPE[0] // GRID_SHAPE[0],
          IMAGE_SHAPE[1] // GRID_SHAPE[1]]

grid_ratio = IMAGE_SHAPE[0]/GRID_SHAPE[0]
iou_threshold = 0.5
feature_size = 256


def classification_model():

    model_input = kl.Input(shape=[GRID_SHAPE[0], GRID_SHAPE[1], feature_size])

    x = model_input
    for i in range(4):
        x = kl.Conv2D(feature_size, 3,
                          padding='same',
                          kernel_initializer='he_normal')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)

    x = kl.Conv2D(1 * len(ANCHOR_ASPECT_RATIO), (1, 1), padding='valid')(x)
    rpn_class_logits = kl.Reshape([-1, 1])(x)  # Reshape to [batch, anchors, 1]
    rpn_confidence = kl.Activation("sigmoid")(rpn_class_logits)

    cls_model = k.models.Model(model_input, rpn_confidence, name='cls_model')
    return cls_model


def bbox_model():

    model_input = kl.Input(shape=[GRID_SHAPE[0], GRID_SHAPE[1], feature_size])

    x = model_input
    for i in range(4):
        x = kl.Conv2D(feature_size, 3,
                          padding='same',
                          kernel_initializer='he_normal')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)

    rpn_bbox = kl.Conv2D(4 * len(ANCHOR_ASPECT_RATIO), (1, 1), padding="valid")(x)
    rpn_bbox = kl.Reshape([-1, 4], name='rpn_bbox_pred')(rpn_bbox)

    bbox_model = k.models.Model(model_input, rpn_bbox, name='bbox_model')
    return bbox_model


def simple_conv_upsample(feature, x):
    feature = k.layers.UpSampling2D()(feature)
    x = k.layers.Concatenate()([feature, x])
    x = kl.Conv2D(feature_size, 3,
                  padding='same',
                  kernel_initializer='he_normal')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)

    return x


def create_rpn_model():

    input_image = kl.Input(shape=IMAGE_SHAPE)

    feature_map = k.applications.ResNet50(include_top=False,
                                        weights='imagenet')(input_image)

    shared = kl.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=(1, 1),
                       name='rpn_conv_shared')(feature_map)

    """ *****************Confidence branch*********************"""
    x = kl.Conv2D(1 * anchors_per_gird, (1, 1), padding='valid')(shared)
    rpn_class_logits = kl.Reshape([-1, 1])(x) # Reshape to [batch, anchors, 1]
    rpn_confidence = kl.Activation("sigmoid", name="rpn_confidence")(rpn_class_logits)

    """ *****************BBox branch*********************"""
    rpn_bbox = kl.Conv2D(4 * anchors_per_gird, (1, 1), padding="valid")(shared)
    rpn_bbox = kl.Reshape([-1, 4], name='rpn_bbox_pred')(rpn_bbox) # Reshape to [batch, anchors, 4]

    # outputs = kl.Concatenate()([rpn_confidence, rpn_bbox])
    model = k.models.Model(input_image, [rpn_confidence, rpn_bbox])
    model.summary()

    return model


def create_fpn_model():

    input_image = kl.Input(shape=IMAGE_SHAPE)

    resnet50 = k.applications.ResNet50(include_top=False,
                                        weights='imagenet',
                                       input_tensor=input_image,
                                       input_shape=IMAGE_SHAPE)

    pyramidal_layer1 = resnet50.get_layer('activation_48').output # 7 x 7
    pyramidal_layer2 = resnet50.get_layer('activation_39').output  # 14 x 14
    pyramidal_layer3 = resnet50.get_layer('activation_21').output  # 28 x 28

    feature0 = kl.Conv2D(feature_size, 3,padding='same')(pyramidal_layer1)
    feature1 = simple_conv_upsample(pyramidal_layer1, pyramidal_layer2)
    feature2 = simple_conv_upsample(pyramidal_layer2, pyramidal_layer3)

    cls_model = classification_model()
    box_model = bbox_model()

    feature0 = k.layers.UpSampling2D((4, 4))(feature0)
    feature1 = k.layers.UpSampling2D((2, 2))(feature1)
    feature2 = feature2

    cls0 = cls_model(feature0)
    cls1 = cls_model(feature1)
    cls2 = cls_model(feature2)

    box0 = box_model(feature0)
    box1 = box_model(feature1)
    box2 = box_model(feature2)

    rpn_confidence = k.layers.Concatenate(axis=1)([cls0, cls1, cls2])
    rpn_bbox = k.layers.Concatenate(axis=1)([box0, box1, box2])

    output = k.layers.Concatenate()([rpn_confidence, rpn_bbox])

    model = k.models.Model(input_image, output)
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
            current_anchors[:, 7] = ideal_anchors[:, 5]  # anchor size
            current_anchors[:, 8] = ideal_anchors[:, 6]  # aspect ratio

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


anchor_list = create_anchors(IMAGE_SHAPE, GRID_SHAPE, ANCHOR_SIZE, ANCHOR_ASPECT_RATIO)
anchors_per_gird = len(ANCHOR_SIZE)*len(ANCHOR_ASPECT_RATIO)


def data_generator(f_batch_size):
    shapes = ShapeDataset()

    while True:
        image_batch = np.zeros([f_batch_size, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3], np.float32)

        target_confidence = np.zeros([f_batch_size, GRID_SHAPE[0] * GRID_SHAPE[1] * anchors_per_gird, 1], np.float32)
        target_bbox = np.zeros([f_batch_size, GRID_SHAPE[0] * GRID_SHAPE[1] * anchors_per_gird, 4], np.float32)

        for itr in range(0, f_batch_size):

            img, rects = shapes.generate_image(IMAGE_SHAPE, 5)
            image_batch[itr, ::] = shapes.preprocessing_img(img)

            for class_rect in rects:
                class_index = class_rect[0]
                c_rect = class_rect[1]
                # print('original:', c_rect)
                iou = batch_iou(c_rect, anchor_list)

                # print(len(iou[iou >= iou_threshold]))
                target_confidence[itr, iou >= iou_threshold, 0] = 1
                # target_confidence[itr, iou >= 0.7, 0] = 1

                target_bbox[itr, iou >= iou_threshold, :] = anchor_list[iou >= iou_threshold, 0:4]

                width = c_rect[2] - c_rect[0]
                height = c_rect[3] - c_rect[1]

                x_centre = c_rect[0] + width / 2
                y_centre = c_rect[1] + height / 2

                chosen_anchors = np.where(iou >= iou_threshold)

                for itr_anc in chosen_anchors[0]:
                    anc = anchor_list[itr_anc]

                    anc_h = anc[2] - anc[0]
                    anc_w = anc[3] - anc[1]

                    anc_cx = anc[1]
                    anc_cy = anc[0]

                    tx = (x_centre - anc_cx) / anc_w
                    ty = (y_centre - anc_cy) / anc_h
                    th = np.log(height/anc_h)
                    tw = np.log(width/anc_w)

                    target_bbox[itr, itr_anc, :] = [tx, ty, tw, th]

        target_out = np.concatenate([target_confidence, target_bbox], axis=-1)

        yield image_batch, target_out


def rpn_loss(y_true, y_pred):
    target_confidence = y_true[:,:,0:1]
    target_bbox = y_true[:,:,1:]

    pred_confidence = y_pred[:, :, 0:1]
    pred_bbox = y_pred[:, :, 1:]

    # target_confidence, target_bbox = y_true[0], y_true[1]
    # pred_confidence, pred_bbox = y_pred[0], y_pred[0]

    confidence_loss = tf.reduce_sum(k.losses.binary_crossentropy(target_confidence, pred_confidence))

    masked_target_conf = kl.Concatenate(name='masked_target_conf')([target_confidence,
                                    target_confidence,
                                    target_confidence,
                                    target_confidence])
    pred_bbox = tf.multiply(pred_bbox, target_confidence, name='pred_bbox_multiply')

    bbox_loss = tf.reduce_sum(k.losses.mean_squared_error(target_bbox, pred_bbox))

    return 4 * confidence_loss + 1 * bbox_loss



def train():

    os.makedirs('./models', exist_ok=True)

    rpn_model = create_fpn_model()
    adam = k.optimizers.Adam(lr=0.001)

    rpn_model.compile(adam, loss=rpn_loss) # , metrics=['mse']

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
    rpn_model = k.models.load_model("models/trained_model_7x7.hdf5")
    f_batch_size = 32
    for data in data_generator(f_batch_size):
        out = rpn_model.predict(data[0])
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

                anc = anchor_list[n]

                anc_h = anc[2] - anc[0]
                anc_w = anc[3] - anc[1]

                anc_cx = anc[1]
                anc_cy = anc[0]

                respective_anchor = anchor_list[n]

                x_centre = int(np.round(tx * anc_w + anc_cx))
                y_centre = int(np.round(ty * anc_h + anc_cy))
                width = np.exp(tw) * anc_w
                height = np.exp(th) * anc_h

                width_by2 = int(np.round(width/2))
                height_by2 = int(np.round(height/2))

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
    # create_fpn_model()
    # test()
    train()