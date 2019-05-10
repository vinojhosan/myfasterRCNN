import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import keras as k
import keras.layers as kl
import keras.backend as kb
import tensorflow as tf
import numpy as np
import cv2 as cv

import synthetic_dataset as dataset
IMAGE_SHAPE = [448, 448, 3]
GRID_SHAPE = [14, 14]
n_class = 3
anchor_size = [[32, 32], [64, 128], [64, 64], [192, 128], [256, 256]]
n_anchors = len(anchor_size)

output_channel = int(n_anchors * (5 + n_class))
anchor_set = 5 + n_class
shapes = dataset.ShapeDataset()
grid_ratio = IMAGE_SHAPE[0]/GRID_SHAPE[0]

def get_anchors(cx, cy, bbox):
    rects = []
    ious = []
    for anc in anchor_size:
        h_by_2 = anc[0]/2
        w_by_2 = anc[1]/2

        rect = [cx - w_by_2, cy - h_by_2,
                cx + w_by_2, cy + h_by_2] # xmin, ymin, xmax, ymax

        rect = shapes.get_restricted_rect(rect, IMAGE_SHAPE)
        rects.append(rect)

        ious.append(intersection_over_union(rect, bbox))

    return rects, ious

def intersection_over_union(boxA, boxB):

    # Area of Intersection rectangle of two boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Areas of individual boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # IOU calculation
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def preprocessing_img(img):
    img /= 255.0
    img -= 0.5
    img = np.expand_dims(img, axis=0)
    return img

def data_generator(f_batch_size):

    while True:
        image_batch = np.zeros([f_batch_size, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3], np.float32)
        target_batch = np.zeros([f_batch_size, GRID_SHAPE[0], GRID_SHAPE[1], output_channel], np.float32)

        for itr in range(0, f_batch_size):

            img, rects = shapes.generate_image(IMAGE_SHAPE, 5)
            image_batch[itr, ::] = preprocessing_img(img)
            for rect in rects:
                class_index = rect[0]
                c_rect = rect[1]

                width = c_rect[2] - c_rect[0]
                height = c_rect[3] - c_rect[1]

                x_centre = c_rect[0] + width / 2
                y_centre = c_rect[1] + height / 2

                anchor_rects, ious = get_anchors(x_centre, y_centre, c_rect)


                for i, iou in enumerate(ious):
                    if iou >= 0.5:
                        anc = anchor_rects[i]
                        grid_x = int(x_centre / grid_ratio)
                        grid_y = int(y_centre / grid_ratio)

                        tx = (x_centre - grid_x * grid_ratio) / grid_ratio
                        ty = (y_centre - grid_y * grid_ratio) / grid_ratio
                        tw = np.log(width/(anc[2] - anc[0]))
                        th = np.log(height/(anc[3] - anc[1]))

                        target_batch[itr, grid_y, grid_x, i * anchor_set + 0] = 1  # confidence
                        target_batch[itr, grid_y, grid_x, i * anchor_set + 1] = tx
                        target_batch[itr, grid_y, grid_x, i * anchor_set + 2] = ty
                        target_batch[itr, grid_y, grid_x, i * anchor_set + 3] = 1.0
                        target_batch[itr, grid_y, grid_x, i * anchor_set + 4] = 1.0

                        target_batch[itr, grid_y, grid_x, i * anchor_set + 5+class_index] = 1.0 # class prob

        yield image_batch, target_batch

def tiny_yolo_model():
    input_image = kl.Input(shape = (448, 448, 3), name='input_image')

    x = kl.Conv2D(16, 3, strides=(1, 1), padding='same')(input_image)
    x = kl.LeakyReLU(0.1)(x)
    x = kl.MaxPool2D(2, padding='same')(x)

    x = kl.Conv2D(32, 3, strides=(1, 1), padding='same')(x)
    x = kl.LeakyReLU(0.1)(x)
    x = kl.MaxPool2D(2, padding='same')(x)

    x = kl.Conv2D(64, 3, strides=(1, 1), padding='same')(x)
    x = kl.LeakyReLU(0.1)(x)
    x = kl.MaxPool2D(2, padding='same')(x)

    x = kl.Conv2D(128, 3, strides=(1, 1), padding='same')(x)
    x = kl.LeakyReLU(0.1)(x)
    x = kl.MaxPool2D(2, padding='same')(x)

    x = kl.Conv2D(256, 3, strides=(1, 1), padding='same')(x)
    x = kl.LeakyReLU(0.1)(x)
    x = kl.MaxPool2D(2, padding='same')(x)

    x = kl.Conv2D(512, 3, strides=(1, 1), padding='same')(x)
    x = kl.LeakyReLU(0.1)(x)
    x = kl.MaxPool2D(2, strides=1, padding='same')(x)

    x = kl.Conv2D(1024, 3, strides=(1, 1), padding='same')(x)
    x = kl.LeakyReLU(0.1)(x)
    x = kl.Conv2D(1024, 3, strides=(1, 1), padding='same')(x)
    x = kl.LeakyReLU(0.1)(x)
    out = kl.Conv2D(output_channel, 1, strides=(1, 1), padding='same', activation='sigmoid')(x)

    yolo_model = k.models.Model(input_image, out)

    return yolo_model


def yolo_loss_main(n_class = n_class, l_anchors=n_anchors, f_anchor_per_set=anchor_set):

    lambda_coord = 5
    lambda_noobj = 0.5

    def yolo_loss_per_set(y_true, y_pred):
        """Loss caculation of only one set
        """

        """Allocation of channels"""
        objectness_true = kl.Lambda(lambda x: x[:, :, :, 0:1])(y_true)
        bbox_true = kl.Lambda(lambda x: x[:, :, :, 1:5])(y_true)
        class_true = kl.Lambda(lambda x: x[:, :, :, 5:5 + n_class])(y_true)

        object_mask = kb.cast(kl.Lambda(lambda x: x > 0.1)(objectness_true), tf.float32)
        no_object_mask = kb.cast(kl.Lambda(lambda x: ~(x > 0.1))(objectness_true), tf.float32)

        objectness_pred = kl.Lambda(lambda x: x[:, :, :, 0:1])(y_pred)
        bbox_pred = kl.Lambda(lambda x: x[:, :, :, 1:5])(y_pred)
        class_pred = kl.Lambda(lambda x: x[:, :, :, 5:5 + n_class])(y_pred)

        '''**************Confidence loss***********************'''
        conf_loss = kb.sum(object_mask * kb.square(objectness_true - objectness_pred))
        no_conf_loss = kb.sum(no_object_mask * kb.square(objectness_true - objectness_pred))

        '''**************Coordination loss***********************'''
        # x and y loss
        xy_true = kl.Lambda(lambda x: x[:, :, :, 0:2])(bbox_true)
        xy_pred = kl.Lambda(lambda x: x[:, :, :, 0:2])(bbox_pred)
        wh_true = kl.Lambda(lambda x: x[:, :, :, 2:4])(bbox_true)
        wh_pred = kl.Lambda(lambda x: x[:, :, :, 2:4])(bbox_pred)
        object_mask_2_channel = kl.Concatenate()([object_mask, object_mask])

        xy_loss = kb.sum(object_mask_2_channel * kb.square(xy_true - xy_pred))
        wh_loss = kb.sum(object_mask_2_channel * kb.square(kb.sqrt(wh_true) - kb.sqrt(wh_pred)))

        """************Class wise loss************************"""
        objection_true_3_channel = kl.Concatenate()([object_mask, object_mask, object_mask])
        class_loss = kb.sum(objection_true_3_channel * kb.square(class_true - class_pred))

        """**************Total loss**************************"""
        total_loss = lambda_coord * (xy_loss + wh_loss) + conf_loss + class_loss + (lambda_noobj * no_conf_loss)

        return total_loss

    def yolo_loss(y_true, y_pred):
        full_total_loss = 0
        for i in range(l_anchors):
            y_true_per = kl.Lambda(lambda x: x[:, :, :, i*f_anchor_per_set:i*f_anchor_per_set+f_anchor_per_set])(y_true)
            y_pred_per = kl.Lambda(lambda x: x[:, :, :, i*f_anchor_per_set:i*f_anchor_per_set+f_anchor_per_set])(y_pred)
            l_loss = yolo_loss_per_set(y_true_per, y_pred_per)
            full_total_loss += l_loss
        return full_total_loss

    return yolo_loss #(y_true_total, y_pred_total)


def my_loss_test():
    y_true = np.random.rand(1, 14, 14, output_channel)
    y_pred = np.random.rand(1, 14, 14, output_channel)

    tf.initialize_all_variables()
    yt = tf.constant(y_true)
    yp = tf.constant(y_pred)

    print(tf.Session().run(yolo_loss_main(yt, yp)))
# my_loss_test()


# for d in data_generator(8):
#     print(d[1].shape)
#     cv.imwrite('test_img.png', d[0][0, ::]*255)
#     break


def identity_metric(y_true, y_pred):
    return kb.mean(y_true, y_pred)

def train():

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    yolo_model = tiny_yolo_model()
    yolo_model.summary()
    adam = k.optimizers.Adam(lr=0.001, beta_1=0.9)

    yolo_model.compile(adam, loss=yolo_loss_main(n_class, n_anchors, anchor_set), metrics=['acc', identity_metric])

    # Model saving and checkpoint callbacks
    yolo_model.save('models/empty_model_32x32.hdf5')
    filepath = "models/weights.best_32x32.hdf5"
    checkpoint = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                                 mode='min')
    tensorboard = k.callbacks.TensorBoard(log_dir="./logs", write_graph=True, write_images=True)
    callbacks_list = [checkpoint, tensorboard]

    batch_size = 8

    yolo_model.fit_generator(data_generator(batch_size),
                             steps_per_epoch=1000,
                             epochs=100,
                             callbacks=callbacks_list)

    yolo_model.save('models/trained_model_32x32.hdf5')

train()
