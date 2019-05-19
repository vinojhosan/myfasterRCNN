import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import keras as k
import keras.layers as kl
import keras.backend as kb
import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import synthetic_dataset as dataset
IMAGE_SHAPE = [224, 224, 3]
GRID_SHAPE = [7, 7]
n_class = 3
anchor_size = [[32, 32], [64, 64], [128, 192], [192, 128], [128, 128]]
n_anchors = 1#len(anchor_size)

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

                # anchor_rects, ious = get_anchors(x_centre, y_centre, c_rect)
                #
                #
                # for i, iou in enumerate(ious):
                #     if iou >= 0.5:
                # anc = anchor_rects[i]
                i = 0
                grid_x = int(x_centre / grid_ratio)
                grid_y = int(y_centre / grid_ratio)

                tx = (x_centre - grid_x * grid_ratio) / grid_ratio
                ty = (y_centre - grid_y * grid_ratio) / grid_ratio
                tw = np.log(width/IMAGE_SHAPE[0])
                th = np.log(height/IMAGE_SHAPE[1])

                target_batch[itr, grid_y, grid_x, i * anchor_set + 0] = 1  # confidence
                target_batch[itr, grid_y, grid_x, i * anchor_set + 1] = tx
                target_batch[itr, grid_y, grid_x, i * anchor_set + 2] = ty
                target_batch[itr, grid_y, grid_x, i * anchor_set + 3] = tw
                target_batch[itr, grid_y, grid_x, i * anchor_set + 4] = th

                target_batch[itr, grid_y, grid_x, i * anchor_set + 5+class_index] = 1.0 # class prob

        yield image_batch, target_batch

def row_data_generator(f_batch_size):

    while True:
        image_batch = np.zeros([f_batch_size, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3], np.float32)
        target_batch = np.zeros([f_batch_size, GRID_SHAPE[0], GRID_SHAPE[1], output_channel], np.float32)
        target_batch_out = []

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

                i = 0
                grid_x = int(x_centre / grid_ratio)
                grid_y = int(y_centre / grid_ratio)

                tx = (x_centre - grid_x * grid_ratio) / grid_ratio
                ty = (y_centre - grid_y * grid_ratio) / grid_ratio
                tw = width / IMAGE_SHAPE[0]
                th = height / IMAGE_SHAPE[1]

                target_batch[itr, grid_y, grid_x, i * anchor_set + 0] = 1  # confidence
                target_batch[itr, grid_y, grid_x, i * anchor_set + 1] = tx
                target_batch[itr, grid_y, grid_x, i * anchor_set + 2] = ty
                target_batch[itr, grid_y, grid_x, i * anchor_set + 3] = tw
                target_batch[itr, grid_y, grid_x, i * anchor_set + 4] = th

                target_batch[itr, grid_y, grid_x, i * anchor_set + 5+class_index] = 1.0 # class prob
            target_batch_out.append(np.reshape(target_batch[itr]))


        yield image_batch, np.array(target_batch_out)

def tiny_yolo_model():
    input_image = kl.Input(shape = (IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3), name='input_image')

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
    out = kl.Conv2D(output_channel, 1, strides=(1, 1), padding='same', activation='linear')(x)

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
        total_loss = lambda_coord * (xy_loss + wh_loss) + conf_loss # + class_loss + (lambda_noobj * no_conf_loss)

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


for d in row_data_generator(8):
    print(d[1].shape)
    cv.imwrite('test_img.png', d[0][0, ::]*255)
    break

exit(0)


def identity_metric(y_true, y_pred):
    return kb.mean(y_true, y_pred)

def train():

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    yolo_model = tiny_yolo_model()
    yolo_model.summary()
    adam = k.optimizers.Adam(lr=0.0001)

    yolo_model.compile(adam, loss=yolo_loss_main(n_class, n_anchors, anchor_set), metrics=['acc', 'mse'])

    # Model saving and checkpoint callbacks
    yolo_model.save('models/empty_model_7x7.hdf5')
    filepath = "models/weights.best_7x7.hdf5"
    checkpoint = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                                 mode='min')
    tensorboard = k.callbacks.TensorBoard(log_dir="./logs", write_graph=True, write_images=True)
    callbacks_list = [checkpoint, tensorboard]

    batch_size = 8

    yolo_model.fit_generator(data_generator(batch_size),
                             steps_per_epoch=int(1000/batch_size),
                             epochs=100,
                             callbacks=callbacks_list)

    yolo_model.save('models/trained_model_7x7.hdf5')


def test():
    # yolo_model = k.models.load_model("models/weights.best_7x7.hdf5", custom_objects={'yolo_loss':yolo_loss_main(n_class, n_anchors, anchor_set)})


    for data in data_generator(1):
        # out = yolo_model.predict(data[0])
        break

    image = data[0][0, ::] *255 + 128
    cv.imwrite('input.png', image)
    out = np.array(data[1])

    for ch in range(0, out.shape[3]):
        out_pos = []
        anc_index = 0
        if ch%anchor_set == 0:
            objectness = out[0][:,:, ch]

            obj_pair = np.nonzero(objectness > 0.5)
            for xy_itr in range(len(obj_pair[0])):
                tx = out[0][obj_pair[0][xy_itr], obj_pair[1][xy_itr], ch + 1]
                ty = out[0][obj_pair[0][xy_itr], obj_pair[1][xy_itr], ch + 2]
                tw = out[0][obj_pair[0][xy_itr], obj_pair[1][xy_itr], ch + 3]
                th = out[0][obj_pair[0][xy_itr], obj_pair[1][xy_itr], ch + 4]

                x = int(np.round((obj_pair[1][xy_itr]) * grid_ratio + tx*grid_ratio))
                y = int(np.round((obj_pair[0][xy_itr]) * grid_ratio + ty * grid_ratio))

                # ty = (y_centre - grid_y * grid_ratio) / grid_ratio
                # anc = anchor_size[anc_index]

                w = int(np.round(np.exp(tw) * IMAGE_SHAPE[0]))
                h = int(np.round(np.exp(th) * IMAGE_SHAPE[1]))
                # th = np.log(height / (anc[3] - anc[1]))

                anc_index += 1
                print("x:", x, "y:", y, "w:", w, "h:", h)
                out_pos.append((x, y, w, h))

        for p in out_pos:
            print(p)
            cv.circle(image, (p[0], p[1]), 3, [0, 255, 0], -1)
            p1 = (int(p[0] - p[2] / 2), int(p[1] - p[3] / 2))
            p2 = (int(p[0] + p[2] / 2), int(p[1] + p[3] / 2))
            cv.rectangle(image, p1, p2, [0, 255, 0], 1)


    cv.imwrite('out.png', image)
# test()
train()
