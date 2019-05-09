import keras as k
import rpn

image_sz = [600, 800, 3]
total_anchors = 17100

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