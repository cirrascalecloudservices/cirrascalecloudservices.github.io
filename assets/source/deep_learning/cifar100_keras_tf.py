from __future__ import print_function
from mxnet import recordio
import os
import argparse
import logging
import pickle
import cv2
import numpy as np
import h5py
import numpy as np
import _pickle as cPickle
import keras
import mxnet as mx
import tensorflow as tf
from skimage.io import imsave
from optparse import OptionParser
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.callbacks import TensorBoard

logging.basicConfig(level=logging.INFO)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

#https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure/48393723
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument("--directory", dest="dir", help="cifar10 directory", default="/mnt/datasets/cifar100")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate. default is 0.0001.')
    parser.add_argument('--wd', type=float, default=1e-6,
                        help='weight decay rate. default is 1e-6.')
    opt = parser.parse_args()

    metadata=  unpickle( os.path.join(opt.dir, "cifar-100-python", "meta"))
    labels = {}
    labels['coarse_label_names'] = metadata['coarse_label_names']
    labels['fine_label_names'] = metadata['fine_label_names']
    
    train_idx = os.path.join(opt.dir, "cifar_mxnet_train.idx")
    train_rec = os.path.join(opt.dir, "cifar_mxnet_train.rec")
    test_idx =  os.path.join(opt.dir, "cifar_mxnet_test.idx")
    test_rec =  os.path.join(opt.dir, "cifar_mxnet_test.rec")

    num_classes = 100
    batch_size = opt.batch_size

    #'channels_first' or 'channels_last'
    keras.backend.set_image_data_format('channels_last')

    x_train = np.zeros((50000,32,32,3))
    y_train = np.zeros((50000,1))
    train_imgrec = recordio.MXIndexedRecordIO(train_idx, train_rec, 'r')
    for i in range(50000-1):
        header, s = recordio.unpack(train_imgrec.read_idx(i+1))
        img_flat = mx.image.imdecode(s).asnumpy()
        #img_flat = img_flat.transpose(2,0,1)
        x_train[i] = img_flat
        #coarse is 0, fine is 1
        y_train[i] = header.label[1]

    x_test = np.zeros((10000,32,32,3))
    y_test = np.zeros((10000,1))
    test_imgrec = recordio.MXIndexedRecordIO(test_idx, test_rec, 'r')
    for i in range(10000-1):
        header, s = recordio.unpack(test_imgrec.read_idx(i+1))
        img_flat = mx.image.imdecode(s).asnumpy()
        #img_flat = img_flat.transpose(2,0,1)
        x_test[i] = img_flat
        #coarse is 0, fine is 1
        y_test[i] = header.label[1]

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    model = Sequential()
    model.add(Conv2D(128, (3, 3), padding='same',
                    input_shape=x_train.shape[1:]))
    model.add(Activation('elu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('elu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('elu'))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    optimizer = keras.optimizers.rmsprop(lr=opt.lr, decay=opt.wd)

    tbcallback = TrainValTensorBoard(log_dir='./logs',
                                     histogram_freq=0,
                                     write_graph=False,
                                     write_images=False,
                                     write_grads=True,
                                     batch_size=batch_size,
                                     update_freq='epoch')

    # train using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=opt.num_epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[tbcallback])

    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Evaluate model with test data set and share sample prediction results
    evaluation = model.evaluate_generator(datagen.flow(x_test, y_test,
                                        batch_size=batch_size),
                                        steps=x_test.shape[0] // batch_size)

    print('Model Accuracy = %.2f' % (evaluation[1]))

    predict_gen = model.predict_generator(datagen.flow(x_test, y_test,
                                      batch_size=batch_size),
                                      steps=x_test.shape[0] // batch_size)

    num_predictions = 20
    for predict_index, predicted_y in enumerate(predict_gen):
        actual_label = labels['fine_label_names'][np.argmax(y_test[predict_index])]
        predicted_label = labels['fine_label_names'][np.argmax(predicted_y)]
        print('Actual Label = %s vs. Predicted Label = %s' % (actual_label,
                                                            predicted_label))
        if predict_index == num_predictions:
            break