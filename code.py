import numpy as np
import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import h5py
import random
import warnings
import time

from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from numpy.random import permutation

np.random.seed(1997)
use_cache = 0
color_type_global = 3


def get_im(path, img_rows, img_cols, color_type=1):
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # resize
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized


def get_im_rotate(path, img_rows, img_cols, color_type=1):
    if color_type == 1:
        img = cv2.imread(path, 0)
    else:
        img = cv2.imread(path)
    # rotate
    rotate = random.randint(-10, 10)
    m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rotate, 1)
    img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))
    # resize
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized


def get_im_crop(path, img_rows, img_cols, color_type=1):
    if color_type == 1:
        img = cv2.imread(path, 0)
    else:
        img = cv2.imread(path)
    # crop
    cx = random.randint(0, 80)
    cy = random.randint(0, 60)
    img = img[cx:(cx + 560), cy:(cy + 420)]
    # resize
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized


def get_driver_data():
    dr = dict()
    path = os.path.join('..', 'input', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr


def load_train(img_rows, img_cols, color_type=1):
    X_train = []
    y_train = []
    driver_id = []
    start_time = time.time()

    driver_data = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('..', 'input', 'imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers


def load_test(img_rows, img_cols, sub_dir, color_type=1):
    print('Read test images')
    # test data put into multiple sub_dirs due to memory limit
    path = os.path.join('..', 'input', 'imgs', 'test', sub_dir, '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files) / 10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))
    return X_test, X_test_id


def cache_data(data, path):
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        print('Restore data from pickle...')
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def save_model(model, index, cross='', save_weights=False):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    if save_weights:
        weight_name = 'model_weights' + str(index) + cross + '.h5'
        model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def read_model(index, cross=''):
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


def read_and_normalize_and_shuffle_train_data(img_rows, img_cols, color_type=1):
    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, driver_id, unique_drivers = load_train(img_rows, img_cols, color_type)
        cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, driver_id, unique_drivers) = restore_data(cache_path)

    print('convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    print('reshape...')
    if color_type == 1:
        train_data = train_data.reshape(train_data.shape[0], color_type, img_rows, img_cols)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))
    print('convert to float...')
    # changed from 32 to 16 to fit memory
    train_data = train_data.astype('float16')
    mean_pixel = [103.939, 116.779, 123.68]
    print('subtract 0...')
    train_data[:, 0, :, :] -= mean_pixel[0]
    print('subtract 1...')
    train_data[:, 1, :, :] -= mean_pixel[1]
    print('subtract 2...')
    train_data[:, 2, :, :] -= mean_pixel[2]

    train_target = np_utils.to_categorical(train_target, 10)

    # BOF permutation
    # perm = permutation(len(train_target))
    # train_data = train_data[perm]
    # train_target = train_target[perm]
    # EOF permutation

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, driver_id, unique_drivers


def read_and_normalize_test_data(img_rows=224, img_cols=224, sub_dir='p00', color_type=1):
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, sub_dir, color_type)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], color_type, img_rows, img_cols)
    else:
        test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float16')
    mean_pixel = [103.939, 116.779, 123.68]
    test_data[:, 0, :, :] -= mean_pixel[0]
    test_data[:, 1, :, :] -= mean_pixel[1]
    test_data[:, 2, :, :] -= mean_pixel[2]

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data)
    target = np.array(target)
    index = np.array(index)
    return data, target, index


def vgg_std16_model(img_rows, img_cols, color_type=3):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type, img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    f = h5py.File('../input/vgg16_weights.h5')
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def run_cross_validation(nfolds=5, model_str=''):
    img_rows, img_cols = 224, 224
    # changed from 64 to fit in memory
    batch_size = 16
    random_state = 97
    nb_epoch = 25

    print('Reading training data...')
    train_data, train_target, driver_id, unique_drivers = \
        read_and_normalize_and_shuffle_train_data(img_rows, img_cols, color_type_global)

    # excl 'p081'
    # selected = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p035', 'p041', 'p042', 'p045', 'p047',
    #             'p049', 'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p075', 'p024', 'p026', 'p039', 'p072']
    # train_data, train_target, train_index = copy_selected_drivers(train_data, train_target, driver_id, selected)

    # BOF permutation
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    # EOF permutation

    print('Start training...')
    # BOF k-fold
    num_fold = 0
    kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
    for train_drivers, test_drivers in kf:
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        model = vgg_std16_model(img_rows, img_cols, color_type_global)
        unique_list_train = [unique_drivers[i] for i in train_drivers]
        X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
        unique_list_valid = [unique_drivers[i] for i in test_drivers]
        X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

        weight_name = 'model_weights' + str(num_fold) + model_str + '.h5'
        weight_path = os.path.join('cache', weight_name)
        # weights saved during fitting
        if not os.path.isfile(weight_path):
            callbacks = [
                EarlyStoppingByLossVal(monitor='val_loss', value=0.00001, verbose=1),
                EarlyStopping(monitor='val_loss', patience=5, verbose=0),
                ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, verbose=0),
            ]
            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                      shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
                      callbacks=callbacks)
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)

        # save architecture only
        save_model(model, num_fold, model_str, save_weights=False)
    # EOF k-fold

    print('Test in batch...')
    mod_indices = range(1, nfolds + 1)
    img_indices = ('p00', 'p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09')
    id_all = []
    pred_all = []

    for i in img_indices:
        print('Reading test data under {0}...'.format(i))
        test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, i, color_type_global)
        print('Predicting for {0}...'.format(i))
        y_pred = []
        for index in mod_indices:
            # Store test predictions
            model = read_model(index, model_str)
            test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
            y_pred.append(test_prediction)
        test_res = merge_several_folds_mean(y_pred, len(mod_indices))
        id_all = id_all + test_id
        pred_all = pred_all + test_res

    print('Combining submissions...')
    info_string = 'loss_' + model_str \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(nfolds) \
                  + '_ep_' + str(nb_epoch) + '_all_selected'

    create_submission(pred_all, id_all, info_string)


def test_model_and_submit(start=1, end=1, model_str=''):
    img_rows, img_cols = 224, 224
    batch_size = 64
    nb_epoch = 15

    print('Start testing...')
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)
    yfull_test = []

    for index in range(start, end + 1):
        # Store test predictions
        model = read_model(index, model_str)
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
        yfull_test.append(test_prediction)

    info_string = 'loss_' + model_str \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(end - start + 1) \
                  + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, end - start + 1)
    create_submission(test_res, test_id, info_string)


# nfolds, nb_epoch, split
run_cross_validation(10, '_vgg_16_no_split_f10')

# train once
# run_train_single(nb_epoch=15, split=0.1, model_str='vgg_16_split_010_v05_single_01')
