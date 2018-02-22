#!/usr/bin/env python3
from collections import namedtuple as namedtuple
import json
import logging
import os
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop


Dataset = namedtuple('Dataset', ['x', 'y'])
Data = namedtuple('Data', ['train', 'test'])

NUM_CLASSES = 10
RANDOM_PERMUTES = 3
HIDDEN_UNITS = [128, 256]
BATCH_SIZES = [16, 32]
LRS = [1e-5, 1e-6]

RANDOM_PERMUTES = 3
HIDDEN_UNITS = [128]
BATCH_SIZES = [16]
LRS = [3e-5, 1e-6]

DATAS_FILE = 'datas.pickle'
MODELS_DIR = 'models'
EPOCH_FILE = 'epoch'

np.random.seed(0)

def get_original_data():
    """Load mnist data."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    logging.info('Loaded train with %d samples', len(x_train))
    logging.info('Loaded test with %d samples', len(x_test))

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return Data(
        train=Dataset(x_train, y_train),
        test=Dataset(x_test, y_test),
    )

def transpose(data):
    """Return the same data, but reverse axes in x."""
    return Data(
        train=Dataset(np.transpose(data.train.x, axes=(0, 2, 1)), data.train.y),
        test=Dataset(np.transpose(data.test.x, axes=(0, 2, 1)), data.test.y),
    )

def flatten(data):
    """Make 1D-version of data."""
    n = len(data.train.x)
    m = len(data.test.x)
    return Data(
        train=Dataset(np.reshape(data.train.x, (n, -1, 1)), data.train.y),
        test=Dataset(np.reshape(data.test.x, (m, -1, 1)), data.test.y),
    )

def permute(data):
    """Permute columns in data."""
    p = np.random.permutation(data.train.x.shape[1])
    return Data(
        train=Dataset(data.train.x[:, p, :], data.train.y),
        test=Dataset(data.test.x[:, p, :], data.test.y),
    )

def make_datas():
    datas = {}
    datas['original'] = get_original_data()
    datas['transposed'] = transpose(datas['original'])
    datas['flattenned'] = flatten(datas['original'])
    datas['transposed-flattenned'] = flatten(datas['transposed'])

    base_datas = ['original', 'transposed', 'flattenned']
    for key in base_datas:
        data = datas[key]
        for i in range(RANDOM_PERMUTES):
            datas[key + ' random permute #{}'.format(i)] = permute(data)

    logging.info('Prepared %d datasets', len(datas))
    return datas

def load_or_make_datas():
    try:
        with open(DATAS_FILE, 'rb') as fin:
            return pickle.load(fin)
    except pickle.UnpicklingError:
        logging.info('Can\'t unpickle datasets. Making new instead')
    except OSError:
        logging.info('Can\'t find file with datasets. Making new instead')
    result = make_datas()
    with open(DATAS_FILE, 'wb') as fout:
        pickle.dump(result, fout)
    logging.info('Loaded datasets from %s', DATAS_FILE)
    return result

def init_model(hidden_units, inputs, lr):
    model = Sequential()
    model.add(
        SimpleRNN(
            hidden_units,
            kernel_initializer=initializers.RandomNormal(stddev=0.001),
            recurrent_initializer=initializers.Identity(gain=1.0),
            activation='relu',
            input_shape=inputs
        )
    )
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    rmsprop = RMSprop(lr=lr)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=rmsprop,
        metrics=['accuracy']
    )
    return model

def reinit_model(hidden_units, inputs, lr, reference):
    model = Sequential()
    model.add(
        SimpleRNN(
            hidden_units,
            kernel_initializer=initializers.RandomNormal(stddev=0.001),
            recurrent_initializer=initializers.Identity(gain=1.0),
            activation='relu',
            input_shape=inputs,
            return_sequences=True,
            weights=reference.layers[0].get_weights()
        )
    )
    model.add(Dense(NUM_CLASSES, weights=reference.layers[1].get_weights()))
    model.add(Activation('softmax'))
    rmsprop = RMSprop(lr=lr)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=rmsprop,
        metrics=['accuracy']
    )
    return model


def init_models(datas):
    models = {}
    for data_name, data in datas.items():
        for batch in BATCH_SIZES:
            for lr in LRS:
                for hidden_units in HIDDEN_UNITS:
                    model_name = '{}@{}@{}@{}'.format(data_name, batch, hidden_units, lr)
                    model = init_model(hidden_units, data.train.x.shape[1:], lr)
                    info = {
                        'name': model_name,
                        'data': data_name,
                        'batch': batch,
                        'hidden_units': hidden_units,
                        'lr': lr,
                        'loss': [],
                        'val_loss': [],
                        'acc': [],
                        'val_acc': []
                    }
                    models[model_name] = (model, info)
    logging.info('Inited %d models', len(models))
    return models

def read_models():
    models = {}
    for fname in os.listdir(MODELS_DIR):
        if not fname.endswith('.model'):
            continue
        model_name = os.path.splitext(fname)[0]
        with open(os.path.join(MODELS_DIR, model_name + '.json')) as fin:
            info = json.load(fin)

        models[model_name] = (
            keras.models.load_model(os.path.join(MODELS_DIR, fname)),
            info
        )
    return models

def _rm_rf(fname):
    try:
        os.unlink(fname)
    except OSError:
        pass


def save_model(model, info):
    try:
        os.makedirs(MODELS_DIR)
    except OSError:
        pass
    basename = os.path.join(MODELS_DIR, info['name'])
    _rm_rf(basename + '.model')
    _rm_rf(basename + '.json')

    model.save(basename + '.model')
    with open(basename + '.json', 'wt') as fout:
        json.dump(info, fout)

def train_model(model, info, datas):
    logging.info('Training model %s for epoch %d', info['name'], len(info['acc']))
    data = datas[info['data']]

    history = model.fit(
        data.train.x, data.train.y,
        batch_size=info['batch'],
        epochs=1,
        verbose=1,
        validation_data=(data.test.x, data.test.y)
    )

    info['acc'].extend(history.history['acc'])
    info['loss'].extend(history.history['loss'])
    info['val_acc'].extend(history.history['val_acc'])
    info['val_loss'].extend(history.history['val_loss'])

    return model, info


def main():
    datas = load_or_make_datas()
    if sys.argv[-1] == 'draw':
        for variant, data in datas.items():
            pixels = data.train.x[7]
            if pixels.shape[1] == 1:
                pixels = pixels.reshape(pixels.shape[0])
                pixels = np.repeat(pixels, 1).reshape([-1, 1])
                pixels = np.repeat(pixels, 10, axis=1)
            plt.axis('off')
            plt.title(variant)
            plt.imshow(pixels, cmap='gray')
            plt.show()
        exit()
    if os.path.isdir(MODELS_DIR):
        models = read_models()
    else:
        models = init_models(datas)
    if sys.argv[-1] == 'when':
        for variant, (model, info) in models.items():
            cmodel = reinit_model(info['hidden_units'], datas[info['data']].test.x.shape[1:], info['lr'], model)
            result = cmodel.predict(datas[info['data']].test.x)
            sum_places = 0
            for sample in result:
                answers = [-1]
                answers.extend(np.argmax(sample, axis=1))
                answer = answers[-1]
                while answers[-1] == answer:
                    answers.pop()
                sum_places += len(answers)
            print(variant, sum_places / len(result))
        exit()
    if sys.argv[-1] == 'damage':
        for variant, (model, info) in models.items():
            data = datas[info['data']].test
            was = model.evaluate(data.x, data.y, verbose=0)[1]
            print(variant, was)
            length = data.x.shape[0] * data.x.shape[1] * data.x.shape[2]
            for damage in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
                mask = np.zeros(length, dtype=int)
                mask[:int(length * damage)] = 1
                np.random.shuffle(mask)
                mask = mask.reshape(data.x.shape)
                nx = data.x * (1 - mask)
                now = model.evaluate(nx, data.y, verbose=0)[1]
                print(variant, damage, now)
        exit()



    while True:
        candidate = None
        for _, info in models.values():
            if candidate is None or len(info['acc']) < len(models[candidate][1]['acc']):
                candidate = info['name']

        model, info = models[candidate]
        model, info = train_model(model, info, datas)
        save_model(model, info)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
