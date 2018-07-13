# image data generator for fcn

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
from functools import partial

# Additional
from skimage import data, img_as_float
from skimage.exposure import adjust_gamma
from skimage import io
import cv2
from glob import glob

from keras import backend as K
from keras.preprocessing.image_fcn import *

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


def _sig_trans(x):
    x = 255. / (1. + np.exp(-1. * 4. / 255. * (x - 255. / 2)))
    return x


def random_gamma_shift(x, gamma):
    x = adjust_gamma(x, gamma=gamma, gain=1)
    return x


def random_hls_shift(x, hls_range):
    if len(hls_range) != 3:
        raise ValueError('`hls_range` should be a tuple or list of three floats. '
                         'Received arg: ', hls_range)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2HLS)
    x[:, :, 0] = (x[:, :, 0] + hls_range[0]) % 180
    x[:, :, 1] = _sig_trans(x[:, :, 1] + hls_range[1])
    x[:, :, 2] = _sig_trans(x[:, :, 2] + hls_range[2])
    x = x.astype(np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_HLS2RGB)
    return x


def load_to_array(path, grayscale=False, target_size=None):
    array = io.imread(path)
    if target_size is not None:
        array = cv2.resize(array, target_size)
    if grayscale:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    return array


def rgb_to_categorical(ary, palette, idx=None, dtype=np.float):
    # 変換したいデータのnumpy arrayへの変換
    ary = np.array(ary, dtype=np.int)  # uint8だとオーバーフローするのでintに変換
    ary_dim = ary.ndim
    # パレット情報がDictionaryである場合のデータ読み込み
    if isinstance(palette, dict):
        idx = list(palette.keys())  # Keyをcolor indexとして読み込む
        idx = np.array(idx)
        palette = list(palette.values())  # ValueをRGBとして読み込む
    palette = np.array(palette, dtype=np.int)
    pl_dim = palette.ndim

    # 末尾の次元が3の場合RGB画像と判断して、
    # 3channlesから1channel(HTML表記の10進数版)へ変換
    if ary.shape[ary_dim - 1] == 3:
        coeff = np.logspace(2, 0, num=3, base=16).astype(np.int)
        # HTML表記を10進数に直した形にする計算のための係数
        r_tuple = np.ones(ary_dim - 1, dtype=np.int)
        r_tuple = tuple(r_tuple) + (3,)
        coeff = coeff.reshape(r_tuple)  # 念のためaryの最終次元に合うよう係数行列を変形
        ary = (ary * coeff).sum(axis=ary_dim - 1)  # 積をとって形式を変更

    # aryと同じ処理をpaletteにも実施
    if palette.shape[pl_dim - 1] == 3:
        coeff = np.logspace(2, 0, num=3, base=16).astype(np.int)
        r_tuple = np.ones(pl_dim - 1, dtype=np.int)
        r_tuple = tuple(r_tuple) + (3,)
        coeff = coeff.reshape(r_tuple)
        palette = (palette * coeff).sum(axis=pl_dim - 1)

    # paletteの次元が1より大きい場合はflatten
    if palette.ndim > 1:
        palette = palette.ravel()

    # indexが与えられていない場合paletteのパレットの指定順をindexとする
    if idx is None:
        idx = np.arange(palette.shape[0])

    n_idx = idx.shape[0]

    rs_ary = ary.shape + (1,)
    ary = ary.reshape(rs_ary)  # 末尾に次元を増やしておく
    ary = np.tile(ary, n_idx)  # indexの数までコピーしておく
    ary = np.where(ary == palette, 1, 0).astype(dtype)  # aryとpaletteとの一致不一致を判定

    return ary


class ImageDataGeneratorFCN(ImageDataGenerator):

    def __init__(self,
                 gamma_shift_base=0,
                 hue_shift_range=0,
                 lightness_shift_range=0,
                 saturation_shift_range=0,
                 autoencoder_mode=False,
                 minmax_standerdize=True,
                 minmax_from=(0, 255),
                 minmax_to=(0, 1)):
        self.gamma_shift_base = gamma_shift_base
        if not (hue_shift_range or lightness_shift_range or saturation_shift_range):
            self.hls_shift_range = 0
        else:
            self.hls_shift_range = [hue_shift_range, lightness_shift_range,
                                    saturation_shift_range]
        self.autoencoder_mode = autoencoder_mode
        self.minmax_standerdize = minmax_standerdize
        self.minmax_from = minmax_from
        self.minmax_to = minmax_to

        super(ImageDataGeneratorFCN, self).__init__()

    def flow_fcn(self, x, y, batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        return NumpyArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def flow_from_directory_fcn(self, directory_img, directory_tgt,
                                target_size=(256, 256), color_mode='rgb',
                                class_mode='categorical', class_palette=None,
                                batch_size=32, shuffle=True, seed=None,
                                save_to_dir=None,
                                save_prefix='',
                                save_format='png',
                                follow_links=False):
        return DirectoryIteratorFCN(
            directory_img, directory_tgt, self,
            target_size=target_size, color_mode=color_mode,
            class_mode=class_mode, class_palette=class_palette,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links)

    def standardize_fcn(self, x):
        """Apply the normalization configuration to a batch of inputs.

        # Arguments
            x: batch of inputs to be normalized.

        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale

        if self.minmax_standerdize:
            x = (x - self.minmax_from[0]
                 / (self.minmax_from[1] - self.minmax_from[0]))
            x = x * (self.minmax_to[1] - self.minmax_to[0]) + self.minmax_to[0]

        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = self.channel_axis - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')

        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')

        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, x.size)
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    def random_hls_gamma(self, x, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if self.hls_shift_range:
            h = np.random.uniform(-self.hls_shift_range[0], self.hls_shift_range[0]) * 180.
            l = np.random.uniform(-self.hls_shift_range[1], self.hls_shift_range[1]) * 255.
            s = np.random.uniform(-self.hls_shift_range[2], self.hls_shift_range[2]) * 255.
            x = random_hls_shift(x, [h, l, s])

        if self.gamma_shift_base:
            gamma = self.gamma_shift_base ** np.random.uniform(-1., 1.)
            x = random_gamma_shift(x, gamma)
        return x

    def random_transform_fcn(self, x, y, seed=None):
        """Randomly augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.
            seed: random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)
            y = apply_transform(y, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                y = flip_axis(y, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                y = flip_axis(y, img_row_axis)

        return x, y


class NumpyArrayIteratorFCN(Iterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """
    def __init__(self, x, y, image_data_generator,
                 class_mode='categorical', class_palette=None,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        if data_format is None:
            data_format = K.image_data_format()
        self.class_mode = class_mode
        self.class_palette = class_palette
        self.x = x
        if self.x.ndim == 3:
            x = x[:, :, :, np.newaxis]
            self.class_num = 1
        elif self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        self.channels_axis = 3 if data_format == 'channels_last' else 1
        if y is None:
            self.autoencoder_mode = True
            self.class_num = x.shape[self.channels_axis]
        else:
            self.autoencoder_mode = False
            self.y = y
            self.class_num = len(self.class_palette)
        # if self.x.shape[channels_axis] not in {1, 3, 4}:
        #     raise ValueError('NumpyArrayIterator is set to use the '
        #                      'data format convention "' + data_format + '" '
        #                      '(channels on axis ' + str(channels_axis) + '), i.e. expected '
        #                      'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
        #                      'However, it was passed an array with shape ' + str(self.x.shape) +
        #                      ' (' + str(self.x.shape[channels_axis]) + ' channels).')

        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIteratorFCN, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
        if self.channels_axis == 1:
            batch_y = np.zeros((current_batch_size,) + (self.class_num,)
                               + self.x.shape[2:4], dtype=K.floatx())
        else:
            batch_y = np.zeros((current_batch_size,) + self.x.shape[1:3]
                               + (self.class_num,), dtype=K.floatx())
        if self.autoencoder_mode:
            for i, j in enumerate(index_array):
                x = self.x[j]
                if x.shape[self.channels_axis - 1] > 2:
                    if self.channels_axis == 1:
                        x = x.transpose(1, 2, 0)
                    x[:, :, :3] = self.image_data_generator.random_hls_gamma(x[:, :, :3])
                    if self.channels_axis == 1:
                        x = x.transpose(2, 0, 1)
                x = self.image_data_generator.random_transform(x.astype(K.floatx()))
                x = self.image_data_generator.standardize_fcn(x)
                batch_x[i] = x
                batch_y[i] = x
            if self.save_to_dir:
                for i in range(current_batch_size):
                    img = array_to_img(batch_x[i], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                      index=current_index + i,
                                                                      hash=np.random.randint(10000),
                                                                      format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
        else:
            categorical = self.class_mode == 'categorical'
            for i, j in enumerate(index_array):
                x = self.x[j]
                y = self.y[j]
                if x.shape[self.channels_axis - 1] > 2:
                    if self.channels_axis == 1:
                        x = x.transpose(1, 2, 0)
                    x[:, :, :3] = self.image_data_generator.random_hls_gamma(x[:, :, :3])
                    if self.channels_axis == 1:
                        x = x.transpose(2, 0, 1)
                if categorical:
                    y = rgb_to_categorical(y, self.class_palette, dtype=K.floatx())
                else:
                    if self.channels_axis == 1:
                        y = y[np.newaxis, :, :]
                    else:
                        y = y[:, :, np.newaxis]
                    y = y.astype(K.floatx())
                    y = self.image_data_generator.standardize_fcn(y)
                x, y = self.image_data_generator.random_transform_fcn(x, y)
                x = self.image_data_generator.standardize_fcn(x)
                batch_x[i] = x
                batch_y[i] = y

            if self.save_to_dir:
                for i in range(current_batch_size):
                    img = array_to_img(batch_x[i], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                      index=current_index + i,
                                                                      hash=np.random.randint(10000),
                                                                      format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
                    img = array_to_img(batch_y[i], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}_lbl.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(10000),
                                                                  format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        return batch_x, batch_y


class DirectoryIteratorFCN(Iterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, directory_img, directory_tgt,
                 image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 class_mode='binary', class_palette=None,
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory_img = directory_img
        self.directory_tgt = directory_tgt
        if self.directory_tgt is None:
            self.autoencoder_mode = True
        else:
            self.autoencoder_mode = False
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.class_palette = class_palette
        self.data_format = data_format

        if class_mode == 'categorical':
            self.num_class = len(self.class_palette)
        else:
            self.num_class = 1
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
            if self.autoencoder_mode:
                self.num_class = 3
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
            if self.autoencoder_mode:
                self.num_class = 1
        # self.classes = classes
        if class_mode not in {'categorical', 'binary', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.samples = 0

        self.name_img = []
        self.name_tgt = []

        for ext in white_list_formats:
            self.name_img += [os.path.basename(r) for r in glob(os.path.join(directory_img, ext))]
            if self.autoencoder_mode:
                self.name_tgt += [os.path.basename(r) for r in glob(os.path.join(directory_tgt, ext))]
            else:
                self.name_tgt = None
        if len(self.name_img) == len(glob(directory_tgt)) or self.autoencoder_mode:
            self.samples = len(self.name_img)
        else:
            raise ValueError('invalid data number',
                             'data number {} target number {}'
                             'these number should be the '
                             'same.'.format(len(self.name_img),
                                            len(self.name_tgt)))

        print('Found %d images.' % self.samples)

        super(DirectoryIteratorFCN, self).__init__(self.samples, batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros((current_batch_size,) + self.target_size
                           + (self.num_class,), dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        binary = self.class_mode == 'binary'
        # build batch of image data
        if self.autoencoder_mode:
            for i, j in enumerate(index_array):
                fname = self.name_img[j]
                img = load_to_array(os.path.join(self.directory_img, fname),
                                    grayscale=grayscale, target_size=self.target_size)
                if not grayscale:
                    img = self.image_data_generator.random_hls_gamma(img)
                img = img.astype(K.floatx())
                img = self.image_data_generator.random_transform(img)
                img = self.image_data_generator.standardize_fcn(img)
                batch_x[i] = img
                batch_y[i] = img
        else:
            for i, j in enumerate(index_array):
                fname = self.name_img[j]
                img = load_to_array(os.path.join(self.directory_img, fname),
                                    grayscale=grayscale, target_size=self.target_size)
                if not grayscale:
                    img = self.image_data_generator.random_hls_gamma(img)
                img = img.astype(K.floatx())
                fname = self.name_tgt[j]
                tgt = load_to_array(os.path.join(self.directory_img, fname),
                                    grayscale=binary, target_size=self.target_size)
                if not binary:
                    tgt = rgb_to_categorical(tgt, self.class_palette, dtype=K.floatx())
                else:
                    tgt = tgt.astype(K.floatx())
                img, tgt = self.image_data_generator.random_transform_fcn(img, tgt)
                img = self.image_data_generator.standardize_fcn(img)
                batch_x[i] = img
                batch_y[i] = tgt
            # optionally save augmented images to disk for debugging purposes
            if self.save_to_dir:
                for i in range(current_batch_size):
                    img = array_to_img(batch_x[i], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                      index=current_index + i,
                                                                      hash=np.random.randint(10000),
                                                                      format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
                    img = array_to_img(batch_y[i], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}_lbl.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(10000),
                                                                  format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        return batch_x, batch_y
