# anogan gen-model

import numpy as np
import os
import yaml

from glob import glob
from os.path import exists, join
from stat import S_IRUSR, S_IWUSR, S_IXUSR

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.utils. generic_utils import Progbar
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam, Adamax, Nadam

from image import ImageDataGeneratorFCN
from data import load_data


def residual_loss(y_true, y_pred):
    '''
    :param y_true: target tensor
    :param y_pred: predict tensor
    :return: residual loss
    '''
    score = K.sum(K.abs(y_true - y_pred))
    return score


class anoGAN(object):

    param_names = ['batch_size', 'd_lr', 'd_optim', 'data_ch',
                   'data_size', 'epoch', 'g_lr', 'g_optim',
                   'image_dir', 'latent_size', 'loss_lambda',
                   'max_filters', 'save_dir']
    white_list = {'bmp', 'jpg', 'jpeg', 'png'}

    def __init__(self):
        self.batch_size = 16
        self.d_lr = 1e-4
        self.d_optim = Adam
        self.data_ch = 50
        self.data_size = 64
        self.epoch = 20
        self.flow_from_dir = True
        self.g_lr = 1e-4
        self.g_optim = Adam
        self.image_dir = None
        self.latent_size = 100
        self.loss_lambda = 0.1
        self.max_filters = 512
        self.save_dir = './'

    def _fetch_optim(self, optim_name):
        optims = {SGD.__name__:SGD,
                  RMSprop.__name__:RMSprop,
                  Adadelta.__name__:Adadelta,
                  Adamax.__name__:Adamax,
                  Adagrad.__name__:Adagrad,
                  Adam.__name__:Adam,
                  Nadam.__name__:Nadam}
        if optim_name in optims:
            return optims[optim_name]
        else:
            print('Warning: `optim_name` is invalid. '
                  'optimizer sets `Adam`.')
            return Adam

    def _set_params(self, yaml_path):
        with open(yaml_path) as f:
            params_dict = yaml.load(f)
        for key, value in params_dict.items():
            if key in self.param_names:
                setattr(self, key, value)
        self.d_optim = self._fetch_optim(self.d_optim)
        self.g_optim = self._fetch_optim(self.g_optim)

    def _save_yaml(self):
        yaml_path = join(self.save_dir, 'params.yaml')
        params_dict = {}
        for param in self.param_names:
            params_dict.update({param: getattr(self, param)})
        with open(yaml_path, mode='w') as f:
            f.write(yaml.dump(params_dict))

    def _set_trainable(self, model, trainable=False):
        model.trainable = trainable
        try:
            layers = model.layers
        except:
            return
        for layer in layers:
            self._set_trainable(layer, trainable)

    def Generator_model(self):
        '''
        :return: generator model
        '''
        resize_size = self.data_size // 8
        n_convs = np.log2(resize_size).astype(np.int)
        filter_sets = self.max_filters * 2 ** np.arange(n_convs)[::-1]

        input_gen = Input(shape=(self.latent_size,))

        x_gen = Dense(1024)(input_gen)
        x_gen = Activation('relu')(x_gen)

        x_gen = Dense(64 * filter_sets[0])(x_gen)
        x_gen = BatchNormalization()(x_gen)
        x_gen = Activation('relu')(x_gen)
        x_gen = Reshape((resize_size, resize_size, filter_sets[0]))(x_gen)

        for n, act in filter_sets[1:]:
            x_gen = Conv2DTranspose(n, (2, 2), strides=(2, 2), padding='same')(x_gen)
            x_gen = Conv2D(n, (5, 5), padding='same')(x_gen)
            x_gen = BatchNormalization()(x_gen)
            x_gen = Activation('relu')(x_gen)

        x_gen = UpSampling2D((2, 2))(x_gen)
        x_gen = Conv2D(self.data_ch, (5, 5), padding='same')(x_gen)
        x_gen = BatchNormalization()(x_gen)
        x_gen = Activation('tanh')(x_gen)

        output_gen = x_gen

        model = Model(inputs=[input_gen], outputs=[output_gen])

        return model

    def Discriminator_model(self):
        '''
        :return: discriminator model
        '''
        size_predense = self.data_size // 8
        n_convs = np.log2(size_predense).astype(np.int)
        filter_sets = self.max_filters * 2 ** np.arange(n_convs + 1)

        input_dis = Input(shape=(self.data_size, self.data_size, self.data_ch))
        x_dis = input_dis

        for n, act in filter_sets[:-1]:
            x_dis = Conv2D(n, (5, 5), stride=(2, 2), padding='same')(x_dis)
            x_dis = BatchNormalization()(x_dis)
            x_dis = LeakyReLU(alpha=0.2)(x_dis)

        x_dis = Flatten()(x_dis)
        x_dis = Dense(size_predense * size_predense *  filter_sets[-1])(x_dis)
        x_dis = LeakyReLU(alpha=0.2)(x_dis)
        x_dis = Dropout(0.5)(x_dis)

        x_dis = Dense(1024)(x_dis)
        x_dis = LeakyReLU(alpha=0.2)(x_dis)
        x_dis = Dropout(0.5)(x_dis)

        x_dis = Dense(1)(x_dis)
        output_dis = Activation('softmax')(x_dis)

        model = Model(inputs=[input_dis], outputs=[output_dis])

        return model

    def GAN_model(self, generator, discriminator):
        '''
        :return: gan model
        '''
        self._set_trainable(discriminator, trainable=False)
        input_gan = Input(shape=(self.latent_size,))
        x_gan = generator(input_gan)
        output_gan = discriminator(x_gan)
        model = Model(inputs=[input_gan], outputs=[output_gan])

        return model

    def Feature_model(self, discriminator):
        model = Model(inputs=discriminator.layer[0].input,
                      outputs=discriminator.layers[-10].output)
        self._set_trainable(model, trainable=False)

        return model

    def Detector_mode(self, generator, discriminator):
        input_latent = Input(shape=(self.latent_size,))
        x_latent = Conv1D(1, self.latent_size)(input_latent)
        x_latent = Activation('sigmoid')(x_latent)

        generator = Model(inputs=generator.layers[1].input, outputs=generator.layers[-1].output)
        generator.trainable = False

        feature = self.Feature_model(discriminator)

        output_gen = generator(x_latent)
        output_fea = feature(output_gen)

        model = Model(inputs=input_latent, outputs=[output_gen, output_fea])

        return model
        
    def train(self, yaml_path):

        self._set_params(yaml_path)

        if not exists(self.save_dir):
            os.makedirs(self.save_dir)
            os.chmod(self.save_dir, S_IRUSR | S_IWUSR | S_IXUSR)

        self._save_yaml()
        size = (self.data_size, self.data_size)

        d = self.Discriminator_model()
        g = self.Generator_model()
        gan = self.GAN_model(d, g)
        g.compile(loss='mse', optimizer=self.g_optim(lr=self.g_lr))
        gan.compile(loss='mse', optimizer=self.g_optim(lr=self.g_lr))
        d.trainable = True
        d.compile(loss='mse', optimizer=self.d_optim(lr=self.d_lr))

        generator = ImageDataGeneratorFCN(rotation_range=0.,
                                          width_shift_range=0.,
                                          height_shift_range=0.,
                                          shear_range=0.,
                                          zoom_range=0.,
                                          fill_mode='nearest',
                                          cval=0.,
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          gamma_shift_base=0,
                                          hue_shift_range=0,
                                          lightness_shift_range=0,
                                          saturation_shift_range=0,
                                          autoencoder_mode=False,
                                          cnn_mode=True,
                                          minmax_standerdize=True,
                                          minmax_from=(0, 255),
                                          minmax_to=(-1, 1))

        if self.flow_from_dir:
            real_path = []
            for ext in self.white_list:
                real_path += glob(join(self.image_dir, '*.' + ext))
            n_iter = len(real_path) // self.batch_size
            g_flow = generator.flow_from_directory_fcn(
                    real_path, None,
                    target_size=size, color_mode='rgb',
                    class_mode='categorical', class_palette=None,
                    batch_size=self.batch_size, shuffle=True, seed=None,
                    save_to_dir=None,
                    save_prefix='',
                    save_format='png',
                    follow_links=False)
        else:
            ext = ['png', 'jpg', 'jpeg', 'bmp']
            images, _, _ = load_data(self.image_dir, ext=ext,
                                     color_mode='color', dtype=np.float32,
                                     size=size, resize_type='ec',
                                     load_lbl=True, lbl_dir='lbl', lbl_suf='_lbl',)
            g_flow = generator.flow_fcn(images, None, batch_size=self.batch_size,  class_mode='categorical',
                                        class_palette=None, shuffle=True, seed=None,
                                        save_to_dir=None, save_prefix='', save_format='png')

        for ep in range(self.epoch):
            print('Epoch: {}/{}'.format(ep, self.epoch))

            progress_bar = Progbar(target=n_iter)

            for idx in range(n_iter):
                noise = np.random.uniform(0, 1, size=(self.batch_size, self.latent_size))

                real_img, _ = g_flow.next()

                fake_img = g.predict(noise, verbose=0)

                X = np.concatenate([real_img, fake_img], axis=0)
                y = np.array([1.] * len(real_img) + [0.] * len(fake_img))

                d_loss = d.train_on_batch(X, y)

                self._set_trainable(d, trainable=False)
                g_loss = gan.train_on_batch(noise, np.array([1] * self.batch_size))
                self._set_trainable(d, trainable=True)

                progress_bar.update(idx, values=[('g', g_loss), ('d', d_loss)])
            print('')

            g.save_weights(join(self.save_dir, 'g_weights.h5'), True)
            d.save_weights(join(self.save_dir, 'd_weights.h5'), True)

        return d, g

    def detect(self, x, generator, discriminator, iterations=500):
        z = np.random.uniform(0, 1, size=(1, self.latent_size))

        feature = self.Feature_model(discriminator)
        feature.compile(loss='binary_crossentropy', optimizer=self.d_optim(lr=self.d_lr))

        detector =  self.Detector_mode(generator, discriminator)
        detector.compile(loss=residual_loss, loss_weights=[1. - self.loss_lambda, self.loss_lambda],
                         optimizer=self.d_optim(lr=self.d_lr))

        features = feature.predict(x)

        loss = detector.fit(z, [x, features], epochs=iterations, verbose=0)
        detections = detector.predict(z)

        loss = loss.history['loss'][-1]

        return loss, detections
