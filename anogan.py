# anogan gen-model

import numpy as np
import os
import yaml
import shutil
import cv2

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

    param_names = ['batch_size', 'd_lr', 'd_optim',
                   'data_ch', 'data_size', 'epoch',
                   'g_final_filter', 'g_lr',
                   'g_optim', 'image_dir', 'latent_size',
                   'loss_lambda', 'max_filters', 'n_convs',
                   'save_dir']
    white_list = {'bmp', 'jpg', 'jpeg', 'png'}

    def __init__(self):
        self.batch_size = 16
        self.color_mode = 'rgb'
        self.d_lr = 1e-4
        self.d_optim = Adam
        self.data_ch = 3
        self.data_size = 64
        self.epoch = 20
        self.flow_from_dir = True
        self.g_final_filters = 64
        self.g_lr = 1e-4
        self.g_optim = Adam
        self.image_dir = None
        self.latent_size = 100
        self.loss_lambda = 0.1
        self.n_convs = 4
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
        # check validity between 'data_size' and 'n_convs'
        tmp_convs = 0
        for i in range(self.n_convs, 0, -1):
            k = 2 ** i
            if self.data_size % k != 0:
                self.n_convs = tmp_convs
                print('Warning: `n_convs` is invalid. '
                      '`n_convs` sets {}.'.format(tmp_convs))
                break
            tmp_convs = i
        self.d_optim = self._fetch_optim(self.d_optim)
        self.g_optim = self._fetch_optim(self.g_optim)

        if self.data_ch == 3:
            self.color_mode = 'rgb'
        elif self.data_ch == 1:
            self.color_mode = 'grayscale'
        else:
            self.color_mode = 'rgb'
            print('Warning: `data_ch` is invalid. '
                  '`data_ch` sets rgb mode.')

        if not exists(self.save_dir):
            os.makedirs(self.save_dir)
            os.chmod(self.save_dir, S_IRUSR | S_IWUSR | S_IXUSR)
        shutil.copyfile(yaml_path, join(self.save_dir, 'params.yaml'))

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
        filter_sets = self.g_final_filters * 2 ** np.arange(self.n_convs)[::-1]
        resize_size = self.data_size // (2 ** self.n_convs)

        input_gen = Input(shape=(self.latent_size,))

        x_gen = Dense(1024)(input_gen)
        x_gen = Activation('relu')(x_gen)

        x_gen = Dense(resize_size * resize_size * filter_sets[0])(x_gen)
        x_gen = BatchNormalization()(x_gen)
        x_gen = Activation('relu')(x_gen)
        x_gen = Reshape((resize_size, resize_size, filter_sets[0]))(x_gen)

        for n in filter_sets:
            x_gen = Conv2D(n, (5, 5), padding='same')(x_gen)
            x_gen = BatchNormalization()(x_gen)
            x_gen = Activation('relu')(x_gen)
            x_gen = Conv2DTranspose(n, (2, 2), strides=(2, 2), padding='same')(x_gen)

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
        filter_sets = self.g_final_filters * 2 ** np.arange(self.n_convs)

        input_dis = Input(shape=(self.data_size, self.data_size, self.data_ch))
        x_dis = input_dis

        for n in filter_sets:
            x_dis = Conv2D(n, (5, 5), strides=(2, 2), padding='same')(x_dis)
            x_dis = BatchNormalization()(x_dis)
            x_dis = LeakyReLU(alpha=0.2)(x_dis)

        x_dis = Flatten()(x_dis)

        x_dis = Dense(1024)(x_dis)
        x_dis = LeakyReLU(alpha=0.2)(x_dis)
        x_dis = Dropout(0.5)(x_dis)

        x_dis = Dense(1)(x_dis)
        output_dis = Activation('sigmoid')(x_dis)

        model = Model(inputs=[input_dis], outputs=[output_dis])

        return model

    def GAN_model(self, generator, discriminator):
        '''
        :return: gan model
        '''
        self._set_trainable(discriminator, trainable=False)
        discriminator.trainable = False
        input_gan = Input(shape=(self.latent_size,))
        x_gan = generator(input_gan)
        output_gan = discriminator(x_gan)
        model = Model(inputs=[input_gan], outputs=[output_gan])

        return model

    def Feature_model(self, discriminator):
        model = Model(inputs=discriminator.layers[0].input,
                      outputs=discriminator.layers[-10].output)
        self._set_trainable(model, trainable=False)

        return model

    def Detector_model(self, generator, discriminator):
        input_latent = Input(shape=(self.latent_size,))
        x_latent = Dense(self.latent_size)(input_latent)
        x_latent = Activation('sigmoid')(x_latent)

        generator = Model(inputs=generator.layers[1].input, outputs=generator.layers[-1].output)
        self._set_trainable(generator, trainable=False)

        feature = self.Feature_model(discriminator)

        output_gen = generator(x_latent)
        output_fea = feature(output_gen)

        model = Model(inputs=input_latent, outputs=[output_gen, output_fea])

        return model
        
    def train(self, yaml_path):

        self._set_params(yaml_path)

        size = (self.data_size, self.data_size)

        g = self.Generator_model()
        g.summary()

        d = self.Discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer=self.d_optim(lr=self.d_lr))
        d.summary()

        gan = self.GAN_model(generator=g, discriminator=d)
        gan.compile(loss='binary_crossentropy', optimizer=self.g_optim(lr=self.g_lr))
        gan.summary()
        self._set_trainable(d, trainable=True)

        # generator = ImageDataGeneratorFCN(rotation_range=0.,
        #                                   width_shift_range=0.,
        #                                   height_shift_range=0.,
        #                                   shear_range=0.,
        #                                   zoom_range=0.,
        #                                   fill_mode='nearest',
        #                                   cval=0.,
        #                                   horizontal_flip=False,
        #                                   vertical_flip=False,
        #                                   gamma_shift_base=0,
        #                                   hue_shift_range=0,
        #                                   lightness_shift_range=0,
        #                                   saturation_shift_range=0,
        #                                   autoencoder_mode=False,
        #                                   cnn_mode=True,
        #                                   minmax_standerdize=True,
        #                                   minmax_from=(0, 255),
        #                                   minmax_to=(-1, 1))

        # if self.flow_from_dir:
        #     real_path = []
        #     for ext in self.white_list:
        #         real_path += glob(join(self.image_dir, '*.' + ext))
        #     n_iter = len(real_path) // self.batch_size
        #     g_flow = generator.flow_from_directory_fcn(
        #             real_path, None,
        #             target_size=size, color_mode=self.color_mode,
        #             class_mode='categorical', class_palette=None,
        #             batch_size=self.batch_size, shuffle=True, seed=None,
        #             save_to_dir='./mnist_test/gen_img',
        #             save_prefix='',
        #             save_format='png',
        #             follow_links=False)
        #     print('input data: {}'.format(len(real_path)))
        # else:
        #     ext = ['png', 'jpg', 'jpeg', 'bmp']
        #     images, _, _ = load_data(self.image_dir, ext=ext,
        #                              color_mode=self.color_mode, dtype=np.float32,
        #                              size=size, resize_type='ec',
        #                              load_lbl=True, lbl_dir='lbl', lbl_suf='_lbl',)
        #     n_iter = len(images) // self.batch_size
        #     g_flow = generator.flow_fcn(images, None, batch_size=self.batch_size,  class_mode='categorical',
        #                                 class_palette=None, shuffle=True, seed=None,
        #                                 save_to_dir='./mnist_test/gen_img', save_prefix='', save_format='png')
        #     print('input data: {}'.format(len(images)))

        ext = ['png', 'jpg', 'jpeg', 'bmp']
        if self.flow_from_dir:
            real_path = []
            for ext in self.white_list:
                real_path += glob(join(self.image_dir, '*.' + ext))
            data_len = len(real_path)
            n_iter = data_len // self.batch_size
            real_path = np.array(real_path)
            print('input data: {}'.format(data_len))

        else:
            images, _, _ = load_data(self.image_dir, ext=ext,
                                     color_mode=0, dtype=np.float32,
                                     size=size, resize_type='ec',
                                     load_lbl=False, lbl_dir='lbl', lbl_suf='_lbl',)
            images = 2. * images / 255. - 1.
            data_len = len(images)
            n_iter = data_len // self.batch_size
            print('input data: {}'.format(data_len))

        for ep in range(self.epoch):
            print('Epoch: {}/{}'.format(ep + 1, self.epoch))

            progress_bar = Progbar(target=n_iter)
            random_idx = np.random.permutation(data_len)

            for idx in range(n_iter):
                # real_img, _ = g_flow.next()
                pick_idx = random_idx[(self.batch_size * idx):(self.batch_size * (idx + 1))]
                if self.flow_from_dir:
                    real_img, _, _ = load_data(real_path[pick_idx], ext=None,
                                               color_mode=0, dtype=np.float32,
                                               size=size, resize_type='ec',
                                               load_lbl=False, lbl_dir='lbl', lbl_suf='_lbl', )
                    real_img = 2. * real_img / 255. - 1.
                else:
                    real_img = images[pick_idx]

                noise = np.random.uniform(0, 1, size=(self.batch_size, self.latent_size))
                fake_img = g.predict(noise, verbose=0)

                X = np.concatenate([real_img, fake_img], axis=0)
                y = np.array([1.] * len(real_img) + [0.] * len(fake_img))
                # d.trainable = True
                d_loss = d.train_on_batch(X, y)
                # d.trainable = False
                g_loss = gan.train_on_batch(noise, np.array([1.] * self.batch_size))

                progress_bar.update(idx, values=[('g-loss', g_loss), ('d-loss', d_loss)])
                if idx == n_iter - 1:
                    np.save(join(self.save_dir, 'e{}.npy'.format(ep)), X)

            print('')
            # save weights per epoch for test
            if ep % 20 == 1:
                g.save_weights(join(self.save_dir, 'g_weights_e{}.h5'.format(ep)), True)
                d.save_weights(join(self.save_dir, 'd_weights_e{}.h5'.format(ep)), True)
                img_save_dir = join(self.save_dir, 'gen_img', str(ep))
                if not exists(img_save_dir):
                    os.makedirs(img_save_dir)
                    os.chmod(img_save_dir, S_IRUSR | S_IWUSR | S_IXUSR)
                for i, img in enumerate(fake_img):
                    img = ((img[:, :, 0] - 1.) * 255. / 2.).astype(np.uint8)
                    img_name = 'ep' + str(ep) + '_' + str(i) + '.png'
                    cv2.imwrite(join(img_save_dir, img_name), img)

        return d, g

    def detect(self, x, yaml_path, iterations=500):

        self._set_params(yaml_path)

        g = self.Generator_model()
        g.load_weights(join(self.save_dir, 'g_weights_e{}.h5'.format(self.epoch - 1)))

        d = self.Discriminator_model()
        d.load_weights(join(self.save_dir, 'd_weights_e{}.h5'.format(self.epoch - 1)))

        z = np.random.uniform(0, 1, size=(1, self.latent_size))

        feature = self.Feature_model(d)
        feature.summary()

        detector = self.Detector_model(g, d)
        detector.compile(loss=residual_loss, loss_weights=[1. - self.loss_lambda, self.loss_lambda],
                         optimizer=self.d_optim(lr=self.d_lr))
        detector.summary()

        features = feature.predict(x)

        loss = detector.fit(z, [x, features], epochs=iterations, verbose=0)
        detections, _ = detector.predict(z)

        loss = loss.history['loss'][-1]

        return loss, detections
