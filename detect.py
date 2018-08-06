# anoGAN test
import cv2
import numpy as np
import csv
import time
import os
from glob import glob
from os.path import exists, join
from stat import S_IRUSR, S_IWUSR, S_IXUSR

from misc import str_num
import anogan


def _mnist_data_load(path):
    lst = []
    labels = []
    for i in range(10):
        lst += glob(join(path, str(i), '*.png'))[:5]
        labels += [i] * 5
    labels = np.array(labels)
    images = []
    for p in lst:
        images.append(cv2.imread(p, 0))
    images = np.array(images, np.float32) * 2. / 255. - 1.
    images = images[:, :, :, np.newaxis]
    return images, labels


def main(save_path='./', data_path='./'):
    if not exists(save_path):
        os.makedirs(save_path)
        os.chmod(save_path, S_IRUSR | S_IWUSR | S_IXUSR)
    images, labels = _mnist_data_load(data_path)
    ano = anogan.anoGAN()
    for i, (img, lbl) in enumerate(zip(images, labels)):
        t1 = time.time()
        img = img[np.newaxis, :, :, :]
        loss, detections = ano.detect(img, './params.yaml')
        detections = (detections[0, :, :, 0] + 1.) * 255. / 2.
        img = img[0, :, :, 0] * 255.
        diff = np.abs(img - detections).astype(np.uint8)
        img = img.astype(np.uint8)
        detections = detections.astype(np.uint8)
        img_name = str_num(i + 1, 2) + '_' + str(lbl) + '.png'
        dt_name = str_num(i + 1, 2) + '_' + str(lbl) + '_dtc.png'
        diff_name = str_num(i + 1, 2) + '_' + str(lbl) + '_diff.png'
        cv2.imwrite(join(save_path, img_name), img)
        cv2.imwrite(join(save_path, dt_name), detections)
        cv2.imwrite(join(save_path, diff_name), diff)
        t2 = time.time()
        proc_time = t2 - t1
        print([img_name, loss, proc_time])
        with open(join(save_path, 'score.csv'), 'a', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([img_name, loss, proc_time])

if __name__ == '__main__':
    save_path = input('input save path: ')
    print('set save dir: %s' % save_path)
    data_path = input('input test data path: ')
    print('set test data dir: %s' % data_path)
    main(save_path, data_path)
