# anoGAN test
import cv2
import numpy as np
import csv
from glob import glob
from os.path import join

from misc import str_num
import anogan


def _mnist_data_load():
    path = '/work/tsutsumi/open_datasets/mnist'
    lst = []
    labels = []
    for i in range(10):
        lst += glob(join(path, str(i), '*.png'))[:5]
        labels += [i] * 5
    labels = np.array(labels)
    images = []
    for p in lst:
        images.append(cv2.imread(p, 0))
    images = np.array(images, np.float32) / 255.
    images = images[:, :, :, np.newaxis]
    return images, labels


def main():
    images, labels = _mnist_data_load()
    ano = anogan.anoGAN()
    with open('./mnist_test/detect/score.csv', 'a', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i, (img, lbl) in enumerate(zip(images, labels)):
            img = img[np.newaxis, :, :, :]
            loss, detections = ano.detect(img, './params.yaml')
            detections = ((detections[0, :, :, 0] + 1.) * 255. / 2.).astype(np.uint8)
            img = (img[0, :, :, 0] * 255.).astype(np.uint8)
            img_name = str_num(i + 1, 2) + '_' + str(lbl) + '.png'
            dt_name = str_num(i + 1, 2) + '_' + str(lbl) + '_d.png'
            cv2.imwrite(join('./mnist_test/detect/', img_name), img)
            cv2.imwrite(join('./mnist_test/detect/', dt_name), detections)
            print([img_name, loss])
            writer.writerow([img_name, loss])

if __name__ == '__main__':
    main()
