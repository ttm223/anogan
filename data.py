import numpy as np
import cv2
from glob import glob
from os.path import basename, join, split, splitext
import re


def load_data(src, ext='png',
              color_mode='color', dtype=np.float32,
              size=None, resize_type='ec',
              load_lbl=True, lbl_dir='lbl', lbl_suf='_lbl',
              load_path=True):
    '''
    src: str of file or list of file path
    ext: set 0 in case of 'src' is file path,
         set extension of file in case of 'src' is dir path
    color_mode: set 'color', 'COLOR' or 1 for color mode
                set 'gray', 'GRAY' or 0 for grayscale mode
    dtype: set numpy data type for output data
    size: 2D tuple for resize
    resize_type: resize method. if 'cut', cut out from input data center
    load_lbl: if true, label image loads at once
    lbl_dir: if str,
    lbl_suf: suffix of label image
    load_path: if true, output image name
    '''

    # generate path list
    if ext:
        if isinstance(ext, str):
            ext = ['*.' + ext]
        else:
            tmp = []
            for e in ext:
                tmp += ['*.' + e]
            ext = tmp
        if isinstance(src, str):
            src = [src]
        tmp = []
        for e in ext:
            for s in src:
                tmp += glob(join(s, e))
        src = tmp
    elif (not ext) and isinstance(src, str):
        src = [src]
    elif not ext:
        pass
    else:
        ValueError('`ext` expects str or list.')
    src.sort()
    print(src)
    if load_lbl and lbl_dir:
        img_path = []
        lbl_path = []
        for s in src:
            dir_path, name = split(s)
            name, ext = splitext(name)
            lp = join(dir_path, lbl_dir, (name + lbl_suf + ext))
            tmp = glob(lp)
            if len(tmp) == 1:
                img_path.append(s)
                lbl_path.append(tmp)

    # get label path
    elif load_lbl and (not lbl_dir):
        img_path = []
        lbl_path = []
        lbl_comp = re.compile(lbl_suf + r'.')
        name_comp = re.compile('')
        for s in src:
            if lbl_comp.search(s) is None:
                img_path.append(s)
                name_comp = re.compile(splitext(basename(img_path[:-1])))
            else:
                lbl_path.append(s)
                if len(img_path) > len(lbl_path):
                    img_path = img_path[:-1]
                elif len(img_path) < len(lbl_path):
                    lbl_path = lbl_path[:-1]
                else:
                    if name_comp.search(s) is None:
                        img_path = img_path[:-1]
                        lbl_path = lbl_path[:-1]
    else:
        img_path = src

    # set rgb or grayscale
    if (color_mode == 'RGB' or color_mode == 'rgb'
        or color_mode == cv2.IMREAD_COLOR):
        color_mode = cv2.IMREAD_COLOR
    elif (color_mode == 'GRAYSCALE' or color_mode == 'grayscale'
          or color_mode == cv2.IMREAD_GRAYSCALE):
        color_mode = cv2.IMREAD_GRAYSCALE
    else:
        print('variant `color_mode` is invalid value.'
              '`color_mode` sets color.')
        color_mode = cv2.IMREAD_COLOR

    if size and isinstance(size, int):
        size = (size, size)

    images = []
    labels = []
    path = []
    if load_lbl:
        for ip, lp in zip(img_path, lbl_path):
            img = cv2.imread(ip, color_mode)
            lbl = cv2.imread(lp, cv2.IMREAD_COLOR)
            if (size and resize_type == 'cut' and img.shape[0] > size[0]
                    and img.shape[1] > size[1]):
                top = (img.shape[0] - size[0]) // 2
                left = (img.shape[1] - size[1]) // 2
                img = img[top:(top + size[0]),
                          left:(left + size[1])]
                lbl = lbl[top:(top + size[0]),
                          left:(left + size[1])]
            if (size and img.shape[0] != size[0]
                    and img.shape[1] != size[1]):
                img = cv2.resize(img, size)
                lbl = cv2.resize(lbl, size)
            images.append(img)
            labels.append(lbl)
        images = np.array(images, dtype=dtype)
        labels = np.array(labels, dtype=dtype)

    else:
        for ip in img_path:
            img = cv2.imread(ip, color_mode)
            if (size and resize_type == 'cut'
                    and img.shape[0] > size[0]
                    and img.shape[1] > size[1]):
                top = (img.shape[0] - size[0]) // 2
                left = (img.shape[1] - size[1]) // 2
                img = img[top:(top + size[0]),
                      left:(left + size[1])]
            if (size and img.shape[0] != size[0]
                    and img.shape[1] != size[1]):
                img = cv2.resize(img, size)
            images.append(img)
        images = np.array(images, dtype=dtype)

    if color_mode == cv2.IMREAD_COLOR:
        images = images[:, :, :, ::-1]
    else:
        images = images[:, :, :, np.newaxis]

    return images, labels, path