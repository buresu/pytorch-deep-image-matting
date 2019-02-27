import cv2
import numpy as np
from scipy import ndimage
import argparse
import glob
import os
import csv
import random

def create_trimap(gray, size):
    _trimap = np.copy(gray)
    rows, cols = gray.shape[:2]
    for r in range(rows):
        for c in range(cols):
            p = int(gray[r][c])
            if 0 < p and p < 255:
                cv2.circle(_trimap, (c, r), size, 127, -1)
    return _trimap

def composite_image(src, dst):

    _src = ndimage.rotate(src, random.randint(0, 359))

    dh, dw = dst.shape[:2]
    sh, sw = _src.shape[:2]

    _dst = np.zeros((dh, dw, 3), np.uint8)

    h_fit = (dw / dh) > (sw / sh)
    cw = sw if h_fit else int(sh * (dw / dh)) 
    ch = sh if not h_fit else int(sw * (dh / dw))

    shift_w = random.randint(0, sw - cw)
    shift_h = random.randint(0, sh - ch)

    _src = _src[shift_h:shift_h+ch,shift_w:shift_w+cw,:]

    _src = cv2.resize(_src, (dw, dh), interpolation=cv2.INTER_LANCZOS4)
    gray = _src[:,:,3]

    circle_size = random.randint(3, 8)

    alpha = gray.astype(float) / 255.0
    for i in range(3):
        _dst[:,:,i] = alpha[:,:] * _src[:,:,i] + (1.0 - alpha[:,:]) * dst[:,:,i]

    return _dst, gray, create_trimap(gray, circle_size)

def create_dataset(foreground_dir, background_dir, output_dir):

    image_dir = os.path.join(output_dir, 'image')
    alpha_dir = os.path.join(output_dir, 'alpha')
    trimap_dir = os.path.join(output_dir, 'trimap')

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(alpha_dir, exist_ok=True)
    os.makedirs(trimap_dir, exist_ok=True)

    foregrounds = glob.glob(os.path.join(foreground_dir, '*.png'))
    backgrounds = glob.glob(os.path.join(background_dir, '*.jpg'))

    f = open(os.path.join(output_dir, 'data.csv'), 'w')
    writer = csv.writer(f, lineterminator='\n')
    i = 0

    for front in foregrounds:
        src = cv2.imread(front, cv2.IMREAD_UNCHANGED)
        if not src is None:
            for back in random.sample(backgrounds, min(100, len(backgrounds))):
                dst = cv2.imread(back, cv2.IMREAD_COLOR)
                if not dst is None:
                    i += 1
                    name = 'image%06d.png' % i
                    print('create %s src:%s dst:%s' % (name, front, back))
                    image, alpha, trimap = composite_image(src, dst)
                    cv2.imwrite(os.path.join(image_dir, name), image)
                    cv2.imwrite(os.path.join(alpha_dir, name), alpha)
                    cv2.imwrite(os.path.join(trimap_dir, name), trimap)
                    writer.writerow(['image/%s' % name, 'alpha/%s' % name, 'trimap/%s' % name])

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--foreground_dir', type=str, default='foreground')
    parser.add_argument('--background_dir', type=str, default='background')
    parser.add_argument('--output_dir', type=str, default='dataset')
    args = parser.parse_args()
    create_dataset(args.foreground_dir, args.background_dir, args.output_dir)
