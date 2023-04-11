import os

import numpy as np
import argparse
import scipy.ndimage
import PIL.Image
import face_alignment
from torch.autograd.grad_mode import enable_grad


def image_align(src_file, dst_file, face_landmarks, output_size=256, transform_size=1024, enable_padding=True):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    lm = np.array(face_landmarks)
    lm_chin          = lm[0  : 17, :2]  # left-right
    lm_eyebrow_left  = lm[17 : 22, :2]  # left-right
    lm_eyebrow_right = lm[22 : 27, :2]  # left-right
    lm_nose          = lm[27 : 31, :2]  # top-down
    lm_nostrils      = lm[31 : 36, :2]  # top-down
    lm_eye_left      = lm[36 : 42, :2]  # left-clockwise
    lm_eye_right     = lm[42 : 48, :2]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60, :2]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68, :2]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print('\nCannot find source image. Please run "--wilds" before "--align".')
        return
    img = PIL.Image.open(src_file)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    img.save(dst_file, 'PNG')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A simple script to extract eye and mouth coordinates from a face image.')
    parser.add_argument('-s', '--src', default='./raw_images', help='directory of raw images')
    parser.add_argument('-d', '--dst', default='./aligned_images', help='directory of aligned images')
    parser.add_argument('-o', '--output_size', default=256, type=int, help='size of aligned output (default: 256)')
    parser.add_argument('-t', '--transform_size', default=1024, type=int, help='size of aligned transform (default: 256)')
    parser.add_argument('--no_padding', action='store_false', help='no padding')

    args = parser.parse_args()

    if not os.path.exists(args.dst):
        os.mkdir(args.dst)

    landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

    for img_name in os.listdir(args.src):
        raw_img_path = os.path.join(args.src, img_name)

        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):

            aligned_face_path = os.path.join(args.dst, f'align-{img_name}')

            image_align(raw_img_path, aligned_face_path, face_landmarks, args.output_size, args.transform_size, args.no_padding)