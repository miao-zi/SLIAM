import cv2
import numpy as np
import os
from PIL import Image
from skimage.feature import greycomatrix
from skimage import io
from skimage.feature import greycomatrix, greycoprops

def image_patch(image, window_size, height, width):
    img = image
    slide_window = window_size
    patch = np.zeros((window_size, window_size, height, width), dtype=np.uint8)

    for i in range(patch.shape[2]):
        for j in range(patch.shape[3]):
            patch[:, :, i, j] = img[i : i + window_size, j : j + window_size]

    return patch

def calculate_glrlm(img, vmin=0, vmax=255, nbit=64, window_size=5, step=[2], angle=[0]):

    vi, va = vmin, vmax
    height, width = img.shape

 
    bins = np.linspace(vi, va+1, nbit+1)
    img1 = np.digitize(img, bins) - 1


    img2 = cv2.copyMakeBorder(img1, int(window_size/2), int(window_size/2),
                              int(window_size/2), int(window_size/2), cv2.BORDER_REPLICATE)  # Image padding

    patch = np.zeros((window_size, window_size, height, width), dtype=np.uint8)
    patch = image_patch(img2, window_size, height, width)

    # Calculate GLRLM
    # greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
    glrlm = np.zeros((nbit, nbit, len(step), len(angle), height, width), dtype=np.uint8)
    for i in range(patch.shape[2]):
        for j in range(patch.shape[3]):
            glrlm[:, :, :, :, i, j] = greycomatrix(patch[:, :, i, j], step, angle, levels=nbit, symmetric=True, normed=True)

    return glrlm

if __name__ == '__main__':
    nbit = 64
    vi, va = 0, 255
    window_size = 7
    step = [2]
    angle = [0]
    path_images = '/home/miaoyu/MY/input_folder/images'
    path_entropy = '/home/miaoyu/MY/glrlm_entropy'
    img_list = os.listdir(path_images)
    for img_file in img_list:
        image_path = path_images + '/' + img_file
        img = np.array(Image.open(image_path))
        img = np.uint8(255.0 * (img - np.min(img)) / (np.max(img) - np.min(img)))
        height, width = img.shape

        glrlm = calculate_glrlm(img, vi, va, nbit, window_size, step, angle)

        for i in range(glrlm.shape[2]):
            for j in range(glrlm.shape[3]):
                glrlm_cut = np.zeros((nbit, nbit, height, width), dtype=np.float32)
                glrlm_cut = glrlm[:, :, i, j, :, :]
                entropy = np.zeros((height, width), dtype=np.float32)
                for m in range(height):
                    for n in range(width):
                        p = glrlm_cut[:, :, m, n]
                        p_normalized = p / p.sum()
                        entropy[m, n] = -np.sum(np.where(p_normalized > 0, p_normalized * np.log2(p_normalized), 0))
                np.save(path_entropy + '/' + img_file.split('.')[0] + '.txt', entropy)
