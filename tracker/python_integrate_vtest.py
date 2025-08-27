import os
import sys
import math
import random
import collections
import glob
import numpy as np
import cv2
from vot.region.io import read_trajectory


Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])

DIR_PATH = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(DIR_PATH)
import vot_utils


AOT_PATH = os.path.join(os.path.dirname(__file__), '../dmaot')
sys.path.append(AOT_PATH)


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)


def read_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def write_file():
    with open('/cluster/VAST/civalab/results/VOTS_2025/test.txt', 'a') as f:
        f.write('Hi')

write_file()

# get first frame and mask
handle = vot_utils.VOT("mask", multiobject=True)

objects = handle.objects()
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

# get first
image = read_img(imagefile)

# Get merged-mask
merged_mask = np.zeros((image.shape[0], image.shape[1]))
object_num = len(objects)
object_id = 1
for object in objects:
    mask = make_full_size(object, (image.shape[1], image.shape[0]))
    mask = np.where(mask > 0, object_id, 0)    
    merged_mask += mask
    object_id += 1

write_file()

def extract_bounding_box(mask):
    indices = np.where(mask > 0)

    x_min, y_min = np.min(indices[1]), np.min(indices[0])
    x_max, y_max = np.max(indices[1]), np.max(indices[0])
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return [[x_min, y_min, x_max, y_max]]




mask_size = merged_mask.shape


while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    image = read_img(imagefile)

    predicted_masks = []
    for object_id in range(1, object_num + 1):
        m = (np.zeros((image.shape[0], image.shape[1]))).astype(np.uint8)
        predicted_masks.append(m)

    handle.report(predicted_masks)
