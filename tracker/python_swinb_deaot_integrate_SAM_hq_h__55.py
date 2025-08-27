
from numpy.lib.type_check import imag
import torch
import torch.nn.functional as F
import os
import sys
import time
import cv2
import importlib
import numpy as np
import math
import random
import collections
import glob
from vot.region.io import read_trajectory
from helpers import draw_mask, save_prediction
from segment_anything_hq import sam_model_registry, SamPredictor
from kalman import KalmanFilter

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])

DIR_PATH = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(DIR_PATH)
import vot_utils
from tools.transfer_predicted_mask2vottype import transfer_mask


AOT_PATH = os.path.join(os.path.dirname(__file__), '../dmaot')
sys.path.append(AOT_PATH)


#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import dmaot.dataloaders.video_transforms as tr
from torchvision import transforms
from dmaot.networks.engines import build_engine
from dmaot.utils.checkpoint import load_network
from dmaot.networks.models import build_vos_model

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(1000000007)
torch.set_num_threads(4)
torch.autograd.set_grad_enabled(False)

class AOTTracker(object):
    def __init__(self, cfg, gpu_id):
        self.with_crop = False
        self.EXPAND_SCALE = None
        self.small_ratio = 12
        self.mid_ratio = 100
        self.large_ratio = 0.5
        self.AOT_INPUT_SIZE = (465, 465)
        self.gpu_id = gpu_id
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
        print('cfg.TEST_CKPT_PATH = ', cfg.TEST_CKPT_PATH)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        self.engine = build_engine(cfg.MODEL_ENGINE,
                                   phase='eval',
                                   aot_model=self.model,
                                   gpu_id=gpu_id,
                                   short_term_mem_skip=cfg.TEST_SHORT_TERM_MEM_GAP,
                                   long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)

        self.transform = transforms.Compose([
        tr.MultiRestrictSize(cfg.TEST_MAX_SHORT_EDGE,
                                cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP,
                                cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
        tr.MultiToTensor()
        ])  
        self.model.eval()

    def add_first_frame(self, frame, mask, object_num): 

        sample = {
            'current_img': frame,
            'current_label': mask,
        }
        sample['meta'] = {
            'obj_num': object_num,
            'height':frame.shape[0],
            'width':frame.shape[1],
        }
        sample = self.transform(sample)
        
        frame = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
        mask = sample[0]['current_label'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)

        mask = F.interpolate(mask, size=frame.size()[2:], mode="nearest")

       
        # add reference frame
        self.engine.add_reference_frame(frame, mask, frame_step=0, obj_nums=object_num)

    
    def track(self, image):
        
        height = image.shape[0]
        width = image.shape[1]
        
        sample = {'current_img': image}
        sample['meta'] = {
            'height': height,
            'width': width,
        }
        sample = self.transform(sample)
        output_height = sample[0]['meta']['height']
        output_width = sample[0]['meta']['width']
        image = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)

        self.engine.match_propogate_one_frame(image)
        pred_logit = self.engine.decode_current_logits((output_height, output_width))
        pred_prob = torch.softmax(pred_logit, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1,
                                    keepdim=True).float()

        _pred_label = F.interpolate(pred_label,
                                    size=self.engine.input_size_2d,
                                    mode="nearest")
        self.engine.update_memory(_pred_label)

        mask = pred_label.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        return mask


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

#####################
# config
#####################

'''config = {
    'exp_name': 'default',
    'model': 'swinb_dm_deaotl',
    'pretrain_model_path': 'pretrained_models/SwinB_DeAOTL_PRE_YTB_DAV_VIP_MOSE_OVIS_LASOT_GOT.pth',
    'config': 'pre_ytb_dav',
    'long_max': 10,
    'long_gap': 30,
    'short_gap': 2,
    'patch_wised_drop_memories': False,
    'patch_max': 999999,
    'gpu_id': 0,
}'''

config = {
    'exp_name': 'default',
    'model': 'swinb_deaotl',
    'pretrain_model_path': 'pretrained_models/SwinB_DeAOTL_PRE_YTB_DAV_VIP_MOSE_OVIS_LASOT_GOT.pth',
    'config': 'pre_ytb_dav',
    'long_max': 10,
    'long_gap': 30,
    'short_gap': 2,
    'patch_wised_drop_memories': False,
    'patch_max': 999999,
    'gpu_id': 0,
}


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


def extract_bounding_box(mask):
    indices = np.where(mask > 0)

    x_min, y_min = np.min(indices[1]), np.min(indices[0])
    x_max, y_max = np.max(indices[1]), np.max(indices[0])
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return [[x_min, y_min, x_max, y_max]]


# sam_checkpoint = "/usr/mvl2/esdft/sam_vit_b_01ec64.pth"
# model_type = "vit_b"
# sam_checkpoint = "/usr/mvl2/esdft/sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# sam_checkpoint = "/usr/mvl2/esdft/sam_vit_l_0b3195.pth"
# model_type = "vit_l"
sam_checkpoint = "/home/esdft/data/sam_hq_vit_h.pth"
model_type = "vit_h"


device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


# set cfg
engine_config = importlib.import_module('configs.' + f'{config["config"]}')
cfg = engine_config.EngineConfig(config['exp_name'], config['model'])
cfg.TEST_CKPT_PATH = os.path.join(DIR_PATH, config['pretrain_model_path'])
cfg.TEST_LONG_TERM_MEM_MAX = config['long_max']
cfg.TEST_LONG_TERM_MEM_GAP = config['long_gap']
cfg.TEST_SHORT_TERM_MEM_GAP = config['short_gap']
cfg.PATCH_TEST_LONG_TERM_MEM_MAX = config['patch_max']
cfg.PATCH_WISED_DROP_MEMORIES = True if config['patch_wised_drop_memories'] else False

### init trackers
tracker = AOTTracker(cfg, config["gpu_id"])

# initialize tracker
tracker.add_first_frame(image, merged_mask, object_num)
mask_size = merged_mask.shape


while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    start_time = time.time()

    image = read_img(imagefile)
    predictor.set_image(image)
    labels = tracker.track(image)
    labels = F.interpolate(torch.tensor(labels)[None, None, :, :], size=mask_size, mode="nearest").numpy().astype(np.uint8)[0][0]

    predicted_masks = []
    for object_id in range(1, object_num + 1):
        m = (labels == object_id).astype(np.uint8)
        if m.sum() == 0:
          predicted_masks.append(m)
          continue

        bbox = extract_bounding_box(m)

        # handling the no bbox case
        input_boxes = torch.tensor(bbox, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

        masks, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=True,
        )
        new_m, max_iou = m, 0.55
        for i in range(len(masks)):
            m1 = masks[i]
            score = scores[i]
            max_index = torch.argmax(score).item()
            pred_mask = (m1[max_index, :, :].cpu().numpy().astype(np.uint8))
            #pred_mask = (mask[0, :, :].cpu().numpy().astype(np.uint8))
            iou = (m * pred_mask + 1e-6).sum() / (np.maximum(m, pred_mask).sum() + 1e-6)
            if iou > max_iou:
                new_m = pred_mask
                max_iou = iou


        predicted_masks.append(new_m)

    end_time = time.time()

    elapsed_time = end_time - start_time

    with open('/cluster/VAST/civalab/results/VOTS_2024/my_method_timing.txt', 'a') as f:
        f.write(f'{imagefile}\t{elapsed_time}\n')

    handle.report(predicted_masks)
