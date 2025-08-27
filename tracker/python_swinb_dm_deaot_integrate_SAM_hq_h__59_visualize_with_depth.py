
from numpy.lib.type_check import imag
import torch
import torch.nn.functional as F
import os
import sys
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
import threading
from PIL import Image

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

config = {
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
}


# DATASET = '/cluster/VAST/civalab/results/VOTS_2024/sequences'

DATASET='/cluster/VAST/civalab/results/development_data_2024/sequences/'
seq_name = sys.argv[1]
seq = os.path.join(DATASET, seq_name)

out_dir = '/cluster/VAST/civalab/results/elel/'

if not os.path.exists(f'{out_dir}/{seq_name}'):
    os.makedirs(f'{out_dir}/{seq_name}')

imgs_paths = sorted(glob.glob(os.path.join(seq, 'color/*.jpg')))
gt_paths = sorted(glob.glob(os.path.join(seq, 'groundtruth*.txt')))
print('gt_paths', gt_paths)

first_img = cv2.imread(imgs_paths[0])
initial_mask = np.zeros(first_img.shape[:2], dtype=np.uint8)
tracks = {}

for i, gt_path in enumerate(gt_paths, 1):

    curr_obj = read_trajectory(gt_path)[0]
    bounds = curr_obj.bounds()
    x1, y1, x2, y2 = bounds
    initial_mask[y1:y2+1, x1:x2+1] = curr_obj.mask * i
    tracks[i] = []


object_num = len(tracks)
merged_mask = initial_mask
image = first_img



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

_palette = [
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128,
    128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0,
    128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128, 64, 0, 0,
    191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22, 22, 22, 23, 23, 23, 24,
    24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30,
    30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36,
    37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43,
    43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49,
    49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55,
    56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62,
    62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68,
    68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74,
    75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81,
    81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87,
    87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93,
    94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99,
    100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104,
    105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109,
    110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114,
    115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119,
    120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124,
    125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129,
    130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134,
    135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139,
    140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144,
    145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149,
    150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154,
    155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159,
    160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164,
    165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169,
    170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174,
    175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179,
    180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184,
    185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189,
    190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194,
    195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199,
    200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204,
    205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209,
    210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214,
    215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219,
    220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224,
    225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229,
    230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234,
    235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239,
    240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244,
    245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249,
    250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254,
    255, 255, 255
]


def _save_mask(mask, path, squeeze_idx=None):
    if squeeze_idx is not None:
        unsqueezed_mask = mask * 0
        for idx in range(1, len(squeeze_idx)):
            obj_id = squeeze_idx[idx]
            mask_i = mask == idx
            unsqueezed_mask += (mask_i * obj_id).astype(np.uint8)
        mask = unsqueezed_mask
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(_palette)
    mask.save(path)



def save_mask(mask_tensor, path, squeeze_idx=None):
    # mask = mask_tensor.cpu().numpy().astype('uint8')
    mask = mask_tensor.astype('uint8')
    threading.Thread(target=_save_mask, args=[mask, path, squeeze_idx]).start()
    # _save_mask(mask, path, squeeze_idx)



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

#######################################################################################
DEPTH_PATH = os.path.join('/cluster/VAST/civalab/results/elham_results/Depth-Anything-V2')
sys.path.append(DEPTH_PATH)

MASK_IMPROVEMENT_PATH = os.path.join('/cluster/VAST/civalab/users/grzc7/HQSMem++/')
sys.path.append(MASK_IMPROVEMENT_PATH)


from net.FuseNet import FuseNetSwin
import segmentation_models_pytorch as smp
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from depth_anything_v2.dpt import DepthAnythingV2

# network input size
imgWidth = 480
imgHeight = 320

numClass = 10

# pretrained se-resnet50 on ImageNet
model_class = smp.Unet(encoder_name='mit_b5', encoder_weights='imagenet', classes=numClass, activation=None)
encoder = model_class.encoder

model = FuseNetSwin(numClass, encoder).to(device)


# load trained model
modelName = 'FuseNet_SwinEncoder'
model.load_state_dict(torch.load('/cluster/VAST/civalab/users/grzc7/HQSMem++/models/' + modelName + '.pt'))
model.eval()


depth_model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

input_size = 518

depth_anything = DepthAnythingV2(**depth_model_configs[encoder])
depth_anything.load_state_dict(torch.load(f'/cluster/VAST/civalab/results/elham_results/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()


def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

common_transforms = get_transforms(img_size=(imgHeight, imgWidth))
#######################################################################################


for img_path in imgs_paths:
    imagefile = img_path
    if not imagefile:
        break
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
        new_m, max_iou = m, 0.59
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

    pred_label = np.stack([np.zeros_like(predicted_masks[0])] + predicted_masks).argmax(0)
    pred_label = torch.tensor(pred_label).cuda().float().unsqueeze(0).unsqueeze(0)
    _pred_label = F.interpolate(pred_label,
                                size=image.shape[:2],
                                mode="nearest").squeeze(0).squeeze(0).cpu().numpy()



    depth = depth_anything.infer_image(image)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

    depth_image = Image.fromarray(depth)
    pred = Image.fromarray(_pred_label)
    frame = Image.fromarray(image)

    inputs = common_transforms(frame)
    inputs2 = common_transforms(pred)
    inputs3 = common_transforms(depth_image)

    inputs = inputs.to(device)
    inputs2 = inputs2.to(device)
    inputs3 = inputs3.to(device)

    inputs = inputs.float()
    inputs2 = inputs2.float()
    inputs3 = inputs3.float()

    inputs = inputs.unsqueeze(0)
    inputs2 = inputs2.unsqueeze(0)
    inputs3 = inputs3.unsqueeze(0)

    pred = model(inputs3, inputs2, inputs)
    pred = F.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    pred_labels = np.argmax(pred[0], axis=0)
    pred_resized = cv2.resize(pred_labels, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    print(np.unique(pred_resized), object_num)

    _masks = transfer_mask(pred_resized, object_num)

    print(_masks[0].shape)

    name = f"{os.path.basename(img_path).split('.')[0]}.png"
    save_mask(_pred_label, f'{out_dir}/{seq_name}/{name}', squeeze_idx=None)
