# Adapted from https://github.com/Xylon-Sean/rfnet/blob/master/example.py


import cv2
import torch
import random
import argparse
import numpy as np
from threading import RLock

import config
config.cfg.set_lib('rfnet', prepend = True)

from rfnet.utils.common_utils import gct
#from rfnet.utils.eval_utils import nearest_neighbor_distance_ratio_match
from rfnet.model.rf_des import HardNetNeiMask
from rfnet.model.rf_det_so import RFDetSO
from rfnet.model.rf_net_so import RFNetSO
from rfnet.config import cfg

from utils_tf import set_tf_logging
from utils_img import img_from_floats
from utils_sys import Printer, print_options

kVerbose = True

# interface for pySLAM
class RfNetFeature2D:
    def __init__(self, do_cuda=True):
        self.lock = RLock()

        print('Using RfNetFeature2D')

        print(f"{gct()} : start time")

        random.seed(cfg.PROJ.SEED)
        torch.manual_seed(cfg.PROJ.SEED)
        np.random.seed(cfg.PROJ.SEED)

        print(f"{gct()} : model init")
        det = RFDetSO(
            cfg.TRAIN.score_com_strength,
            cfg.TRAIN.scale_com_strength,
            cfg.TRAIN.NMS_THRESH,
            cfg.TRAIN.NMS_KSIZE,
            cfg.TRAIN.TOPK,
            cfg.MODEL.GAUSSIAN_KSIZE,
            cfg.MODEL.GAUSSIAN_SIGMA,
            cfg.MODEL.KSIZE,
            cfg.MODEL.padding,
            cfg.MODEL.dilation,
            cfg.MODEL.scale_list,
        )
        des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
        model = RFNetSO(
            det, des, cfg.LOSS.SCORE, cfg.LOSS.PAIR, cfg.PATCH.SIZE, cfg.TRAIN.TOPK
        )

        print(f"{gct()} : to device")

        use_cuda = torch.cuda.is_available() & do_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = model.to(self.device)
        resume = config.cfg.root_folder + '/thirdparty/rfnet/runs/10_24_09_25/model/e121_NN_0.480_NNT_0.655_NNDR_0.813_MeanMS_0.649.pth.tar'
        print(f"{gct()} : in {resume}")

        if use_cuda:
            checkpoint = torch.load(resume)
        else:
            checkpoint = torch.load(resume, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["state_dict"])


    def detectAndCompute(self, frame, mask=None):  # mask is a fake input
        with self.lock:
            self.frame = frame
            #self.kps, self.des = self.compute_kps_des(frame)
            kps, des, img = self.model.detectAndCompute(frame, self.device, (frame.shape[0], frame.shape[1]))
            self.kps = list(map(self.to_cv2_kp, kps.numpy()))
            self.des = des.detach().numpy()
            #self.kps = to_cv2_kp

            if kVerbose:
                print('detector: RFNET , descriptor: LFNET , #features: ', len(self.kps), ', frame res: ',
                      frame.shape[0:2])
            return self.kps, self.des

    # return keypoints if available otherwise call detectAndCompute()
    def detect(self, frame, mask=None):  # mask is a fake input
        with self.lock:
            if self.frame is not frame:
                self.detectAndCompute(frame)
            return self.kps

    # return descriptors if available otherwise call detectAndCompute()
    def compute(self, frame, kps=None, mask=None):  # kps is a fake input, mask is a fake input
        with self.lock:
            if self.frame is not frame:
                Printer.orange('WARNING: RFNET  is recomputing both kps and des on last input frame', frame.shape)
                self.detectAndCompute(frame)
            return self.kps, self.des

    def to_cv2_kp(self, kp):
        # kp is like [batch_idx, y, x, channel]

        return cv2.KeyPoint(float(kp[2]), float(kp[1]), 0)