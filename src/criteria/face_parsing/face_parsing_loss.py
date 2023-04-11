import torch
from torch import nn
import os
import time
import torch
import datetime
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms

import cv2
import PIL
from src.criteria.face_parsing.unet import unet
from src.criteria.face_parsing.utils import generate_label, generate_label_plain
from PIL import Image

class FaceParsingLoss(nn.Module):
    def __init__(self,opts):
        super(FaceParsingLoss, self).__init__()
        print('Loading Face Parsing Net')
        
        self.opts = opts
        self.face_pool = torch.nn.AdaptiveAvgPool2d((512, 512))
        
        self.G = unet()
        self.G.load_state_dict(torch.load(opts.face_parsing_model_path))
        self.G.eval()
        
        self.set_requires_grad(False)
            
    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag
    

    def inference(self, x):
        x = self.face_pool(x)  if x.shape[2]!=512 else  x # resize to 512 if needed
        labels_predict = self.G(x)
        
        labels_predict_plain = generate_label_plain(labels_predict,imsize=512)  # np.array [N,H,W]
        labels_predict_color = generate_label(labels_predict,imsize=512) # torch.Tensor [N,3,H,W]
        
        return labels_predict_plain, labels_predict_color
        
    def extract_feats(self, x):
        x = self.face_pool(x)  if x.shape[2]!=512 else  x # resize to 512 if needed
        x_feats = self.G.extract_feats(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats_ms = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats_ms = self.extract_feats(y_hat)
        y_feats_ms = [y_f.detach() for y_f in y_feats_ms]  # features in different levels
        
        loss_all = 0
        sim_improvement_all = 0
        
        for y_hat_feats, y_feats in zip(y_hat_feats_ms, y_feats_ms):
            loss = 0
            sim_improvement = 0
            count = 0
             
            for i in range(n_samples):
                sim_target = y_hat_feats[i].dot(y_feats[i])
                sim_views = y_feats[i].dot(y_feats[i])
                
                loss += 1 - sim_target  # id loss
                sim_improvement +=  float(sim_target) - float(sim_views)
                count += 1
            
            loss_all += loss / count
            sim_improvement_all += sim_improvement / count
    
        return loss_all, sim_improvement_all
