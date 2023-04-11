
import numpy as np
import os
import json
import sys
import pprint
import random
import shutil
from PIL import Image
import glob
import copy
import torch
import cv2

# 19 attributes in total, skin-1,nose-2,...cloth-18, background-0
celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                        'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                        'neck_l', 'neck', 'cloth']

# face-parsing.PyTorch also includes 19 attributes，but with different permutation
FFHQ_label_list = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
                                    'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 
                                    'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 
                                    'cloth', 'hair', 'hat']  # skin-1 l_brow-2 ... 

# 12 attributes with left-right aggrigation
faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
                                  'nose', 'skin', 'ears', 'belowface', 'mouth', 
                                  'eye_glass', 'ear_rings']


def swap_head_mask_revisit_considerGlass(source, target, hair_first=True):
    res = np.zeros_like(target)
    
    
    target_regions = [np.equal(target, i) for i in range(12)]
    source_regions = [np.equal(source, i) for i in range(12)]
    
    
    # the background, neck, ear and earrings regions of target
    res[target_regions[0]] = 99  # a place-holder magic number 
    res[target_regions[8]] = 8 # neck
    res[target_regions[7]] = 7  # ear 
    res[target_regions[11]] = 11 # ear_rings

    if hair_first:
        res[target_regions[4]] = 4  # hair
    
    # the inner-face of source
    res[np.logical_and(source_regions[1], np.not_equal(res,99))] = 1 # lip
    res[np.logical_and(source_regions[2], np.not_equal(res,99))] = 2 # eyebrows
    res[np.logical_and(source_regions[3], np.not_equal(res,99))] = 3 # eyes
    res[np.logical_and(source_regions[5], np.not_equal(res,99))] = 5 # nose
    res[np.logical_and(source_regions[6], np.not_equal(res,99))] = 6  # skin
    res[np.logical_and(source_regions[9], np.not_equal(res,99))] = 9 # mouth
    
    # res[source_regions[1]] = 1 # lip
    # res[source_regions[2]] = 2 # eyebrows
    # res[source_regions[3]] = 3 # eyes
    # res[source_regions[5]] = 5 # nose
    # res[source_regions[6]] = 6  # skin
    # res[source_regions[9]] = 9 # mouth
    
    
    if not hair_first:
        res[target_regions[4]] = 4  # hair
    
    # the eye_glass of target
    res[target_regions[10]] = 10 # eye_glass
    
    
    # the missing pixels, fill in skin
    if np.sum(res==0) != 0:
        hole_map = 255*(res==0)
        res[res==0] = 6
    else:
        hole_map = np.zeros_like(res)
        
    # restore the background
    res[res==99] = 0
     
    return res, hole_map
