import gradio as gr
import numpy as np


COMP = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
        'nose', 'skin', 'ears', 'belowface', 'mouth', 
        'eye_glass', 'ear_rings']

COMP2INDEX = {'background':0, 'lip':1, 'eyebrows':2, 'eyes':3, 'hair':4, 
            'nose':5, 'skin':6, 'ears':7, 'belowface':8, 'mouth':9, 
            'eye_glass':10, 'ear_rings':11}

COMP_COLORS = {
    0: '#000000',
    1: '#cc0000',
    2: '#4c9900',
    3: '#cccc00',
    4: '#3333ff',
    5: '#cc00cc',
    6: '#00ffff',
    7: '#ffcccc',
    8: '#663300',
    9: '#ff0000',
    10: '#66cc00',
    11: '#ffff00',
    12: '#000099',
    13: '#0000cc',
    14: '#ff3399',
    15: '#00cccc',
    16: '#003300',
    17: '#ff9933',
    18: '#00cc00',
}

COMP_COLORS_NUMPY = np.array(
    [[0,  0,  0],
    [204, 0,  0],
    [76, 153, 0],
    [204, 204, 0],##
    [51, 51, 255],##
    [204, 0, 204],##
    [0, 255, 255],##
    [255, 204, 204],##
    [102, 51, 0],##
    [255, 0, 0],##
    [102, 204, 0],##
    [255, 255, 0],##
    [0, 0, 153],##
    [0, 0, 204],##
    [255, 51, 153],##
    [0, 204, 204],##
    [0, 51, 0],##
    [255, 153, 51],
    [0, 204, 0],
    ]
)

def label_map_to_colored_mask(pred):

    num_labels=19
    
    h, w = np.shape(pred)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     print(color.shape)
    for ii in range(num_labels):
        #         print(ii)
        mask = pred == ii
        rgb[mask, None] = COMP_COLORS_NUMPY[ii, :]
    # Correct unk
    unk = pred == 255
    rgb[unk, None] = COMP_COLORS_NUMPY[0, :]

    return rgb

def colored_mask_to_label_map(colored_mask):
    num_labels=19
    h, w = np.shape(colored_mask)[:2]

    label_map = np.zeros((h, w), dtype=np.uint8)
    for i in range(num_labels):
        mm = (np.sum(colored_mask == COMP_COLORS_NUMPY[i], axis=-1) == 3)
        label_map[mm] = i
    
    return label_map