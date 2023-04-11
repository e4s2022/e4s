# import cv2
import torch
import numpy as np

from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib.pyplot as plt

def saveTensorToFile(tensor, save_path):
    """ save PyTorch Tensor to file

    Args:
        tensor (torch.Tensor): [C,H,W], 假设为范围为[0,1]
        save_path (str): save location
    """

    C, H, W = tensor.size()
    arr = np.transpose((tensor.numpy()*255).astype(np.uint8), (1, 2, 0))

    if C == 1:
        arr = np.squeeze(arr, axis=-1)
        Image.fromarray(arr).save(save_path)
        # cv2.imwrite(save_path, arr)
    else:
        Image.fromarray(arr).save(save_path)
        # cv2.imwrite(save_path, arr[:, :, ::-1])  # RGB to BGR, cv2需要BGR的图片


def interpolate(img, size):
    if type(size) == tuple:
        assert size[0] == size[1]
        size = size[0]

    orig_size = img.size(3)
    if size < orig_size:
        mode = 'area'
    else:
        mode = 'bilinear'
    return F.interpolate(img, (size, size), mode=mode)


def readImgAsTensor(img_path, gray=False, to_tensor=True, size=1024):
    """read image to  tensor within [0,1] range"""
    if not gray:
        img = Image.open(img_path).convert('RGB')
    else:
        img = Image.open(img_path).convert('L')

    img = TF.to_tensor(img)
    img = img.unsqueeze(0)
    if img.size(-1) != size:
        img = interpolate(img, size)
    return img


def featMap2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = (var - var.min())/(var.max()-var.min())
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)  # [-1,1]  --> [0,1]
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def tensor2map(var,shown_mask_indices=None):
    mask = np.argmax(var.data.cpu().numpy(), axis=0)
    colors = get_colors()
    mask_image = np.ones(shape=(mask.shape[0], mask.shape[1], 3))
    if shown_mask_indices is None:
        for class_idx in np.unique(mask):
            mask_image[mask == class_idx] = colors[class_idx]
    else:
        assert isinstance(shown_mask_indices, list)
        for class_idx in shown_mask_indices:
            mask_image[mask == class_idx] = colors[class_idx]
    mask_image = mask_image.astype('uint8')
    return Image.fromarray(mask_image)

# Visualization utils

def vis_mask_in_color(mask):
    colors = get_colors()
    mask_image = np.ones(shape=(mask.shape[0], mask.shape[1], 3))
    
    for class_idx in np.unique(mask):
        mask_image[mask == class_idx] = colors[class_idx]
   
    mask_image = mask_image.astype('uint8')
    return mask_image



def get_colors():
    # currently support up to 19 classes (for the celebs-hq-mask dataset)
    colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
              [255, 204, 204], [102, 51, 0], [255, 0, 0], [
                  102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
              [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    return colors


def vis_faces(log_hooks1):
    display_count = len(log_hooks1)
    fig = plt.figure(figsize=(6*2, 4 * display_count))
    gs = fig.add_gridspec(display_count, 3)
    for i in range(display_count):
        hooks_dict1 = log_hooks1[i]

        fig.add_subplot(gs[i, 0])
        vis_faces_no_id(hooks_dict1, fig, gs, i)

    plt.tight_layout()
    return fig


def vis_faces_no_id(hooks_dict1, fig, gs, i):
    plt.imshow(hooks_dict1['input_face'])
    plt.title('input_face1')

    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict1['input_mask'])
    plt.title('input_mask1')

    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict1['recon_styleCode'])
    plt.title('recon_styleCode1')
    
    # fig.add_subplot(gs[i, 3])
    # plt.imshow(hooks_dict1['recon_styleCode_feats'])
    # plt.title('recon_styleCode_feats1')
    
    # fig.add_subplot(gs[i, 3])
    # plt.imshow(hooks_dict2['input_face'])
    # plt.title('input_face2')

    # fig.add_subplot(gs[i, 4])
    # plt.imshow(hooks_dict2['input_mask'])
    # plt.title('input_mask2')

    # fig.add_subplot(gs[i, 5])
    # plt.imshow(hooks_dict2['recon_face'])
    # plt.title('recon_face2')


def aggregate_loss_dict(agg_loss_dict):
    mean_vals = {}
    for output in agg_loss_dict:
        for key in output:
            mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
    for key in mean_vals:
        if len(mean_vals[key]) > 0:
            mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
        else:
            print('{} has no value'.format(key))
            mean_vals[key] = 0
    return mean_vals


def labelMap2OneHot(label, num_cls):
    # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
    bs, _, H, W = label.size()
    onehot = torch.zeros(bs, num_cls, H, W, device=label.device)
    onehot = onehot.scatter_(1, label, 1.0)

    return onehot


def remove_module_prefix(state_dict,prefix):
    new_state_dict={}
    
    for k,v in state_dict.items():
        new_key=k.replace(prefix,"",1)
        new_state_dict[new_key]=v

    return new_state_dict

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
