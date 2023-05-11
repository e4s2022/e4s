import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
import random
import torch
import torchvision.transforms.functional as TF

from src.datasets.utils import make_dataset


# 19 attributes in total, skin-1,nose-2,...cloth-18, background-0
celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                        'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                        'neck_l', 'neck', 'cloth']

# face-parsing.PyTorch also includes 19 attributes，but with different permutation
face_parsing_PyTorch_label_list = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
                                    'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 
                                    'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 
                                    'cloth', 'hair', 'hat']  # skin-1 l_brow-2 ...
 
# 9 attributes with left-right aggrigation
faceParser_label_list = ['background', 'mouth', 'eyebrows', 'eyes', 'hair', 
                         'nose', 'skin', 'ears', 'belowface']

# 12 attributes with left-right aggrigation
faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
                                  'nose', 'skin', 'ears', 'belowface', 'mouth', 
                                  'eye_glass', 'ear_rings']

TO_TENSOR = transforms.ToTensor()
MASK_CONVERT_TF = transforms.Lambda(
    lambda celebAHQ_mask: __celebAHQ_masks_to_faceParser_mask(celebAHQ_mask))

MASK_CONVERT_TF_DETAILED = transforms.Lambda(
    lambda celebAHQ_mask: __celebAHQ_masks_to_faceParser_mask_detailed(celebAHQ_mask))

FFHQ_MASK_CONVERT_TF = transforms.Lambda(
    lambda mask: __ffhq_masks_to_faceParser_mask(mask))

FFHQ_MASK_CONVERT_TF_DETAILED = transforms.Lambda(
    lambda mask: __ffhq_masks_to_faceParser_mask_detailed(mask))

NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def get_transforms(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __ffhq_masks_to_faceParser_mask_detailed(mask):
    """Convert the esitimated semantic image by face-parsing.PyTorch to reduced categories (12-class). 

    Args:
        mask (PIL image): with shape [H,W]
    Return:
        aggrigated mask, with same shape [H,W] but the number of segmentation classes is less
    """

    converted_mask = np.zeros_like(mask)

    backgorund = np.equal(mask, 0)
    converted_mask[backgorund] = 0

    lip = np.logical_or(np.equal(mask, 12), np.equal(mask, 13)) 
    converted_mask[lip] = 1

    eyebrows = np.logical_or(np.equal(mask, 2),
                             np.equal(mask, 3))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(mask, 4), np.equal(mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(mask, 17)
    converted_mask[hair] = 4

    nose = np.equal(mask, 10)
    converted_mask[nose] = 5

    skin = np.equal(mask, 1)
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(mask, 7), np.equal(mask, 8))
    converted_mask[ears] = 7

    belowface = np.equal(mask, 14)
    converted_mask[belowface] = 8
    
    mouth = np.equal(mask, 11)   
    converted_mask[mouth] = 9

    eye_glass = np.equal(mask, 6)
    converted_mask[eye_glass] = 10
    
    ear_rings = np.equal(mask, 9)
    converted_mask[ear_rings] = 11

    return converted_mask

def __ffhq_masks_to_faceParser_mask(mask):
    """Convert the esitimated semantic image by face-parsing.PyTorch to reduced categories (9-class). 

    Args:
        mask (PIL image): with shape [H,W]
    Return:
        aggrigated mask, with same shape [H,W] but the number of segmentation classes is less
    """
    converted_mask = np.zeros_like(mask)

    backgorund = np.equal(mask, 0)
    converted_mask[backgorund] = 0

    mouth = np.logical_or(
        np.logical_or(np.equal(mask, 11), np.equal(mask, 12)),
        np.equal(mask, 13)
    )
    converted_mask[mouth] = 1

    eyebrows = np.logical_or(np.equal(mask, 2),
                             np.equal(mask, 3))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(mask, 4), np.equal(mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(mask, 17)
    converted_mask[hair] = 4

    nose = np.equal(mask, 10)
    converted_mask[nose] = 5

    skin = np.equal(mask, 1)
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(mask, 7), np.equal(mask, 8))
    converted_mask[ears] = 7

    belowface = np.equal(mask, 14)
    converted_mask[belowface] = 8

    return converted_mask

def __celebAHQ_masks_to_faceParser_mask_detailed(celebA_mask):
    """Convert the semantic image of CelebAMaskHQ to reduced categories (12-class). 

    Args:
        mask (PIL image): with shape [H,W]
    Return:
        aggrigated mask, with same shape [H,W] but the number of segmentation classes is less
    """
    # 19 attributes in total, skin-1,nose-2,...cloth-18, background-0
    celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                            'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                            'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                            'neck_l', 'neck', 'cloth']# 12 attributes with left-right aggrigation
    faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
                                    'nose', 'skin', 'ears', 'belowface', 'mouth', 
                                  'eye_glass', 'ear_rings']

    converted_mask = np.zeros_like(celebA_mask)

    backgorund = np.equal(celebA_mask, 0)
    converted_mask[backgorund] = 0

    lip = np.logical_or(np.equal(celebA_mask, 11), np.equal(celebA_mask, 12))
    converted_mask[lip] = 1

    eyebrows = np.logical_or(np.equal(celebA_mask, 6),
                             np.equal(celebA_mask, 7))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(celebA_mask, 4), np.equal(celebA_mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(celebA_mask, 13)
    converted_mask[hair] = 4

    nose = np.equal(celebA_mask, 2)
    converted_mask[nose] = 5

    skin = np.equal(celebA_mask, 1)
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(celebA_mask, 8), np.equal(celebA_mask, 9))
    converted_mask[ears] = 7

    belowface = np.equal(celebA_mask, 17)
    converted_mask[belowface] = 8
    
    mouth = np.equal(celebA_mask, 10)   
    converted_mask[mouth] = 9

    eye_glass = np.equal(celebA_mask, 3)
    converted_mask[eye_glass] = 10
    
    ear_rings = np.equal(celebA_mask, 15)
    converted_mask[ear_rings] = 11
    
    return converted_mask

def __celebAHQ_masks_to_faceParser_mask(celebA_mask):
    """Convert the semantic image of CelebAMaskHQ to reduced categories (9-class). 

    Args:
        mask (PIL image): with shape [H,W]
    Return:
        aggrigated mask, with same shape [H,W] but the number of segmentation classes is less
    """

    assert len(celebA_mask.size) == 2, "The provided mask should be with [H,W] format"

    converted_mask = np.zeros_like(celebA_mask)

    backgorund = np.equal(celebA_mask, 0)
    converted_mask[backgorund] = 0

    mouth = np.logical_or(
        np.logical_or(np.equal(celebA_mask, 10), np.equal(celebA_mask, 11)),
        np.equal(celebA_mask, 12)
    )
    converted_mask[mouth] = 1

    eyebrows = np.logical_or(np.equal(celebA_mask, 6),
                             np.equal(celebA_mask, 7))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(celebA_mask, 4), np.equal(celebA_mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(celebA_mask, 13)
    converted_mask[hair] = 4

    nose = np.equal(celebA_mask, 2)
    converted_mask[nose] = 5

    skin = np.equal(celebA_mask, 1)
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(celebA_mask, 8), np.equal(celebA_mask, 9))
    converted_mask[ears] = 7

    belowface = np.equal(celebA_mask, 17)
    converted_mask[belowface] = 8

    return converted_mask


class CelebAHQDataset(Dataset):
    def __init__(self, dataset_root, mode="test",
                 img_transform=TO_TENSOR, label_transform=TO_TENSOR,
                 load_vis_img=False,fraction=1.0,
                 flip_p=-1):  # negative number for no flipping

        self.mode = mode
        self.root = dataset_root
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.load_vis_img = load_vis_img
        self.fraction=fraction
        self.flip_p = flip_p

        if mode == "train":
            self.imgs = sorted([osp.join(self.root, "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000)])
            self.labels = sorted([osp.join(self.root, "CelebA-HQ-mask", "%d.png"%idx) for idx in range(28000)])
            self.labels_vis =  sorted([osp.join(self.root, "vis", "%d.png"%idx) for idx in range(28000)]) if self.load_vis_img else None
        else:
            self.imgs = sorted([osp.join(self.root, "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000, 30000)])
            self.labels = sorted([osp.join(self.root, "CelebA-HQ-mask", "%d.png"%idx) for idx in range(28000, 30000)])
            self.labels_vis =  sorted([osp.join(self.root, "vis", "%d.png"%idx) for idx in range(28000, 30000)]) if self.load_vis_img else None

        self.imgs= self.imgs[:int(len(self.imgs)*self.fraction)]
        self.labels= self.labels[:int(len(self.labels)*self.fraction)]
        self.labels_vis= self.labels_vis[:int(len(self.labels_vis)*self.fraction)]  if self.load_vis_img else None

        if self.load_vis_img:
            assert len(self.imgs) == len(self.labels) == len(self.labels_vis)
        else:
            assert len(self.imgs) == len(self.labels)

        # image pairs indices
        self.indices = np.arange(len(self.imgs))


    def __len__(self):
        return len(self.indices)

    def load_single_image(self, index):
        """Load one sample for training, inlcuding 
            - the image, 
            - the semantic image, 
            - the corresponding visualization image

        Args:
            index (int): index of the sample
        Return:
            img: RGB image
            label: seg mask
            label_vis: visualization of the seg mask
        """
        img = self.imgs[index]
        img = Image.open(img).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)

        label = self.labels[index]
        label = Image.open(label).convert('L')
        if self.label_transform is not None:
            label = self.label_transform(label)

        if self.load_vis_img:
            label_vis = self.labels_vis[index]
            label_vis = Image.open(label_vis).convert('RGB')
            label_vis = TO_TENSOR(label_vis)
        else:
            label_vis = -1  # unified interface
        return img, label, label_vis

    def __getitem__(self, idx):
        index = self.indices[idx]

        img, label, label_vis = self.load_single_image(index)
        
        if self.flip_p > 0:
            if random.random() < self.flip_p:
                img = TF.hflip(img)
                label = TF.hflip(label)
                
        return img, label, label_vis
    
        
class FFHQDataset(Dataset):
    def __init__(self, dataset_root,
                 img_transform=TO_TENSOR, label_transform=TO_TENSOR,
                 fraction=1.0,
                 load_raw_label=False,
                 flip_p = -1):

        self.root = dataset_root
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.fraction=fraction
        self.load_raw_label = load_raw_label
        self.flip_p = flip_p
        
        with open(osp.join(self.root,"images_1024", "ffhq_list.txt"),"r") as f:
            f_lines = f.readlines()
        
        self.imgs = sorted([osp.join(self.root, "images_1024", line.replace("\n","")) for line in f_lines])
        self.imgs = self.imgs[:int(len(self.imgs)*self.fraction)]
        self.labels = [img.replace("images_1024", "BiSeNet_mask") for img in self.imgs]
    
        assert len(self.imgs) == len(self.labels)
        
        self.indices = np.arange(len(self.imgs))

    def __len__(self):
        return len(self.indices)

    def load_single_image(self, index):
        """Load one sample for training, inlcuding 
            - the image, 
            - the semantic image, 
            - the corresponding visualization image

        Args:
            index (int): index of the sample
        Return:
            img: RGB image
            label: seg mask
            label_vis: visualization of the seg mask
        """
        img = self.imgs[index]
        img = Image.open(img).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)

        label = self.labels[index]
        label = Image.open(label).convert('L')
        
        if self.load_raw_label:
            original_label = TO_TENSOR(label)
        
        if self.label_transform is not None:
            label = self.label_transform(label)

        label_vis = -1  # unified interface
        
        if self.load_raw_label:
            return img, original_label, label, label_vis
        else:
            return img, label, label_vis
        
    def __getitem__(self, idx):
        index = self.indices[idx]

        img, label, label_vis = self.load_single_image(index)
        
        if self.flip_p > 0:
            if random.random() < self.flip_p:
                img = TF.hflip(img)
                label = TF.hflip(label)
        
        return img, label, label_vis    


if __name__ == '__main__':
    ds = CelebAHQDataset(dataset_root="/mnt/hdd8T/lza/py_projs/e4s/data/CelebAMask-HQ")
    sample = ds.__getitem__(25)
    print(-1)