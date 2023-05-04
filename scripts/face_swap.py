import os
import copy
import cv2
from argparse import ArgumentParser
from PIL import Image
import torch
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torchvision.transforms as transforms
from torch.nn import functional as F
from skimage.transform import resize

os.sys.path.append('/apdcephfs_cq2/share_1290939/branchwang/projects/e4s')

from src.pretrained.face_vid2vid.driven_demo import init_facevid2vid_pretrained_model, drive_source_demo
from src.pretrained.gpen.gpen_demo import init_gpen_pretrained_model, GPEN_demo
from src.pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps
from src.utils.swap_face_mask import swap_head_mask_revisit_considerGlass

from src.utils import torch_utils
from src.utils.alignmengt import crop_faces, calc_alignment_coefficients
from src.utils.morphology import dilation, erosion
from src.utils.multi_band_blending import blending

from src.options.swap_options import SwapFacePipelineOptions
from src.models.networks import Net3
from src.datasets.dataset import TO_TENSOR, NORMALIZE, __celebAHQ_masks_to_faceParser_mask_detailed


def create_masks(mask, outer_dilation=0, operation='dilation'):
    radius = outer_dilation
    temp = copy.deepcopy(mask)
    if operation == 'dilation':
        full_mask = dilation(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        border_mask = full_mask - temp
    elif operation == 'erosion':
        full_mask = erosion(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        border_mask = temp - full_mask
    # 'expansion' means to obtain a boundary that expands to both sides
    elif operation == 'expansion':
        full_mask = dilation(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        erosion_mask = erosion(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        border_mask = full_mask - erosion_mask

    border_mask = border_mask.clip(0, 1)
    content_mask = mask
    
    return content_mask, border_mask, full_mask 

def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)

def logical_and_reduce(*tensors):
    return torch.stack(tensors, dim=0).all(dim=0)

def paste_image_mask(inverse_transform, image, dst_image, mask, radius=0, sigma=0.0):
    image_masked = image.copy().convert('RGBA')
    pasted_image = dst_image.copy().convert('RGBA')
    if radius != 0:
        mask_np = np.array(mask)
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        image_masked.putalpha(mask)

    projected = image_masked.transform(dst_image.size, Image.PERSPECTIVE, inverse_transform, Image.BILINEAR)
    pasted_image.alpha_composite(projected)
    return pasted_image

def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image


def smooth_face_boundry(image, dst_image, mask, radius=0, sigma=0.0):
    
    image_masked = image.copy().convert('RGBA')
    pasted_image = dst_image.copy().convert('RGBA')
    if radius != 0:
        mask_np = np.array(mask) 
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)  
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        image_masked.putalpha(mask)

    pasted_image.alpha_composite(image_masked)
    return pasted_image

# ===================================     
def crop_and_align_face(target_files):
    image_size = 1024
    scale = 1.0
    center_sigma = 0
    xy_sigma = 0
    use_fa = False
    
    print('Aligning images')
    crops, orig_images, quads = crop_faces(image_size, target_files, scale, center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    
    inv_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]
    
    return crops, orig_images, quads, inv_transforms
                    
def swap_comp_style_vector(style_vectors1, style_vectors2, comp_indices=[], belowFace_interpolation=False):
    """Replace the style_vectors1 with style_vectors2

    Args:
        style_vectors1 (Tensor): with shape [1,#comp,512], style vectors of target image 
        style_vectors2 (Tensor): with shape [1,#comp,512], style vectors of source image
    """
    assert comp_indices is not None
    
    style_vectors = copy.deepcopy(style_vectors1)
    
    for comp_idx in comp_indices:
        style_vectors[:,comp_idx,:] =  style_vectors2[:,comp_idx,:]
        
    # if no ear(7) region for source
    if torch.sum(style_vectors2[:,7,:]) == 0:
        style_vectors[:,7,:] = (style_vectors1[:,7,:] + style_vectors2[:,7,:]) / 2   
    
    # if no teeth(9) region for source
    if torch.sum(style_vectors2[:,9,:]) == 0:
        style_vectors[:,9,:] = style_vectors1[:,9,:] 
    
    # # use the ear_rings(11) of target
    # style_vectors[:,11,:] = style_vectors1[:,11,:] 
    
    # neck(8) interpolation
    if belowFace_interpolation:
        style_vectors[:,8,:] = (style_vectors1[:,8,:] + style_vectors2[:,8,:]) / 2
    
    return style_vectors
    

@torch.no_grad()
def faceSwapping_pipeline(source, target, opts, save_dir, target_mask=None, need_crop = False, verbose = False, only_target_crop=False):
    """
    The overall pipeline of face swapping:

        Input: source image, target image

        (1) Crop the faces from the source and target and align them, obtaining S and T ; (cropping is optional)
        (2) Use faceVid2Vid & GPEN to re-enact S, resulting in driven face D, and then parsing the mask of D
        (3) Extract the texture vectors of D and T using RGI
        (4) Texture and shape swapping between face D and face T        
        (5) Feed the swapped mask and texture vectors to the generator, obtaining swapped face I;
        (6) Stich I back to the target image

    Args:
        source (str): path to source
        target (str): path to target
        opts (): args
        save_dir (str): the location to save results
        target_mask (ndarray): 12-class segmap, will be estimated if not provided 
        need_crop (bool): 
        verbose (bool): 
        only_target_crop (bool): only crop target image
    """
    os.makedirs(save_dir, exist_ok=True)

    source_and_target_files = [source, target]
    source_and_target_files = [(os.path.basename(f).split('.')[0], f) for f in source_and_target_files]
    result_name = "swap_%s_to_%s.png"%(source_and_target_files[0][0], source_and_target_files[1][0])

    # (1) Crop the faces from the source and target and align them, obtaining S and T 
    if only_target_crop:
        crops, orig_images, quads, inv_transforms = crop_and_align_face(source_and_target_files[1:])
        crops = [crop.convert("RGB") for crop in crops]
        T = crops[0]
        S = Image.open(source).convert("RGB").resize((1024, 1024))
    elif need_crop:
        crops, orig_images, quads, inv_transforms = crop_and_align_face(source_and_target_files)
        crops = [crop.convert("RGB") for crop in crops]
        S, T = crops
    else:
        S = Image.open(source).convert("RGB").resize((1024, 1024))
        T = Image.open(target).convert("RGB").resize((1024, 1024))
        crops = [S, T]
    
    S_256, T_256 = [resize(np.array(im)/255.0, (256, 256)) for im in [S,T]]  # 256,[0,1] range
    T_mask = faceParsing_demo(faceParsing_model, T, convert_to_seg12=True) if target_mask is None else target_mask
    if verbose:
        Image.fromarray(T_mask).save(os.path.join(save_dir,"T_mask.png"))
        T_mask_vis = vis_parsing_maps(T, T_mask)
        Image.fromarray(T_mask_vis).save(os.path.join(save_dir,"T_mask_vis.png"))
    
    # (2) faceVid2Vid  input & output [0,1] range with RGB
    predictions = drive_source_demo(S_256, [T_256], generator, kp_detector, he_estimator, estimate_jacobian)
    predictions = [(pred*255).astype(np.uint8) for pred in predictions]
    # del generator, kp_detector, he_estimator
    
    # (2) GPEN input & output [0,255] range with BGR
    drivens = [GPEN_demo(pred[:,:,::-1], GPEN_model, aligned=False) for pred in predictions]
    D = Image.fromarray(drivens[0][:,:,::-1]) # to PIL.Image
    if verbose:
        D.save(os.path.join(save_dir,"D_%s_to_%s.png"%(source_and_target_files[0][0], source_and_target_files[1][0])))
    

    # (2) mask of D
    D_mask = faceParsing_demo(faceParsing_model, D, convert_to_seg12=True)
    if verbose:
        Image.fromarray(D_mask).save(os.path.join(save_dir,"D_mask.png"))
        D_mask_vis = vis_parsing_maps(D, D_mask) 
        Image.fromarray(D_mask_vis).save(os.path.join(save_dir,"D_mask_vis.png"))
        
    # driven_m_dilated, dilated_verbose = dilate_mask(D_mask, D , radius=3, verbose=True)

    # wrap data 
    driven = transforms.Compose([TO_TENSOR, NORMALIZE])(D)
    driven = driven.to(opts.device).float().unsqueeze(0)
    driven_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(D_mask))
    driven_mask = (driven_mask*255).long().to(opts.device).unsqueeze(0)
    driven_onehot = torch_utils.labelMap2OneHot(driven_mask, num_cls = opts.num_seg_cls)

    target = transforms.Compose([TO_TENSOR, NORMALIZE])(T)
    target = target.to(opts.device).float().unsqueeze(0)
    target_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(T_mask))
    target_mask = (target_mask*255).long().to(opts.device).unsqueeze(0)
    target_onehot = torch_utils.labelMap2OneHot(target_mask, num_cls = opts.num_seg_cls)
    
    # (3) Extract the texture vectors of D and T using RGI
    driven_style_vector, _ = net.get_style_vectors(driven, driven_onehot) 
    target_style_vector, _ = net.get_style_vectors(target, target_onehot)
    if verbose:
        torch.save(driven_style_vector, os.path.join(save_dir,"D_style_vec.pt"))
        driven_style_codes = net.cal_style_codes(driven_style_vector)
        driven_face, _ , structure_feats = net.gen_img(torch.zeros(1,512,32,32).to(opts.device), driven_style_codes, driven_onehot)                
        driven_face_image = torch_utils.tensor2im(driven_face[0])
        driven_face_image.save(os.path.join(save_dir,"D_recon.png"))

        torch.save(target_style_vector, os.path.join(save_dir,"T_style_vec.pt"))
        target_style_codes = net.cal_style_codes(target_style_vector)
        target_face, _ , structure_feats = net.gen_img(torch.zeros(1,512,32,32).to(opts.device), target_style_codes, target_onehot)                
        target_face_image = torch_utils.tensor2im(target_face[0])
        target_face_image.save(os.path.join(save_dir,"T_recon.png"))

    # (4) shape swapping between face D and face T 
    swapped_msk, hole_map = swap_head_mask_revisit_considerGlass(D_mask, T_mask) 
    
    if verbose:
        cv2.imwrite(os.path.join(save_dir,"swappedMask.png"), swapped_msk)
        swappped_one_hot = torch_utils.labelMap2OneHot(torch.from_numpy(swapped_msk).unsqueeze(0).unsqueeze(0).long(), num_cls=12)
        torch_utils.tensor2map(swappped_one_hot[0]).save(os.path.join(save_dir,"swappedMaskVis.png"))
    
    # Texture swapping between face D and face T. Retain the style_vectors of backgroun(0), hair(4), era_rings(11), eye_glass(10) from target
    comp_indices = set(range(opts.num_seg_cls)) - {0, 4, 11, 10}  # 10 glass, 8 neck
    swapped_style_vectors =  swap_comp_style_vector(target_style_vector, driven_style_vector, list(comp_indices), belowFace_interpolation=False)
    if verbose:
        torch.save(swapped_style_vectors, os.path.join(save_dir,"swapped_style_vec.pt"))
    
    # (5) Feed the swapped mask and texture vectors to the generator, obtaining swapped face I;
    swapped_msk = Image.fromarray(swapped_msk).convert('L')
    swapped_msk = transforms.Compose([TO_TENSOR])(swapped_msk)
    swapped_msk = (swapped_msk*255).long().to(opts.device).unsqueeze(0)
    swapped_onehot = torch_utils.labelMap2OneHot(swapped_msk, num_cls = opts.num_seg_cls)
    #        
    swapped_style_codes = net.cal_style_codes(swapped_style_vectors)
    swapped_face, _ , structure_feats = net.gen_img(torch.zeros(1,512,32,32).to(opts.device), swapped_style_codes, swapped_onehot)                
    swapped_face_image = torch_utils.tensor2im(swapped_face[0])
    
    # (6) Stich I back to the target image
    #
    # Gaussian blending with mask
    outer_dilation = 5  
    mask_bg = logical_or_reduce(*[swapped_msk == clz for clz in [0,11, 4    ]])   #For face swapping in video，condisder 4,8,7 as part of background.  11 earings 4 hair 8 neck 7 ear
    is_foreground = torch.logical_not(mask_bg)
    hole_index = hole_map[None][None] == 255
    is_foreground[hole_index[None]] = True
    foreground_mask = is_foreground.float()
    
    if opts.lap_bld:
        content_mask, border_mask, full_mask = create_masks(foreground_mask, outer_dilation=outer_dilation, operation='expansion')
    else:
        content_mask, border_mask, full_mask = create_masks(foreground_mask, outer_dilation=outer_dilation)
    
    content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
    content_mask_image = Image.fromarray(255*content_mask[0,0,:,:].cpu().numpy().astype(np.uint8))
    full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=False)
    full_mask_image = Image.fromarray(255*full_mask[0,0,:,:].cpu().numpy().astype(np.uint8))


    # Paste swapped face onto the target's face
    if opts.lap_bld:
        content_mask = content_mask[0, 0, :, :, None].cpu().numpy()
        border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=False)
        border_mask = border_mask[0, 0, :, :, None].cpu().numpy()
        border_mask = np.repeat(border_mask, 3, axis=-1)

        swapped_and_pasted = swapped_face_image * content_mask + T * (1 - content_mask)
        swapped_and_pasted = Image.fromarray(np.uint8(swapped_and_pasted))
        swapped_and_pasted = Image.fromarray(blending(np.array(T), np.array(swapped_and_pasted), mask=border_mask))
    else:
        if outer_dilation == 0:
            swapped_and_pasted = smooth_face_boundry(swapped_face_image, T, content_mask_image, radius=outer_dilation)
        else:
            swapped_and_pasted = smooth_face_boundry(swapped_face_image, T, full_mask_image, radius=outer_dilation)
    
    # Restore to original image from cropped area
    if only_target_crop:                
        inv_trans_coeffs, orig_image = inv_transforms[0], orig_images[0]
        swapped_and_pasted = swapped_and_pasted.convert('RGBA')
        pasted_image = orig_image.convert('RGBA')
        swapped_and_pasted.putalpha(255)
        projected = swapped_and_pasted.transform(orig_image.size, Image.PERSPECTIVE, inv_trans_coeffs, Image.BILINEAR)
        pasted_image.alpha_composite(projected)
    elif need_crop:                
        inv_trans_coeffs, orig_image = inv_transforms[1], orig_images[1]
        swapped_and_pasted = swapped_and_pasted.convert('RGBA')
        pasted_image = orig_image.convert('RGBA')
        swapped_and_pasted.putalpha(255)
        projected = swapped_and_pasted.transform(orig_image.size, Image.PERSPECTIVE, inv_trans_coeffs, Image.BILINEAR)
        pasted_image.alpha_composite(projected)
    else:
        pasted_image = swapped_and_pasted

    pasted_image.save(os.path.join(save_dir, result_name))

    

if __name__ == "__main__":
    opts = SwapFacePipelineOptions().parse()
    opts.checkpoint_path = '/apdcephfs_cq2/share_1290939/branchwang/projects/pytorch-DDP-demo/iteration_300000.pt'
    # ================= Pre-trained models initilization =========================
    # TODO make a ckpts check in advance
    # face_vid2vid 
    face_vid2vid_cfg = "/apdcephfs_cq2/share_1290939/branchwang/projects/One-Shot_Free-View_Neural_Talking_Head_Synthesis/config/vox-256.yaml"
    face_vid2vid_ckpt = "/apdcephfs_cq2/share_1290939/branchwang/projects/One-Shot_Free-View_Neural_Talking_Head_Synthesis/ckpts/00000189-checkpoint.pth.tar"
    generator, kp_detector, he_estimator, estimate_jacobian = init_facevid2vid_pretrained_model(face_vid2vid_cfg, face_vid2vid_ckpt)
    
    # GPEN 
    gpen_model_params = {
        "base_dir": "/apdcephfs_cq2/share_1290939/branchwang/projects/GPEN/",  # a sub-folder named <weights> should exist
        "in_size": 512,
        "model": "GPEN-BFR-512", 
        "use_sr": True,
        "sr_model": "realesrnet",
        "sr_scale": 4,
        "channel_multiplier": 2,
        "narrow": 1,
    }
    GPEN_model = init_gpen_pretrained_model(model_params = gpen_model_params)

    # face parsing 
    faceParsing_ckpt = "/apdcephfs_cq2/share_1290939/branchwang/pretrained_models/face-parsing.PyTorch/79999_iter.pth"
    faceParsing_model = init_faceParsing_pretrained_model(faceParsing_ckpt)
    print("Load pre-trained face parsing models success!") 

    # E4S model
    net = Net3(opts)
    net = net.to(opts.device)
    save_dict = torch.load(opts.checkpoint_path)
    net.load_state_dict(torch_utils.remove_module_prefix(save_dict["state_dict"], prefix="module."))
    net.latent_avg = save_dict['latent_avg'].to(opts.device)
    print("Load E4S pre-trained model success!") 
    # ========================================================  

    if len(opts.target_mask)!= 0:
        target_mask = Image.open(opts.target_mask).convert("L")
        target_mask_seg12 = __celebAHQ_masks_to_faceParser_mask_detailed(target_mask)
    else:
        target_mask_seg12 = None
    
    faceSwapping_pipeline(opts.source, opts.target, opts, save_dir=opts.output_dir, 
                          target_mask = target_mask_seg12, need_crop = False, verbose = opts.verbose) 
    
