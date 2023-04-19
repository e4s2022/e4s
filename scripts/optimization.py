import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import json
import sys
import pprint
import torch
from functools import partial
import random
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import make_grid
import torch.nn as nn
import glob
from PIL import Image

from src.datasets.dataset import CelebAHQDataset, get_transforms, TO_TENSOR, NORMALIZE, MASK_CONVERT_TF, MASK_CONVERT_TF_DETAILED, FFHQ_MASK_CONVERT_TF_DETAILED
from src.models.networks import Net3
from src.options.optim_options import OptimOptions
from src.datasets.dataset import CelebAHQDataset, get_transforms, TO_TENSOR, NORMALIZE, MASK_CONVERT_TF, MASK_CONVERT_TF_DETAILED, FFHQ_MASK_CONVERT_TF_DETAILED
from src.models.networks import Net3
from src.options.optim_options import OptimOptions
from src.criteria.id_loss import IDLoss
from src.criteria.lpips.lpips import LPIPS
from src.criteria.style_loss import StyleLoss
from src.criteria.face_parsing.face_parsing_loss import FaceParsingLoss
from src.utils import torch_utils
from src.utils.alignmengt import crop_faces, calc_alignment_coefficients
from src.pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps

sys.path.append(".")
sys.path.append("..")

toPIL = transforms.ToPILImage()

def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image

def save_image(image, output_folder, image_name, image_index, ext='jpg'):
    if ext == 'jpeg' or ext == 'jpg':
        image = image.convert('RGB')
    folder = os.path.join(output_folder, image_name)
    os.makedirs(folder, exist_ok=True)
    image.save(os.path.join(folder, "%04d.%s"%(image_index,ext)))
    
class Optimizer:
    def __init__(self, opts):
        self.opts = opts
        
        self.test_ds = CelebAHQDataset(dataset_root=self.opts.dataset_root, mode="test",
                                       img_transform=transforms.Compose(
                                           [TO_TENSOR, NORMALIZE]),
                                       label_transform=transforms.Compose(
                                           [ MASK_CONVERT_TF_DETAILED,TO_TENSOR]), # MASK_CONVERT_TF,
                                       fraction=self.opts.ds_frac)
        print(f"Number of test samples: {len(self.test_ds)}")
        
        assert self.opts.checkpoint_path is not None, "please specify the pre-trained weights!"
        self.net = Net3(self.opts).eval().to(self.opts.device)
        
        ckpt_dict=torch.load(self.opts.checkpoint_path)
        self.net.latent_avg = ckpt_dict['latent_avg'].to(self.opts.device) if self.opts.start_from_latent_avg else None
        if self.opts.load_ema:
            self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict_ema"],prefix="module."))
        else:            
            self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"],prefix="module."))
            
        print("Load pre-trained weights.")
        
        
        # loss 函数
        self.mse_loss = nn.MSELoss().to(self.opts.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.opts.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = IDLoss(self.opts).to(self.opts.device).eval()
        if self.opts.face_parsing_lambda > 0:
            self.face_parsing_loss = FaceParsingLoss(self.opts).to(self.opts.device).eval()
       
        self.img_transform = transforms.Compose([TO_TENSOR, NORMALIZE])
        self.label_transform_wo_converter = transforms.Compose([TO_TENSOR])
        self.label_transform_w_converter = transforms.Compose([MASK_CONVERT_TF_DETAILED, TO_TENSOR])
    
    def calc_loss(self, img, img_recon, mask):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(img_recon, img)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(img_recon, img)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = 0
            for i in range(3):
                loss_lpips_1 = self.lpips_loss(
                    F.adaptive_avg_pool2d(img_recon,(1024//2**i,1024//2**i)), 
                    F.adaptive_avg_pool2d(img,(1024//2**i,1024//2**i))
                )
                loss_lpips += loss_lpips_1
            
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.face_parsing_lambda > 0:
            loss_face_parsing, face_parsing_sim_improvement = self.face_parsing_loss(img_recon, img)
            loss_dict['loss_face_parsing'] = float(loss_face_parsing)
            loss_dict['face_parsing_improve'] = float(face_parsing_sim_improvement)
            loss += loss_face_parsing * self.opts.face_parsing_lambda
        # if self.opts.style_lambda > 0:  # gram matrix loss
        #     loss_style = self.style_loss(img_recon, img, mask_x = (mask==4).float(), mask_x_hat = (mask==4).float())
        #     loss_dict['loss_style'] = float(loss_style)
        #     loss += loss_style * self.opts.style_lambda
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs
          

    def setup_W_optimizer(self, W_init, noise_init=None):
        """
        Args:
            W_init (Tensor): W^{r+} inverted code, with shape [bs,#seg_cls,18,512]

        Returns:
            [type]: [description]
        """

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        
        tmp = W_init.clone().detach()  
        tmp.requires_grad = True

        params = [tmp]
        
        if noise_init is not None:
            noises = []
            for noise in noise_init:
                noise_cp = noise.clone().detach()
                noise_cp.requires_grad = True
                
                noises.append(noise_cp)
                
            params.extend(noises)  
            
        optimizer_W = opt_dict[self.opts.opt_name](params, lr=self.opts.lr)

        if noise_init is not None:
            return optimizer_W, tmp, noises
        else:
            return optimizer_W, tmp
    
    # @torch.no_grad()
    def invertion(self, sample_idx):
        img_name = os.path.basename(self.test_ds.imgs[sample_idx]).split(".")[0]
        intermediate_folder = os.path.join(self.opts.output_dir, img_name)
        os.makedirs(intermediate_folder, exist_ok=True)
        
        img, mask, mask_vis = self.test_ds[sample_idx]
        
        img, mask = img.unsqueeze(0), mask.unsqueeze(0)
        img = img.to(self.opts.device).float()
        mask = (mask * 255).long().to(self.opts.device)
        
        onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opts.num_seg_cls)
        
        # invertion
        with torch.no_grad():
            style_vectors, struc_code = self.net.get_style_vectors(img, onehot)
            style_codes = self.net.cal_style_codes(style_vectors)
            
        # struc_code, style_codes = torch.zeros(1,512,16,16).to(self.opts.device), torch.randn(1,12,18,512).to(self.opts.device)
        ## ================= Recon ======================
        # channels = {
        #     4: 512,
        #     8: 512,
        #     16: 512,
        #     32: 512,
        #     64: 256 * 2,
        #     128: 128 * 2,
        #     256: 64 * 2,
        #     512: 32 * 2,
        #     1024: 16 * 2,
        # }
        # noise = [torch.randn(1,512,4,4).to(self.opts.device)]
        # for i in [8,16,32,64,128,256,512,1024]:
        #     noise.append(torch.randn(1,channels[i],i,i).to(self.opts.device))
        #     noise.append(torch.randn(1,channels[i],i,i).to(self.opts.device))
            
        recon, _, structure_feats = self.net.gen_img(torch.zeros_like(struc_code), style_codes, onehot)
                                                    #  randomize_noise=False,noise=noise)
        
        img_vis = torch_utils.tensor2im(img[0])
        img_vis.save(os.path.join(self.opts.output_dir, img_name, img_name+"_gt.png"))
        recon_vis = torch_utils.tensor2im(recon[0])
        recon_vis.save(os.path.join(self.opts.output_dir, img_name, img_name+"_recon.png"))
        ## =======================================
        
        optimizer_W, latent = self.setup_W_optimizer(style_vectors,noise_init=None)
        pbar = tqdm(range(self.opts.W_steps), desc='Optimizing style code...', leave=False)
        for step in pbar:
            optimizer_W.zero_grad()
            
            style_codes = self.net.cal_style_codes(latent)
            
            recon_i, _, structure_feats_i  = self.net.gen_img(torch.zeros_like(struc_code), style_codes, onehot)
                                                            #   randomize_noise=False,noise=noise)
            
            loss, loss_dict, id_logs = self.calc_loss(img, recon_i, mask)
            
            loss.backward()
            optimizer_W.step()

            if self.opts.verbose:
                verbose_str = "[%s]"%img_name
                for k,v in loss_dict.items():
                    if k[:4]=="loss":
                        verbose_str += (k + " : {:.4f}, ".format(v))
                pbar.set_description(verbose_str)

            if self.opts.save_intermediate and (step+1) % self.opts.save_interval== 0:
                self.save_W_intermediate_results(img_name, recon_i, latent, step+1)
        
        self.save_W_intermediate_results(img_name, recon_i, latent, self.opts.W_steps, noise=None)
        
    def save_W_intermediate_results(self, img_name, gen_im, latent, step, noise=None):
              
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent.detach().cpu().numpy()
        
        intermediate_folder = os.path.join(self.opts.output_dir,img_name)
        os.makedirs(intermediate_folder, exist_ok=True)

        latent_path = os.path.join(intermediate_folder, f'{img_name}_{step:04}.npy')
        image_path = os.path.join(intermediate_folder, f'{img_name}_{step:04}.png')

        save_im.save(image_path)
        
        if noise is not None:
            save_stats = {}
            save_stats["style_code"] = save_latent
            save_stats["noise"] = [n.detach().cpu().numpy() for n in noise]
            torch.save(save_stats,latent_path)
        else:
            np.save(latent_path, save_latent)

        
if __name__ == '__main__':
    opts = OptimOptions().parse()
    print("make dir %s"%opts.output_dir)
    os.makedirs(opts.output_dir, exist_ok=True)
    
    optimizer = Optimizer(opts)
    optimizer.invertion(123)  # image index of CelebAMask-HQ test split
    
