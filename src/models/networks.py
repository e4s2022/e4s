import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
import torch.nn.utils.spectral_norm as spectral_norm

#
from src.models.stylegan2.model import EqualLinear
from src.models.stylegan2.model import Generator
from src.models.encoders import psp_encoders
from src.models.encoders.psp_encoders import FSEncoder_PSP
from src.models.encoders.helpers import get_block, Flatten, bottleneck_IR, bottleneck_IR_SE
from src.models.stylegan2.model import EqualLinear, EqualConv2d

class LocalMLP(nn.Module):
    """MLP module to map texture code to the latnet space of StyleGAN, i.e., W^{r+} space"""
    def __init__(self, dim_component=512, dim_style=512, num_w_layers=18,latent_squeeze_ratio=1):
        super(LocalMLP, self).__init__()
        self.dim_component = dim_component
        self.dim_style = dim_style
        self.num_w_layers = num_w_layers

        self.mlp = nn.Sequential(
            EqualLinear(dim_component, dim_style//latent_squeeze_ratio, lr_mul=1),
            nn.LeakyReLU(),
            EqualLinear(dim_style//latent_squeeze_ratio, dim_style*num_w_layers, lr_mul=1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): with shape [bs, dim_component]

        Returns:
            out: [bs,18,512]
        """
        out=self.mlp(x)
        out=out.view(-1,self.num_w_layers,self.dim_style) # [bs,18,512]
        return out

class Net3(nn.Module):
    """ Multi-scale Style Extracion + StyleGAN with mask-guided injection """

    def __init__(self,opts,):
        super(Net3, self).__init__()
        self.opts = opts
        assert self.opts.fsencoder_type in ["psp"]
        
        self.encoder = FSEncoder_PSP(mode='ir_se', opts=self.opts)
        dim_s_code = 256 + 512 + 512
    
        self.split_layer_idx = 5
        self.remaining_layer_idx = self.opts.remaining_layer_idx
        
        # MLPs in W^{r+} space 
        self.MLPs = nn.ModuleList()
        for i in range(self.opts.num_seg_cls):
            self.MLPs.append(
                LocalMLP(
                    dim_component=dim_s_code,
                    dim_style=512,
                    num_w_layers= self.remaining_layer_idx if self.remaining_layer_idx != 17 else 18
                )
            )
   
        self.G = Generator(size=self.opts.out_size, style_dim=512, n_mlp=8, split_layer_idx = self.split_layer_idx, remaining_layer_idx = self.remaining_layer_idx)

        # fine-tune StyleGAN or not
        if not self.opts.train_G:
            for param in self.G.parameters():
                param.requires_grad = False
        # notice that the 8-layer fully connected module is always fixed
        else:
            for param in self.G.style.parameters():
                param.requires_grad = False
                
        # fix the last layers in StyleGAN (including convs & ToRGBs)
        if self.remaining_layer_idx != 17:
            for param in self.G.convs[-(17-self.remaining_layer_idx):].parameters():
                param.requires_grad = False
            for param in self.G.to_rgbs[-(17-self.remaining_layer_idx)//2 - 1:].parameters():
                param.requires_grad = False
            
    
    def forward(self, img, mask, resize=False, randomize_noise=True,return_latents=False):
       
        codes_vector, structure_feats = self.encoder(F.interpolate(img,(256,256),mode='bilinear'), mask)  # [bs,#seg_cls, D], [bs,C,32,32]
        
        codes=[]
        bs, num_comp = codes_vector.size(0), codes_vector.size(1)
        for i in range(num_comp):
            codes.append(self.MLPs[i](codes_vector[:,i,:])) 
        codes=torch.stack(codes,dim=1)   # [bs, #seg_cls, 13, 512]
        
        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                 # To unified the interface，expand a dummy #seg_cls (i.e., regional) demension in the last 18-K StyleGAN layers
                codes = codes + self.latent_avg[:self.remaining_layer_idx, :].repeat(codes.shape[0],codes.shape[1],1)
                remaining_codes =  self.latent_avg[self.remaining_layer_idx:, :].repeat(bs, num_comp, 1) 
                codes = torch.cat([codes, remaining_codes],dim=2)
            else:
                if self.remaining_layer_idx != 17:
                    codes = codes + self.latent_avg[:self.remaining_layer_idx, :].repeat(codes.shape[0],codes.shape[1],1, 1)
                    remaining_codes =  self.latent_avg[self.remaining_layer_idx:, :].repeat(bs, num_comp, 1, 1)    
                    codes = torch.cat([codes, remaining_codes],dim=2)
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0],codes.shape[1],1, 1)
                
        # G(w)
        images1, result_latent, structure_feats_GT = self.G([codes], structure_feats, mask, input_is_latent=True,
                                                            randomize_noise=randomize_noise,return_latents=return_latents,
                                                            use_structure_code=False)
    
        
        if return_latents:
            return images1, structure_feats_GT, result_latent
        else:
            return images1, structure_feats_GT

    def get_style_vectors(self, img, mask):
        """Given RGB image 和 the corresponding mask, extract the style vectors of each facial component
        
        Args:
            img (Tensor): RGB, each with shape [bs,3,1024,1024]
            mask (Tensor): mask, each with shape [bs,#seg_cls,1024,1024]
           
        Returns:
            style_vectors(Tensor): with shape [bs,#seg_cls,512]
        """
        style_vectors, structure_feats = self.encoder(F.interpolate(img,(256,256),mode='bilinear'), mask)  # [bs,#seg_cls, D], [bs,C,32,32]
        
        return style_vectors, structure_feats
    
    def cal_style_codes(self,style_vectors):
        """Given per-region style vector, map to the style code in StyleGAN latent space via MLPs"""
        
        codes=[]
        bs, num_comp = style_vectors.size(0), style_vectors.size(1)
        for i in range(num_comp):
            codes.append(self.MLPs[i](style_vectors[:,i,:])) 
        codes=torch.stack(codes,dim=1)   # [bs, #seg_cls, 11,512]

        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                codes = codes + self.latent_avg[:self.remaining_layer_idx, :].repeat(codes.shape[0],codes.shape[1],1)
                remaining_codes =  self.latent_avg[self.remaining_layer_idx:, :].repeat(bs, num_comp, 1) 
                style_codes = torch.cat([codes, remaining_codes],dim=2)
            else:
                if self.remaining_layer_idx != 17:
                    codes = codes + self.latent_avg[:self.remaining_layer_idx, :].repeat(codes.shape[0],codes.shape[1],1, 1)
                    remaining_codes =  self.latent_avg[self.remaining_layer_idx:, :].repeat(bs, num_comp, 1, 1)    
                    style_codes = torch.cat([codes, remaining_codes],dim=2)
                else:
                    style_codes = codes + self.latent_avg.repeat(codes.shape[0],codes.shape[1],1, 1)
          
        return style_codes

    def gen_img(self, struc_codes, style_codes, mask, randomize_noise=True, noise=None, return_latents=False):
        """Given mask and the texture vectors of each facial components, generate an image
        
        Args:
            style_codes (Tensor): texture vectors, with shape [bs,#comp,18,512]
            struc_codes (Tensor):
            mask (Tensor): mask, with shape [bs,#seg_cls,1024,1024]
            
            randomize_noise (bool, optional): Defaults to True.
            return_latents (bool, optional): Defaults to False.

        Returns:
            [type]: [description]
        """
        
        images, result_latent, structure_feats = self.G([style_codes], struc_codes, mask, input_is_latent=True,
                                       randomize_noise=randomize_noise,noise=noise,return_latents=return_latents,
                                       use_structure_code=False)

        if return_latents:
            return images, result_latent, structure_feats
        else:
            return images,-1, structure_feats
        
        