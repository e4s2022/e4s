import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np



VGG_MEAN = np.array([0.485, 0.456, 0.406]).astype(np.float32)
VGG_STD = np.array([0.229, 0.224, 0.225]).astype(np.float32)


def custom_loss(x, y, mask=None, loss_type="l2", include_bkgd=True):
    """
    x, y: [N, C, H, W]
    Computes L1/L2 loss

    if include_bkgd is True:
        use traditional MSE and L1 loss
    else:
        mask out background info using :mask
        normalize loss with #1's in mask
    """
    if include_bkgd:
        # perform simple mse or l1 loss
        if loss_type == "l2":
            loss_rec = F.mse_loss(x, y)
        elif loss_type == "l1":
            loss_rec = F.l1_loss(x, y)

        return loss_rec

    Nx, Cx, Hx, Wx = x.shape
    Nm, Cm, Hm, Wm = mask.shape
    mask = prepare_mask(x, mask)

    x_reshape = torch.reshape(x, [Nx, -1])
    y_reshape = torch.reshape(y, [Nx, -1])
    mask_reshape = torch.reshape(mask, [Nx, -1])

    if loss_type == "l2":
        diff = (x_reshape - y_reshape) ** 2
    elif loss_type == "l1":
        diff = torch.abs(x_reshape - y_reshape)

    # diff: [N, Cx * Hx * Wx]
    # set elements in diff to 0 using mask
    masked_diff = diff * mask_reshape
    sum_diff = torch.sum(masked_diff, axis=-1)
    # count non-zero elements; add :mask_reshape elements
    norm_count = torch.sum(mask_reshape, axis=-1)
    diff_norm = sum_diff / (norm_count + 1.0)

    loss_rec = torch.mean(diff_norm)

    return loss_rec


def prepare_mask(x, mask):
    """
    Make mask similar to x.
    Mask contains values in [0, 1].
    Adjust channels and spatial dimensions.
    """
    Nx, Cx, Hx, Wx = x.shape
    Nm, Cm, Hm, Wm = mask.shape
    if Cm == 1:
        mask = mask.repeat(1, Cx, 1, 1)

    mask = F.interpolate(mask, scale_factor=Hx / Hm, mode="nearest")

    return mask


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class VGG16_Activations(nn.Module):
    def __init__(self, feature_idx):
        super(VGG16_Activations, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        requires_grad(vgg16, flag=False)
        
        features = list(vgg16.features)
        self.features = nn.ModuleList(features).eval()
        self.layer_id_list = feature_idx

    def forward(self, x):
        activations = []
        for i, model in enumerate(self.features):
            x = model(x)
            if i in self.layer_id_list:
                activations.append(x)

        return activations


class StyleLoss(nn.Module):
    def __init__(self, VGG16_ACTIVATIONS_LIST=[21], normalize=False, distance="l2", in_size=256):

        super(StyleLoss, self).__init__()

        self.vgg16_act = VGG16_Activations(VGG16_ACTIVATIONS_LIST)
        self.vgg16_act.eval()

        
        self.in_size = in_size
        # self.upsample2d = nn.Upsample(scale_factor=256 / in_size, mode="bilinear", align_corners=True)
        
        
        self.normalize = normalize
        self.distance = distance

    def normalize_img(self, x):
        """
            https://pytorch.org/vision/stable/models.html
            
            x: [bs,3,H,W]  with range [-1,1]
        """
        x = (x + 1) / 2
        
        mean = torch.from_numpy(VGG_MEAN).view(1,3,1,1).to(x.device)
        std = torch.from_numpy(VGG_STD).view(1,3,1,1).to(x.device)
        
        x = (x - mean) / std
        
        return x
        
    def forward(self, x, x_hat, mask_x=None, mask_x_hat=None):
        # x = x.cuda()
        # x_hat = x_hat.cuda()
        # resize images to 256px resolution
        
        N, C, H, W = x.shape
        
        # x = self.upsample2d(x)
        # x_hat = self.upsample2d(x_hat)
        
        x = F.interpolate(x, size=(256,256), mode="bilinear")
        x_hat = F.interpolate(x_hat, size=(256,256), mode="bilinear")

        if self.normalize:
            x = self.normalize_img(x)
            x_hat = self.normalize_img(x_hat)
            
        loss = self.cal_style(self.vgg16_act, x, x_hat, mask_x=mask_x, mask_x_hat=mask_x_hat)

        return loss

    def cal_style(self, model, x, x_hat, mask_x=None, mask_x_hat=None):
        # Get features from the model for x and x_hat
        
        # with torch.no_grad():
        #     act_x = self.get_features(model, x)
        # for layer in range(0, len(act_x)):
        #     act_x[layer].detach_()
        
        # mask 图片
        if mask_x is not None:
            assert mask_x_hat is not None, "mask_x_hat must be non-empty!"
            H, W = x.size(2), x.size(3)
            mask_x = F.interpolate(mask_x, size=(H,W),mode="bilinear")
            x = x * mask_x
            
            mask_x_hat = F.interpolate(mask_x_hat, size=(H,W),mode="bilinear")
            x_hat = x_hat * mask_x_hat

        act_x = self.get_features(model, x)
        act_x_hat = self.get_features(model, x_hat)

        loss = 0.0
        for layer in range(0, len(act_x)):
            # # mask features if present
            # if mask_x is not None:
            #     feat_x = self.mask_features(act_x[layer], mask_x)
            # else:
            #     feat_x = act_x[layer]
                
            # if mask_x_hat is not None:
            #     feat_x_hat = self.mask_features(act_x_hat[layer], mask_x_hat)
            # else:
            #     feat_x_hat = act_x_hat[layer]
            
            feat_x = act_x[layer]
            feat_x_hat = act_x_hat[layer]

            # compute Gram matrix for x and x_hat
            G_x = self.gram_matrix(feat_x)
            G_x_hat = self.gram_matrix(feat_x_hat)

            # compute layer wise loss and aggregate
            loss += custom_loss(
                G_x, G_x_hat, mask=None, loss_type=self.distance, include_bkgd=True
            )

        loss = loss / len(act_x)

        return loss

    def get_features(self, model, x):

        return model(x)

    def mask_features(self, x, mask):

        mask = prepare_mask(x, mask)
        return x * mask

    def gram_matrix(self, x):
        """
        :x is an activation tensor
        """
        N, C, H, W = x.shape
        x = x.view(N * C, H * W)
        G = torch.mm(x, x.t())

        return G.div(N * H * W * C)

if __name__=="__main__":
    style = StyleLoss(distance="l2", VGG16_ACTIVATIONS_LIST=[3,8,15,22], normalize=True).cuda()

    x = torch.randn(1,3,256,256).cuda()
    x_hat = torch.randn(1,3,256,256).cuda()
    
    loss = style(x, x_hat, mask_x=None, mask_x_hat=None)
    print(-1)