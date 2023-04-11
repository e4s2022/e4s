import torch
from torch import nn
import torch.nn.functional as F
from src.models.stylegan2.op import conv2d_gradfix
from torch import autograd
import math

class AdvGLoss(nn.Module):

	def __init__(self):
		super(AdvGLoss, self).__init__()

	def forward(self, fake_pred):
		loss = F.softplus(-fake_pred).mean()
		return loss

class AdvDLoss(nn.Module):

	def __init__(self):
		super(AdvDLoss, self).__init__()

	def forward(self, real_pred, fake_pred):
		real_loss = F.softplus(-real_pred)
		fake_loss = F.softplus(fake_pred)
		return real_loss.mean() + fake_loss.mean()



class DR1Loss(nn.Module):
    def __init__(self):
        super(DR1Loss, self).__init__()

    def forward(self,real_pred, real_img):
        with conv2d_gradfix.no_weight_gradients():
            grad_real, = autograd.grad(
                outputs=real_pred.sum(), inputs=real_img, create_graph=True
            )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty
    

class GPathRegularizer(nn.Module):
    def __init__(self):
        super(GPathRegularizer, self).__init__()
    
    def forward(self, fake_img, latents, mean_path_length, decay=0.01):
        noise = torch.randn_like(fake_img) / math.sqrt(
            fake_img.shape[2] * fake_img.shape[3]
        )
        grad, = autograd.grad(
            outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
        )
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

        path_penalty = (path_lengths - path_mean).pow(2).mean()

        return path_penalty, path_mean.detach(), path_lengths