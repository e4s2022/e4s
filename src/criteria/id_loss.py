import torch
from torch import nn
from src.models.encoders.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self,opts):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.opts = opts 
        
        self.face_pool_1 = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(opts.ir_se50_path))
        self.face_pool_2 = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        
        self.set_requires_grad(False)
            
    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag
    
    def extract_feats(self, x):
        x = self.face_pool_1(x)  if x.shape[2]!=256 else  x # (1) resize to 256 if needed
        x = x[:, :, 35:223, 32:220]  # (2) Crop interesting region
        x = self.face_pool_2(x) # (3) resize to 112 to fit pre-trained model
        x_feats = self.facenet(x, multi_scale=self.opts.id_loss_multiscale)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats_ms = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats_ms = self.extract_feats(y_hat)
        y_feats_ms = [y_f.detach() for y_f in y_feats_ms]
        
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
    
        return loss_all, sim_improvement_all, None
