from utils import torch_utils
import torchvision.transforms as transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch.distributed as dist
import math
# torch.autograd.set_detect_anomaly(True)
from collections import OrderedDict

from src.datasets.dataset import CelebAHQDataset, get_transforms, TO_TENSOR, NORMALIZE, MASK_CONVERT_TF, FFHQDataset, FFHQ_MASK_CONVERT_TF, MASK_CONVERT_TF_DETAILED, FFHQ_MASK_CONVERT_TF_DETAILED
from src.criteria.w_norm import WNormLoss
from src.criteria.id_loss import IDLoss
from src.criteria.face_parsing.face_parsing_loss import FaceParsingLoss
from src.criteria.lpips.lpips import LPIPS
from src.criteria.adv_loss import AdvDLoss,AdvGLoss,DR1Loss,GPathRegularizer
from src.criteria.style_loss import StyleLoss
from src.training.ranger import Ranger
from src.models.networks import Net, Net2, Net3, NetStage2,MultiScaleNet
from src.models.stylegan2.model import Generator,Discriminator


ACCUM = 0.5 ** (32 / (100 * 1000))  #  0.9977843871238888

class Coach:
     
    def __init__(self, opts):
        self.opts = opts
        self.global_step = 0

        # distributed training
        if self.opts.dist_train:
            self.num_gpus = torch.cuda.device_count()
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])

            torch.cuda.set_device(self.rank % self.num_gpus)

            dist.init_process_group(
                backend='nccl',
                world_size=self.world_size,
                rank=self.rank,
            )
            self.device = torch.device("cuda", self.local_rank)
            
            
        else:
            self.rank=0 # dummy rank
            self.device = torch.device("cuda", 0)
        
        self.opts.device=self.device

        # ==== Initialize network ====
        self.net = Net3(self.opts)
        # print(self.device)
        self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.net = self.net.to(self.device)
        
        self.net_ema = Net3(self.opts).to(self.device).eval()
        torch_utils.accumulate(self.net_ema,self.net, 0)
        
        if self.opts.train_D:
            self.D = Discriminator(self.opts.out_size).to(self.device).eval()
    
        if self.opts.dist_train:
            # Wrap the model
            self.net = nn.parallel.DistributedDataParallel(self.net,
            device_ids=[self.local_rank], output_device=self.local_rank,
            broadcast_buffers=False, 
            find_unused_parameters=True
            )

            if self.opts.train_D: 
                self.D = nn.parallel.DistributedDataParallel(self.D,
                device_ids=[self.local_rank], output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True
                )
            
        # 加载整个模型预训练好的参数，继续训练
        if self.opts.checkpoint_path is not None:
            ckpt_dict=torch.load(self.opts.checkpoint_path)
            self.global_step= ckpt_dict["opts"]["max_steps"]+1
            
            if self.opts.dist_train:    
                self.net.module.latent_avg = ckpt_dict['latent_avg'].to(self.device)
                self.net.load_state_dict(ckpt_dict["state_dict"])
                if self.opts.train_D:
                    self.D.module.load_state_dict(ckpt_dict["D_state_dict"]) 
            else:
                self.net.latent_avg = ckpt_dict['latent_avg'].to(self.device)
                self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"],prefix="module."))
                if self.opts.train_D:
                    self.D.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["D_state_dict"],prefix="module."))

            print("Resume training at step %d..."%self.global_step)            
        
        # 加载 stage-1 训练好的参数
        elif self.opts.stage1_checkpoint_path is not None:
            stage1_ckpt = torch.load(self.opts.stage1_checkpoint_path)
            
            if self.opts.dist_train:
                self.net.module.stage1_net.load_state_dict(torch_utils.remove_module_prefix(stage1_ckpt["state_dict"],prefix="module."))
                if self.opts.train_D:
                    self.D.module.load_state_dict(torch_utils.remove_module_prefix(stage1_ckpt["D_state_dict"],prefix="module."))
                    
                # avg latent code
                self.net.module.latent_avg = stage1_ckpt['latent_avg'].to(self.device)    
                if self.opts.learn_in_w:
                    self.net.module.latent_avg = self.net.module.latent_avg
                else:
                    self.net.module.latent_avg = self.net.module.latent_avg
            else:
                self.net.stage1_net.load_state_dict(torch_utils.remove_module_prefix(stage1_ckpt["state_dict"],prefix="module."))
                if self.opts.train_D:
                    self.D.load_state_dict(torch_utils.remove_module_prefix(stage1_ckpt["D_state_dict"],prefix="module."))
                  
                # avg latent code
                self.net.latent_avg = stage1_ckpt['latent_avg'].to(self.device)    
                if self.opts.learn_in_w:
                    self.net.latent_avg = self.net.latent_avg
                else:
                    self.net.latent_avg = self.net.latent_avg
            
            print('Loading stage-1 pretrained weights!')
        
        # 加载styleGAN预训练权重
        else:
            styleGAN2_ckpt = torch.load(self.opts.stylegan_weights)
            
            if self.opts.dist_train:
                self.net.module.G.load_state_dict(styleGAN2_ckpt['g_ema'], strict=False)
                if self.opts.train_D:
                    if self.opts.out_size == 1024:
                        self.D.module.load_state_dict(styleGAN2_ckpt['d'], strict=False) # 1024分辨率 可以直接加载
                    else:
                        self.custom_load_D_state_dict(self.D.module, styleGAN2_ckpt['d'])  # 只加载判别器的部分层
                # avg latent code
                self.net.module.latent_avg = styleGAN2_ckpt['latent_avg'].to(self.device)    
                if self.opts.learn_in_w:
                    self.net.module.latent_avg = self.net.module.latent_avg.repeat(1, 1)
                else:
                    self.net.module.latent_avg = self.net.module.latent_avg.repeat(2 * int(math.log(self.opts.out_size, 2)) -2 , 1)
            else:
                self.net.G.load_state_dict(styleGAN2_ckpt['g_ema'], strict=False)
                if self.opts.train_D:
                    if self.opts.out_size == 1024:
                        self.D.load_state_dict(styleGAN2_ckpt['d'], strict=False) # 1024分辨率 可以直接加载
                    else:
                        self.custom_load_D_state_dict(self.D, styleGAN2_ckpt['d']) # 只加载判别器的部分层 
                # avg latent code
                self.net.latent_avg = styleGAN2_ckpt['latent_avg'].to(self.device)    
                if self.opts.learn_in_w:
                    self.net.latent_avg = self.net.latent_avg.repeat(1, 1)
                else:
                    self.net.latent_avg = self.net.latent_avg.repeat(2 * int(math.log(self.opts.out_size, 2)) -2 , 1)
            
            print('Loading pretrained styleGAN2 weights!')

        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.opts.dist_train:
            if self.net.module.latent_avg is None:
                self.net.module.latent_avg = self.net.module.G.mean_latent(int(1e5))[0].detach()
        else:
            if self.net.latent_avg is None:
                self.net.latent_avg = self.net.G.mean_latent(int(1e5))[0].detach()

        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = IDLoss(self.opts).to(self.device).eval()
        if self.opts.face_parsing_lambda > 0:
            self.face_parsing_loss = FaceParsingLoss(self.opts).to(self.device).eval()
        if self.opts.w_norm_lambda > 0:
            self.w_norm_loss = WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
        if self.opts.style_lambda > 0:  # gram matrix loss
            self.style_loss = StyleLoss(distance="l2", VGG16_ACTIVATIONS_LIST=[3,8,15,22],
                                        normalize = self.opts.style_loss_norm==1,
                                        in_size=self.opts.out_size).to(self.device).eval()

        self.adv_d_loss=AdvDLoss().to(self.device).eval()
        self.adv_g_loss=AdvGLoss().to(self.device).eval()
        self.d_r1_reg_loss=DR1Loss().to(self.device).eval()
        self.g_path_reg_loss=GPathRegularizer().to(self.device).eval()
        
        # Initialize optimizer
        self.optimizer,self.optimizer_D = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        if self.opts.dist_train:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset,shuffle=True)
            self.train_dataloader = DataLoader(self.train_dataset,
                                            batch_size=self.opts.batch_size,
                                            num_workers=int(self.opts.workers),
                                            drop_last=True,
                                            pin_memory=True,
                                            sampler=self.train_sampler)
        else:
            self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        # 测试集不要分布式计算
        self.test_dataloader = DataLoader(self.test_dataset,
                                        batch_size=self.opts.test_batch_size,
                                        shuffle=False,
                                        num_workers=int(
                                            self.opts.test_workers),
                                        drop_last=False)

        # Initialize tensorborad logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        if self.rank==0:
            self.logger = SummaryWriter(logdir =log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps
    
    
    def custom_load_D_state_dict(self, module, state_dict):
        """导入 styleGAN 预训练判别器 特定层的权值
        Args:
            module (nn.Module): 即将更新参数的module
            state_dict (): styleGAN 预训练判别器的参数, convs.0对应1024尺度
        """
        # 开始导入预训练的 ResBlocks 权重
        local_state = {k: v for k, v in module.named_parameters() if v is not None}

        # 因为styleGAN2 的判别器第一层是 1*1 卷积，且输出的通道和分辨率相关，所以这里没办法导入 第一个预训练的conv层权重
        del local_state["convs.0.0.weight"]
        del local_state["convs.0.1.bias"]

        idx_gap = int(math.log(1024, 2)) - int(math.log(self.opts.out_size, 2))
        
        new_state_dict = OrderedDict()
        for name, param in local_state.items():
            if name[:5]=="convs":
                layer_idx = int(name[6])
                name_in_pretrained = name[:6] + str(layer_idx + idx_gap) + name[7:]
                new_state_dict[name] = state_dict[name_in_pretrained]
            else:
                new_state_dict[name] = state_dict[name]  # 最后几层FC
        
        module.load_state_dict(new_state_dict, strict=False)
        # print(-1)
                
    def configure_optimizers(self):
        self.params=list(filter(lambda p: p.requires_grad ,list(self.net.parameters())))
        self.params_D=list(filter(lambda p: p.requires_grad ,list(self.D.parameters()))) if self.opts.train_D else None
        
        d_reg_ratio = self.opts.d_reg_every / (self.opts.d_reg_every + 1) if self.opts.d_reg_every >0 else 1
        
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(self.params, lr=self.opts.learning_rate)
            optimizer_D = torch.optim.Adam(self.params_D, lr=self.opts.learning_rate * d_reg_ratio) if self.opts.train_D else None
        else:
            optimizer = Ranger(self.params, lr=self.opts.learning_rate)
            optimizer_D = Ranger(self.params_D, lr=self.opts.learning_rate * d_reg_ratio) if self.opts.train_D else None
        return optimizer,optimizer_D

    def configure_datasets(self):
        if self.opts.dataset_name=="ffhq":    
            train_ds = FFHQDataset(dataset_root=self.opts.ffhq_dataset_root,
                                    img_transform=transforms.Compose(
                                        [TO_TENSOR, NORMALIZE]),
                                    label_transform=transforms.Compose(
                                        [FFHQ_MASK_CONVERT_TF_DETAILED, TO_TENSOR]),  # FFHQ_MASK_CONVERT_TF
                                    fraction=self.opts.ds_frac,
                                    flip_p=self.opts.flip_p)
        else:
            train_ds = CelebAHQDataset(dataset_root=self.opts.celeba_dataset_root, mode="train",
                                    img_transform=transforms.Compose(
                                        [TO_TENSOR, NORMALIZE]),
                                    label_transform=transforms.Compose(
                                        [MASK_CONVERT_TF_DETAILED, TO_TENSOR]),  # MASK_CONVERT_TF_DETAILED
                                    fraction=self.opts.ds_frac,
                                    flip_p=self.opts.flip_p)
        
        test_ds = CelebAHQDataset(dataset_root=self.opts.celeba_dataset_root, mode="test",
                                  img_transform=transforms.Compose(
                                      [TO_TENSOR, NORMALIZE]),
                                  label_transform=transforms.Compose(
                                      [MASK_CONVERT_TF_DETAILED, TO_TENSOR]),  # MASK_CONVERT_TF
                                  fraction=self.opts.ds_frac)
        print(f"Number of training samples: {len(train_ds)}")
        print(f"Number of test samples: {len(test_ds)}")
        return train_ds, test_ds

    # @torch.no_grad()
    def train(self):
        self.net.train()  # 这里也会把styleGAN给设置成 train模式了，最好这里判断一下是否是finetune阶段
        if self.opts.train_D:
            self.D.train()
        
        while self.global_step <= self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):                    
                img, mask, mask_vis = batch
                
                img = img.to(self.device).float()
                mask = (mask*255).long().to(self.device)
                # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
                onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opts.num_seg_cls)
                # onehot_1024 = F.interpolate(onehot, size=(1024,1024), mode='nearest')
                
                # ============ 训练D ===============
                if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
                    torch_utils.requires_grad(self.net, False)
                    torch_utils.requires_grad(self.D, True)
                
                    # recon1, recon2, structure_codes, structure_codes_GT, latent  = self.net(img, onehot, return_latents=True)
                    recon1, structure_codes_GT, latent = self.net(img, onehot, return_latents=True)
                    
                    fake_pred_1 = self.D(recon1)
                    # fake_pred_2 = self.D(recon2)
                    real_pred = self.D(img)
                    
                    # fake_pred = self.D(torch.cat([recon, onehot_1024], dim=1))
                    # real_pred = self.D(torch.cat([img, onehot_1024], dim=1))
                    
                    # d_loss = self.adv_d_loss(real_pred,torch.cat([fake_pred_1,fake_pred_2], dim=0))
                    d_loss = self.adv_d_loss(real_pred,fake_pred_1)
                    
                    d_loss_dict={}
                    d_loss_dict["d_loss"]=float(d_loss)
                    d_loss_dict["real_score"]=float(real_pred.mean())
                    d_loss_dict["fake_score_1"]=float(fake_pred_1.mean())
                    # d_loss_dict["fake_score_2"]=float(fake_pred_2.mean())
                    
                    self.D.zero_grad()
                    d_loss.backward()
                    self.optimizer_D.step()
                    
                    r1_loss = torch.tensor(0.0, device=self.device)
                    # 每间隔几次算一次 R1 regularization
                    if self.opts.d_reg_every!=-1 and batch_idx % self.opts.d_reg_every==0:
                        img.requires_grad=True
                        
                        real_pred = self.D(img)
                        r1_loss = self.d_r1_reg_loss(real_pred, img)
                        
                        self.D.zero_grad()
                        (self.opts.r1_lambda / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]).backward()
                        self.optimizer_D.step()
                        
                    d_loss_dict["r1_loss"] = r1_loss
                
                # ============ 训练G ===============
                # self.opts.train_G 和 self.opts.train_D 要么都是true，要么都是false
                # 对抗学习的时候训练G,此时的G视为 Encoder + StyleGAN Generator
                if self.opts.train_G and self.opts.train_D:  
                    torch_utils.requires_grad(self.net, True)
                    torch_utils.requires_grad(self.net.module.G.style, False)  # z到W 的 mapping参数永远不会更新
                    
                    # styleGAN的倒数几层不更新
                    if self.opts.remaining_layer_idx != 17:
                        torch_utils.requires_grad(self.net.module.G.convs[-(17-self.opts.remaining_layer_idx):],False)
                        torch_utils.requires_grad(self.net.module.G.to_rgbs[-(17-self.opts.remaining_layer_idx)//2 - 1:],False)
                
                # 只训练Encoder
                elif not self.opts.train_G and not self.opts.train_D:  
                    torch_utils.requires_grad(self.net.module.G, False)
                    # torch_utils.requires_grad(self.net.module.stage1_net.G, False)
                
                if self.opts.train_D:
                    torch_utils.requires_grad(self.D, False)  
                
                # recon1 完全使用style code生成, recon2 使用style code和structure code生成
                # recon1, recon2, structure_codes, structure_codes_GT, latent = self.net(img, onehot, return_latents=True)    
                recon1, structure_codes_GT, latent = self.net(img, onehot, return_latents=True)
                
                g_loss = torch.tensor(0.0, device=self.device)                
                if self.opts.train_D:
                    # fake_pred = self.D(torch.cat([recon, onehot_1024], dim=1))
                    fake_pred_1 = self.D(recon1)
                    # fake_pred_2 = self.D(recon2)
                    # g_loss = self.adv_g_loss(torch.cat([fake_pred_1,fake_pred_2], dim=0))
                    g_loss = self.adv_g_loss(fake_pred_1)
                
                loss_, loss_dict, id_logs = self.calc_loss(img, recon1, mask, latent)
                loss_dict["g_loss"] = float(g_loss)
                
                overall_loss = loss_ + self.opts.g_adv_lambda * g_loss
                loss_dict["loss"] = float(overall_loss)
                
                # if self.rank==0:
                #     for name, param in self.net.named_parameters():
                #         if param.requires_grad:
                #             print('-->name:', name, '--weight', torch.mean(param.data), ' -->grad_value:', torch.mean(param.grad), '-->is_finite', torch.isfinite(param.grad).all()) 

                self.net.zero_grad()
                overall_loss.backward()
                self.optimizer.step()
                
                # Logging related
                if self.rank==0 and (self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0)):
                    # imgs_1和imgs_2都是list，每个元素构成一个pair,即 imgs_1[i]和imgs_2[i]是一对图片
                    imgs = self.parse_images(onehot, img, recon1)
                    self.log_images('images/train/faces', imgs1_data=imgs, subscript=None)

                if self.rank==0 and (self.global_step % self.opts.board_interval == 0):
                    self.print_metrics(loss_dict, prefix='train')
                    if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
                        self.print_metrics(d_loss_dict, prefix='train')
                        
                    self.log_metrics(loss_dict, prefix='train')
                    if self.opts.train_D and (self.global_step % self.opts.d_every == 0):
                        self.log_metrics(d_loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                # 验证保存最好的模型
                if self.rank==0 and (self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps):
                    val_loss_dict = self.validate()
                if self.rank==0 and (val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss)):
                    self.best_val_loss = val_loss_dict['loss']
                    self.checkpoint_me(val_loss_dict, is_best=True)

                if self.rank==0 and (self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps):
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                self.global_step += 1
                if self.global_step == 100000:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.opts.learning_rate * 0.1
            
                # 参数滑动平均
                if self.opts.dist_train:
                    torch_utils.accumulate(self.net_ema, self.net.module, ACCUM)
                else:
                    torch_utils.accumulate(self.net_ema, self.net, ACCUM)
                
        if self.rank==0:
            print('OMG, finished training!')

    def calc_loss(self, img, recon1, mask, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        
        if self.opts.face_parsing_lambda > 0:
            loss_face_parsing_1, face_parsing_sim_improvement_1 = self.face_parsing_loss(recon1, img)
            
            loss_dict['loss_face_parsing'] = float(loss_face_parsing_1)
            loss_dict['face_parsing_improve'] = float(face_parsing_sim_improvement_1)
            loss += loss_face_parsing_1 * self.opts.face_parsing_lambda
            
        if self.opts.id_lambda > 0:
            loss_id_1, sim_improvement_1, id_logs_1 = self.id_loss(recon1, img)
                   
            loss_dict['loss_id'] = float(loss_id_1)
            loss_dict['id_improve'] = float(sim_improvement_1)
            loss += loss_id_1 * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2_1 = F.mse_loss(recon1, img)
            
            loss_dict['loss_l2'] = float(loss_l2_1)
            loss += loss_l2_1 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = 0
            for i in range(3):
                loss_lpips_1 = self.lpips_loss(
                    F.adaptive_avg_pool2d(recon1,(1024//2**i,1024//2**i)), 
                    F.adaptive_avg_pool2d(img,(1024//2**i,1024//2**i))
                )
               
                loss_lpips += loss_lpips_1
            
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.w_norm_lambda > 0:
            if self.opts.dist_train:
                loss_w_norm = self.w_norm_loss(latent, self.net.module.latent_avg)
            else:
                loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
                
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.opts.w_norm_lambda
        if self.opts.style_lambda > 0:  # gram matrix loss
            loss_style_1 = self.style_loss(recon1, img, mask_x = (mask==3).float(), mask_x_hat = (mask==3).float())
            
            loss_dict['loss_style'] = float(loss_style_1)
            loss += loss_style_1 * self.opts.style_lambda
            
        loss_dict['loss'] = float(loss)
        return (loss, loss_dict, id_logs_1) if self.opts.id_lambda > 0 else (loss, loss_dict, None)

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(
                f'{prefix}/{key}', value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_images(self, mask, img, recon1, display_count=2):
        im_data = []

        display_count=min(display_count,len(img))
        for i in range(display_count):
            cur_im_data = {
                'input_face': torch_utils.tensor2im(img[i]),
                'input_mask': torch_utils.tensor2map(mask[i]),
                'recon_styleCode': torch_utils.tensor2im(recon1[i]),
                # 'recon_styleCode_feats': torch_utils.tensor2im(recon2[i]),
            }
            im_data.append(cur_im_data)

        return im_data

    def log_images(self, name, imgs1_data, subscript=None, log_latest=False):
        fig = torch_utils.vis_faces(imgs1_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.logdir , name,
                                f'{subscript}_{step:06d}.jpg')
        else:
            path = os.path.join(self.logger.logdir , name, f'{step:06d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts),
            'state_dict_ema': self.net_ema.state_dict(),
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.module.latent_avg if self.opts.dist_train else self.net.latent_avg
            
        if self.opts.train_D:
            save_dict['D_state_dict'] = self.D.state_dict()
        return save_dict

    def validate(self):
        # show_images=False
        # Logging related 虽然每个epoch都 validate一下，但是每间隔3个epoch再显示图片，防止图片太多了
        if self.global_step % (4*self.opts.val_interval) == 0 or self.global_step == self.opts.max_steps:
            show_images=True
        else:
            show_images=False
            
        self.net.eval()
        if self.opts.train_D:
            self.D.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            
            img, mask, mask_vis = batch
            
            with torch.no_grad():
                img = img.to(self.device).float()
                mask = (mask*255).long().to(self.device)
                # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
                onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opts.num_seg_cls)
                # onehot_1024 = F.interpolate(onehot, size=(1024,1024), mode='nearest')
                
                # recon1 完全使用style code生成, recon2 使用style code和structure code生成
                # recon1, recon2, structure_codes, structure_codes_GT, latent = self.net(img, onehot, return_latents=True)
                recon1, structure_codes_GT, latent = self.net(img, onehot, return_latents=True)    
                              
                g_loss = torch.tensor(0.0, device=self.device)                
                if self.opts.train_D:
                    # fake_pred = self.D(torch.cat([recon, onehot_1024], dim=1))
                    fake_pred_1 = self.D(recon1)
                    # fake_pred_2 = self.D(recon2)
                    # g_loss = self.adv_g_loss(torch.cat([fake_pred_1,fake_pred_2], dim=0))
                    g_loss = self.adv_g_loss(fake_pred_1)
                
                loss_, loss_dict, id_logs = self.calc_loss(img, recon1, mask, latent)
                loss_dict["g_loss"] = float(g_loss)
                
                overall_loss = loss_ + self.opts.g_adv_lambda * g_loss
                loss_dict["loss"] = float(overall_loss)
                
            agg_loss_dict.append(loss_dict)

            if show_images:
                imgs = self.parse_images(onehot, img, recon1)
                self.log_images('images/test/faces', imgs1_data=imgs, subscript='{:04d}'.format(batch_idx))

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                if self.opts.train_D:
                    self.D.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = torch_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        if self.opts.train_D:
            self.D.train()
        return loss_dict
