import os
from PIL import Image
import torch
import warnings
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms

from src.pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo
from src.utils import torch_utils
from src.options.edit_options import EditOptions
from src.models.networks import Net3
from src.datasets.dataset import TO_TENSOR, NORMALIZE


COMP2INDEX = {'background':0, 'lip':1, 'eyebrows':2, 'eyes':3, 'hair':4, 
            'nose':5, 'skin':6, 'ears':7, 'belowface':8, 'mouth':9, 
            'eye_glass':10, 'ear_rings':11}

class Editor:
    def __init__(self,opts):
        self.opts = opts
        REGIONS = list(COMP2INDEX.keys())

        for region in self.opts.regions:
            assert region in REGIONS, "The input %s is invalid, please choose one from %s"%(region, ",".join(REGIONS))

        assert self.opts.checkpoint_path is not None, "please specify the pre-trained weights!"
        self.net = Net3(self.opts).eval().to(self.opts.device)
        
        ckpt_dict = torch.load(self.opts.checkpoint_path)
        self.net.latent_avg = ckpt_dict['latent_avg'].to(self.opts.device) if self.opts.start_from_latent_avg else None
        self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"],prefix="module."))
        
        self.faceParsing_model = init_faceParsing_pretrained_model(self.opts.faceParser_name, self.opt.faceParsing_ckpt, self.opt.segnext_config)
        
        print("Load pre-trained weights.")    

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * 2,
            128: 128 * 2,
            256: 64 * 2,
            512: 32 * 2,
            1024: 16 * 2,
        }
        self.noise = [torch.randn(1,512,4,4).to(self.opts.device)]
        for i in [8,16,32,64,128,256,512,1024]:
            self.noise.append(torch.randn(1,channels[i],i,i).to(self.opts.device))
            self.noise.append(torch.randn(1,channels[i],i,i).to(self.opts.device))
        
    @torch.no_grad()
    def interpolation(self):
        # =========== For source =============
        src_img = Image.open(self.opts.source).convert("RGB").resize((1024,1024))
        src_label_map = faceParsing_demo(self.faceParsing_model, src_img, convert_to_seg12=True, model_name=self.opts.faceParser_name)
        # wrap data 
        src = transforms.Compose([TO_TENSOR, NORMALIZE])(src_img)
        src = src.to(self.opts.device).float().unsqueeze(0)
        src_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(src_label_map))
        src_mask = (src_mask*255).long().to(self.opts.device).unsqueeze(0)
        src_onehot = torch_utils.labelMap2OneHot(src_mask, num_cls = self.opts.num_seg_cls)
        # Extract the texture vectors using RGI
        src_texture_vectors, _ = self.net.get_style_vectors(src, src_onehot) 

        # =========== For reference =============
        ref_img = Image.open(self.opts.reference).convert("RGB").resize((1024,1024))
        ref_label_map = faceParsing_demo(self.faceParsing_model, ref_img, convert_to_seg12=True, model_name=self.opts.faceParser_name)
        # wrap data
        ref = transforms.Compose([TO_TENSOR, NORMALIZE])(ref_img)
        ref = ref.to(self.opts.device).float().unsqueeze(0)
        ref_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(ref_label_map))
        ref_mask = (ref_mask*255).long().to(self.opts.device).unsqueeze(0)
        ref_onehot = torch_utils.labelMap2OneHot(ref_mask, num_cls = self.opts.num_seg_cls)
        
        ref_texture_vectors, _ = self.net.get_style_vectors(ref, ref_onehot)

        # =========== interpolation =============
        mixed_texture_vectors = src_texture_vectors.clone()

        alpha = self.opts.alpha
        for region in self.opts.regions:
            idx = COMP2INDEX[region]
            mixed_texture_vectors[0,idx,:] = (1-alpha) * src_texture_vectors[0,idx,:] + \
                                            alpha * ref_texture_vectors[0,idx,:]
        
        mixed_style_codes = self.net.cal_style_codes(mixed_texture_vectors)
        # [bs,1,H,W] to one-hot format，i.e., [bs,#seg_cls,H,W]
        onehot = torch_utils.labelMap2OneHot(
            (TO_TENSOR(src_label_map)*255).long().to(self.opts.device).unsqueeze(0),
            num_cls=self.opts.num_seg_cls
        )
        
        generated, _, _ = self.net.gen_img(torch.zeros(1,512,32,32).to(onehot.device), mixed_style_codes, onehot,
                                                randomize_noise=False, noise=self.noise)

        res_img = torch_utils.tensor2im(generated[0])

        return res_img

            
     
if __name__=='__main__':
    opts = EditOptions().parse()
    editor = Editor(opts)
    res = editor.interpolation()

    os.makedirs(opts.output_dir, exist_ok=True)
    res.save(os.path.join(opts.output_dir, "edit_res.png"))