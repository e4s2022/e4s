import gradio as gr
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from src.pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo
from src.options.edit_options import EditOptions
from src.models.networks import Net3
from src.datasets.dataset import TO_TENSOR, NORMALIZE
from src.utils import torch_utils

from demo.gradio_utils import *

# ================================ A helper class for storing intermediate results ================================
class DemoHelper:
    def __init__(self):
        self.opt = EditOptions().parse()

        # initialize the pre-trained models
        self.faceParsing_model = init_faceParsing_pretrained_model(self.opt.faceParsing_ckpt)
        assert self.opt.faceParsing_ckpt is not None, "please fetch the pre-trained faceParsing model checkpoint!"
        
        # RGI model
        self.rgi_ckpt = self.opt.checkpoint_path
        assert self.opt.checkpoint_path is not None, "please fetch the pre-trained E4S model checkpoint!"
        
        self.net = Net3(self.opt).eval().to(self.opt.device)
        ckpt_dict = torch.load(self.opt.checkpoint_path)
        self.net.latent_avg = ckpt_dict['latent_avg'].to(self.opt.device) if self.opt.start_from_latent_avg else None
        self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"],prefix="module."))
        print("Loading Done!")   

        self.src_img = None
        self.initial_label_map = None
        self.ref_img = None
        self.ref_label_map = None
        
        self.src_texture_vectors = None
        self.ref_texture_vectors = None

        # Noise
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
        self.noise = [torch.randn(1,512,4,4).to(self.opt.device)]
        for i in [8,16,32,64,128,256,512,1024]:
            self.noise.append(torch.randn(1,channels[i],i,i).to(self.opt.device))
            self.noise.append(torch.randn(1,channels[i],i,i).to(self.opt.device))

    def esitmate_mask(self, image):
        label_map = faceParsing_demo(self.faceParsing_model, image, convert_to_seg12=True)
        
        return label_map
    
    @torch.no_grad()
    def extract_src_texture_vectors(self):
        if self.initial_label_map is not None:
            # wrap data 
            src = transforms.Compose([TO_TENSOR, NORMALIZE])(self.src_img)
            src = src.to(self.opt.device).float().unsqueeze(0)
            src_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(self.initial_label_map))
            src_mask = (src_mask*255).long().to(self.opt.device).unsqueeze(0)
            src_onehot = torch_utils.labelMap2OneHot(src_mask, num_cls = self.opt.num_seg_cls)

            # Extract the texture vectors using RGI
            self.src_texture_vectors, _ = self.net.get_style_vectors(src, src_onehot) 
    
    @torch.no_grad()
    def extract_ref_texture_vectors(self):
        if self.ref_label_map is not None:
            ref = transforms.Compose([TO_TENSOR, NORMALIZE])(self.ref_img)
            ref = ref.to(self.opt.device).float().unsqueeze(0)
            ref_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(self.ref_label_map))
            ref_mask = (ref_mask*255).long().to(self.opt.device).unsqueeze(0)
            ref_onehot = torch_utils.labelMap2OneHot(ref_mask, num_cls = self.opt.num_seg_cls)
            
            self.ref_texture_vectors, _ = self.net.get_style_vectors(ref, ref_onehot)


helper = DemoHelper()

# ================================ Callback Functions for Gradio Widgets ================================
def esitimate_init_mask_fn(image):
    label_map = helper.esitmate_mask(image)
    
    # cache the image & label_map
    helper.initial_label_map = label_map
    helper.initial_colored_map = label_map_to_colored_mask(helper.initial_label_map)
    helper.src_img = image
    

    # extract texture vectors of source image
    helper.extract_src_texture_vectors()

    colored_mask = label_map_to_colored_mask(label_map)

    return colored_mask, "✅ Load input image success!"

def esitimate_referece_mask_fn(image):
    label_map = helper.esitmate_mask(image)
    
    # cache the label_map
    helper.ref_label_map = label_map
    helper.ref_img = image

     # Extract texture vectors of reference image
    helper.extract_ref_texture_vectors()

    return "✅ Load reference image success!"


def edit_mask_fn(region_radio, edited_mask):
    if region_radio is None:
        return helper.initial_colored_map, "❌ Please choose the region you want to edit on, and try again."
    else:
        # mask_after_edit = edited_mask["image"]    
        mask = np.sum(edited_mask["mask"][:,:,0:3], axis=-1) != 0 # gradio return a RGBA mask
        comp_idx = COMP2INDEX[region_radio]
        
        label_map = colored_mask_to_label_map(helper.initial_colored_map)
        label_map[mask] =  comp_idx
        colored_mask_edited = label_map_to_colored_mask(label_map)
        
        return colored_mask_edited, "✅ Edit %s region success!"%region_radio

@torch.no_grad()
def face_shape_edit_fn(edited_mask):
    """According to original texture vector and edited mask, get the resulted face.
    """
    mask = edited_mask["image"]
    label_map = colored_mask_to_label_map(mask)

    # [bs,1,H,W] to one-hot format，i.e., [bs,#seg_cls,H,W]
    onehot = torch_utils.labelMap2OneHot(
        (TO_TENSOR(label_map)*255).long().to(helper.opt.device).unsqueeze(0),
        num_cls=helper.opt.num_seg_cls
    )
    
    style_codes = helper.net.cal_style_codes(helper.src_texture_vectors)
    generated, _, _ = helper.net.gen_img(torch.zeros(1,512,32,32).to(onehot.device), style_codes, onehot,
                                            randomize_noise=False, noise=helper.noise)

    res_img = torch_utils.tensor2im(generated[0])


    return res_img, "✅ Edit shape success!"

@torch.no_grad()
def face_texture_edit_fn(region_groups, alpha):
    """According to original texture vector and reference image, get the resulted face via texture vector mixing.
    """
    regions = region_groups
    if len(regions) == 0: 
        return helper.src_img, "❌ Please choose the region you want to mix, and try again."
    else: 
        mixed_texture_vectors = helper.src_texture_vectors.clone()

        for region in regions:
            idx = COMP2INDEX[region]
            mixed_texture_vectors[0,idx,:] = (1-alpha) * helper.src_texture_vectors[0,idx,:] + \
                                            alpha * helper.ref_texture_vectors[0,idx,:]
        
        mixed_style_codes = helper.net.cal_style_codes(mixed_texture_vectors)
        # [bs,1,H,W] to one-hot format，i.e., [bs,#seg_cls,H,W]
        onehot = torch_utils.labelMap2OneHot(
            (TO_TENSOR(helper.initial_label_map)*255).long().to(helper.opt.device).unsqueeze(0),
            num_cls=helper.opt.num_seg_cls
        )
        
        generated, _, _ = helper.net.gen_img(torch.zeros(1,512,32,32).to(onehot.device), mixed_style_codes, onehot,
                                                randomize_noise=False, noise=helper.noise)

        res_img = torch_utils.tensor2im(generated[0])


        return res_img, "✅ Edit %s region(s) success!"%" ".join(regions)
    


with gr.Blocks() as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        
        <img src="file/assets/cvpr_banner_homepage.svg" alt="CVPR2023" style="width:250px">

        <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
             Welcome to E4S demo page!
        </h1>
        <h2 style="font-weight: 450; font-size: 1rem; margin-top: 0.8rem">
        Zhian Liu<sup>1*</sup>, Maomao Li<sup>2*</sup>, 
        <a href="https://yzhang2016.github.io" style="color:blue;">Yong Zhang</a><sup>2*</sup>, 
        Cairong Wang</a><sup>3</sup>, 
        <a href="https://qzhang-cv.github.io" style="color:blue;">Qi Zhang</a><sup>2</sup>, 
        <a href="https://juewang725.github.io" style="color:blue;">Jue Wang</a><sup>2</sup>, 
        <a href="https://nieyongwei.net" style="color:blue;">Yongwei Nie</a><sup>1✉️</sup><br> 
        [<a href="https://arxiv.org/abs/2211.14068" style="color:red;">arXiv</a>] 
        [<a href="https://e4s2022.github.io" style="color:red;">Project page</a>] 
        [<a href="https://github.com/e4s2022/e4s" style="color:red;">GitHub</a>]
        </h2>
        <h3 style="font-weight: 450; font-size: 1rem; margin: 0rem">
        <sup>1</sup> South China University of Technology, <sup>2</sup>Tencent AI Lab, <sup>3</sup>Tsinghua Shenzhen International Graduate School
        </h3>
        *: equal contributions, &emsp; ✉️: corresponding author
        
        
        </div>
        """)
    with gr.Row():
        input_img = gr.Image(label = "input image", type="pil", shape=(1024, 1024))
        input_img.style(height=400, width=400)
        input_mask = gr.Image(label = "mask", source = "upload", tool = "sketch", shape=(512, 512))
        input_mask.style(height=400, width=400)
        
    with gr.Row():
        with gr.Tab("Shape editing"):
            region_radio = gr.Radio(COMP, value="hair", label="Facial regions", info="Which region(s) are you intersted in?")
            shape_edit_logging_text = gr.Textbox(label = "Operations logging:", value = "Ready to edit shape...", lines = 2, interactive=False)
            with gr.Row():
                edit_mask_btn = gr.Button("Confirm mask editing")
                face_shape_edit_btn = gr.Button("Get edited face")
                

        with gr.Tab("Texture editing"):
            region_groups = gr.CheckboxGroup(
                choices = COMP,
                label="Facial regions", 
                info="Which region(s) are you intersted in?",
            )
            with gr.Row():
                reference_img = gr.Image(label = "Reference image", type="pil", shape=(1024, 1024))
                reference_img.style(height=256, width=256)
                with gr.Column():
                    alpha = gr.Slider(0, 1, value=1.0, label="Editing extent", info="Choose betwen 0 and 1")
                    texture_edit_logging_text = gr.Textbox(label = "Operations logging:", value = "Ready to edit texture...", lines = 2, interactive=False)
                    face_texture_edit_btn = gr.Button("Get edited face")
        
        output_img = gr.Image(label = "result", type="pil", shape=(1024, 1024))
        output_img.style(height=400, width=400)

    # register callbacks
    input_img.change(fn=esitimate_init_mask_fn, inputs=[input_img], outputs=[input_mask, shape_edit_logging_text], queue=False)
    reference_img.change(fn=esitimate_referece_mask_fn, inputs=[reference_img], outputs=[texture_edit_logging_text], queue=False)
    edit_mask_btn.click(fn = edit_mask_fn, inputs=[region_radio, input_mask], outputs=[input_mask, shape_edit_logging_text])
    
    face_texture_edit_btn.click(fn = face_texture_edit_fn, inputs = [region_groups, alpha], outputs=[output_img, texture_edit_logging_text])
    face_shape_edit_btn.click(fn = face_shape_edit_fn, inputs=[input_mask] ,outputs=[output_img, shape_edit_logging_text])
    
    
        
if __name__ == '__main__':
    demo.launch()