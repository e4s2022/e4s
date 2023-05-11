from argparse import ArgumentParser
import glob
import os
from PIL import Image
from tqdm import tqdm
from src.pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps


parser = ArgumentParser()
parser.add_argument('--faceParser_name', default='default', type=str, help='face parser name, [ default | segnext] is currently supported.')
parser.add_argument('--faceParsing_ckpt', type=str, default="./pretrained_ckpts/face_parsing/79999_iter.pth")  
parser.add_argument('--segnext_config', default='', type=str, help='Path to pre-trained SegNeXt faceParser configuration file, '
                                                                    'this option is valid when --faceParsing_ckpt=segenext')
		
parser.add_argument('--FFHQ_root', type=str, default="./data/FFHQ")  
parser.add_argument('--save_vis', action='store_true')
parser.add_argument('--seg12', action='store_true')

args = parser.parse_args()

mask_save_dir = os.path.join(args.FFHQ_root, "BiSeNet_mask")
os.makedirs(mask_save_dir, exist_ok=True)
if args.save_vis:
    mask_vis_save_dir = os.path.join(args.FFHQ_root, "BiSeNet_mask_vis")
    os.makedirs(mask_vis_save_dir, exist_ok=True)

faceParsing_model = init_faceParsing_pretrained_model(args.faceParser_name, args.faceParsing_ckpt, args.segnext_config)    
imgs_dirs = sorted(os.listdir(os.path.join(args.FFHQ_root, "images_1024")))
for d in imgs_dirs:
    print("Esitmating %s directory"%d)
    
    os.makedirs(os.path.join(mask_save_dir, d), exist_ok=True)
    if args.save_vis:
        os.makedirs(os.path.join(mask_vis_save_dir, d), exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(args.FFHQ_root, "images_1024", d, "*.png")))
    for img in tqdm(imgs, total=len(imgs)):
        pil_im = Image.open(img).convert("RGB")
        mask = faceParsing_demo(faceParsing_model, pil_im, convert_to_seg12=args.seg12, model_name=args.faceParser_name)

        Image.fromarray(mask).save(os.path.join(mask_save_dir, d, os.path.basename(img)))
        if args.save_vis:
            mask_vis = vis_parsing_maps(pil_im, mask)
            Image.fromarray(mask_vis).save(os.path.join(mask_vis_save_dir, d, os.path.basename(img)))