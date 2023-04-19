
from argparse import ArgumentParser
import glob
import os


parser = ArgumentParser()
parser.add_argument('--FFHQ_root', type=str, default="./data/FFHQ")  
args = parser.parse_args()


with open(os.path.join(args.FFHQ_root, "ffhq_list.txt"), "w") as f:
    all_files = []

    imgs_dirs = sorted(os.listdir(os.path.join(args.FFHQ_root, "images_1024")))
    for d in imgs_dirs:
        imgs = sorted(glob.glob(os.path.join(args.FFHQ_root, "images_1024", d, "*.png")))
        
        all_files.extend('\n'.join([os.path.join(d, os.path.basename(i)) for i in imgs]))
    
    f.writelines(all_files, )