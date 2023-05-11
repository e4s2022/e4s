from argparse import ArgumentParser


class EditOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--num_seg_cls', type=int, default=12,help='Segmentation mask class number')
		self.parser.add_argument('--regions', type=str, nargs = '+', help='The facial component to operate on')
		self.parser.add_argument('--remaining_layer_idx', type=int, default=13, help='mask-guided style injection, i.e., K in paper')
		self.parser.add_argument('--out_size', type=int, default=1024, help='output image size') 
		self.parser.add_argument('--n_styles', default=18, type=int, help='StyleGAN层数')
		self.parser.add_argument('--fsencoder_type', type=str, default="psp", help='FS Encode type')
		self.parser.add_argument('--train_G', default=False, type=bool, help='Whether to train the model')
		
		self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
		self.parser.add_argument('--checkpoint_path', default='./pretrained_ckpts/e4s/iteration_300000.pt', type=str, help='Path to pre-trained E4S model checkpoint')
		self.parser.add_argument('--faceParser_name', default='default', type=str, help='face parser name, [ default | segnext] is currently supported.')
		self.parser.add_argument('--faceParsing_ckpt', default='./pretrained_ckpts/face_parsing/79999_iter.pth', type=str, help='Path to pre-trained faceParsing model checkpoint')
		self.parser.add_argument('--segnext_config', default='', type=str, help='Path to pre-trained SegNeXt faceParser configuration file, this option is valid when --faceParsing_ckpt=segenext')
		self.parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU(s) to use')

		self.parser.add_argument('--start_from_latent_avg', action='store_true',default=True, help='Whether to add average latent vector to generate codes from encoder.')
		self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')
        
		self.parser.add_argument('--source', type=str, default="example/input/faceedit/source.jpg",help='Path to the source image')
		self.parser.add_argument('--reference', type=str, default="example/input/faceedit/reference.jpg",help='Path to the reference image')
		self.parser.add_argument('--output_dir', default="./example/output/faceedit", type=str, help='Path to save dir')  
		self.parser.add_argument('--alpha', type=float, default=1.0, help='interplokation extent') 
		
        
  
	def parse(self):
		opts = self.parser.parse_args()
		return opts
