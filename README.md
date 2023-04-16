# E4S: Fine-grained Face Swapping via Regional GAN Inversion, CVPR 2023

<a href='https://arxiv.org/abs/2211.14068'><img src='https://img.shields.io/badge/ArXiv-2211.14068-red'></a> &nbsp;&nbsp;&nbsp;<a href='https://e4s2022.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) &nbsp;&nbsp;&nbsp;


<a href="#">Zhian Liu<sup>1*</sup></a>&emsp;<a href="#">Maomao Li<sup>2*</sup></a>&emsp;<a href="https://yzhang2016.github.io">Yong Zhang<sup>2*</sup></a>&emsp;<a href="#">Cairong Wang<sup>3</sup></a>&emsp;<a href="https://qzhang-cv.github.io/">Qi Zhang<sup>2</sup></a>&emsp;<a href="https://juewang725.github.io/">Jue Wang<sup>2</sup></a>&emsp;<a href="https://nieyongwei.net/">Yongwei Nie<sup>1✉️</sup></a>

<sup>1</sup>South China University of Technology &emsp;
<sup>2</sup>Tencent AI Lab &emsp;
<sup>3</sup>Tsinghua Shenzhen International Graduate School<br>
*: equal contributions, &emsp; ✉️: corresponding author


![pipeline](./assets/e4s_pipeline.png)

<b>TL;DR: A face swapping method from fine-grained face editing perspective, realized by texture and shape extraction and swapping for each facial region.</b>


## 🧑‍💻 Changelog

  - __[2023.04.11]__: Add face swapping inference demo (continue updating). 

  - __[2023.03.29]__: E4S repository initialized.

  - __[2023.02.28]__: E4S has been accepted by CVPR 2023!

___

# Usage
## <details><summary>1. Installation</summary>

### 1.1 Env
The environment is tested with Ubuntu 20.04 and Python 3.8, with NVIDIA GPU plus CUDA enabled. [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://conda.io/miniconda.html) is recommended to install the running environment. All the packages dependencies can be found in `e4s_env.yaml`, and it's convinient to create a conda environment via `conda env create -f e4s_env.yaml` command.

> 💡 Hint: If you find some problems when installing *dlib*, please consider to install it from conda forge or build it manually.

### 1.2 pre-trained model
We provide a pre-trained RGI model that was trained on FFHQ dataset for 300K iterations, please fetch the model from this [Google Drive link](https://drive.google.com/file/d/1cyJTYRO5G4kcugAcgSJ7cMsE96GzV_hq/view?usp=share_link) and place it in the `pretrained_ckpts/e4s` folder.


### 1.3 Other dependencies
- face-parsing.PyTorch: [repo](https://github.com/zllrunning/face-parsing.PyTorch)

Please download the pre-trained model [here](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812), and place it in the `pretrained_ckpts/face_parsing` folder.

> 💡 Hint: The following FaceVid2Vid and GPEN are only applied for face swapping. If noly face editing is needed, just skip to Section 2 directly.

- FaceVid2Vid: [paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjv_uua1Iz-AhU8DEQIHSpEBOwQFnoECA4QAQ&url=https%3A%2F%2Farxiv.org%2Fabs%2F2011.15126&usg=AOvVaw0V7kwcY9EHwMhhlodsD397) | [unofficial-repo](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis)

This face reenactment model is applied to drive source face to show similar pose and expression as the target. Currently, we use [zhanglonghao's impl.](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis) of FaceVid2Vid, where the pre-trained model can be downloaded [here (Vox-256-New)](https://www.mediafire.com/folder/fcvtkn21j57bb/TalkingHead_Update). Similarly, please put it in the `pretrained_ckpts/facevid2vid` folder.

- GPEN: [paper](https://arxiv.org/abs/2105.06070) | [repo](https://github.com/yangxy/GPEN)

A face restoration model ([GPEN](https://github.com/yangxy/GPEN)) is used to improve the resolution of the intermediate driven face. You need to download some pre-trained models as follows:

| Model | download link |
| - | - |
| RetinaFace-R50 | https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth, for face detection |
| RealESRNet_x4 | https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/realesrnet_x4.pth, for x4 super resolution|
| GPEN-BFR-512 | https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth, GEPN pre-trained model |
| ParseNet | https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/ParseNet-latest.pth, for face parsing |

Place these checkpoint files into `pretrained_ckpts/gpen` folder. Alternatively, you can execute the following script to fetch them automatically:
```sh
cd pretrained_ckpts/gpen
sh ./fetch_gpen_models.sh
```
</details>


## <details><summary>2. Inference Demo</summary>

### 2.1 face swapping 
#### Face swapping in defult settings:
```sh
python scripts/face_swap.py --source=example/input/faceswap/source.jpg --target=example/input/faceswap/target.jpg
```
The reuslts will be saved to `example/output/faceswap` folder. Left to right: source, target, swapped face

<img src="./example/input/faceswap/source.jpg" width="256" height="256"><img src="./example/input/faceswap/target.jpg" width="256" height="256"><img src="./example/output/faceswap/swap_res.png" width="256" height="256">


You can optionally provide the face parsing result of the target image via `--target_mask` arg, and turn on the `--verbose=True` for detailed visulize. The results will be saved in the `--output_dir` folder (default to `example/output/faceswap`). 
```sh
python scripts/face_swap.py \
      --source=./example/input/faceswap/source.jpg \
      --target=./example/input/faceswap/target.jpg \
      --target_mask=./example/input/faceswap/target_mask.png \
      --verbose=True
```
For more information and supported args, run `python scripts/face_swap.py -h` for help.

### 2.2 face editing 
For texture related editting or interpolation, run 
```sh
python scripts/face_edit.py \
      --source=./example/input/faceedit/source.jpg \
      --reference=./example/input/faceedit/reference.jpg \
      --region hair eyes \
      --alpha=1
```

The reuslts will be saved to `example/output/faceedit` folder. 


<img src="./assets/gradio_UI.jpg" >

For shape related editing, we provide an interactive editing demo that was build upon graido, just run `python demo/gradio_demo.py`.


TODO: 
- [ ] Share the gradio demo on Huggingface.
- [ ] Privide the optimization script for better results.

</details>

## <details><summary>3. Train </summary>
If you plan to train the model from scratch, you will need to do a bit more stuffs. Machine with multiple GPUs is recommanded for the training.

### 3.1 dataset

### 3.2 pre-trained models

- StyleGANv2: [paper](https://arxiv.org/abs/1912.04958) | [code](https://github.com/rosinality/stylegan2-pytorch)

Please download the pre-trained ckpt(364M) [here](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view), and put it in the `pretrained_ckpts/stylegan2` folder.

- Auxiliary model

We utilitize a pre-trained IR-SE50 model during training to calculate the identity loss, which is taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) repo. Please download it [here](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view)  accordingly from the following table and put them in the `pretrained_ckpts/auxiliary` folder.

<!-- | Model    | Description |
| -        | -           |
| February | $80     |
| March    | $420    | -->

### 3.3 training script
</details>


## 🔗 Citation
If you find our work useful in your research, please consider citing:
```
@article{liu2022fine,
  title={Fine-Grained Face Swapping via Regional GAN Inversion},
  author={Liu, Zhian and Li, Maomao and Zhang, Yong and Wang, Cairong and Zhang, Qi and Wang, Jue and Nie, Yongwei},
  journal={arXiv preprint arXiv:2211.14068},
  year={2022}
}
```


## 🌟 Ackowledgements

Code borrows heavily from [PSP](https://github.com/eladrich/pixel2style2pixel), [SEAN](https://github.com/ZPdesu/SEAN). We thank the authors for sharing their wonderful codebase.

### Related repositories:
