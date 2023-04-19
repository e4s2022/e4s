## Installation
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