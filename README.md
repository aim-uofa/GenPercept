<div align="center">

<h1> What Matters When Repurposing Diffusion Models for General Dense Perception Tasks?</h1>

Former Title: "Diffusion Models Trained with Large Data Are Transferable Visual Models"

[Guangkai Xu](https://github.com/guangkaixu/), &nbsp; 
[Yongtao Ge](https://yongtaoge.github.io/), &nbsp; 
[Mingyu Liu](https://mingyulau.github.io/), &nbsp;
[Chengxiang Fan](https://leaf1170124460.github.io/), &nbsp;<br>
[Kangyang Xie](https://github.com/felix-ky), &nbsp;
[Zhiyue Zhao](https://github.com/ZhiyueZhau), &nbsp;
[Hao Chen](https://stan-haochen.github.io/), &nbsp;
[Chunhua Shen](https://cshen.github.io/), &nbsp;

Zhejiang University

### [HuggingFace (Space)](https://huggingface.co/spaces/guangkaixu/GenPercept) | [HuggingFace (Model)](https://huggingface.co/guangkaixu/genpercept-models) | [arXiv](https://arxiv.org/abs/2403.06090)

#### 🔥 Fine-tune diffusion models for perception tasks, and inference with only one step! ✈️

</div>

<div align="center">
<img width="800" alt="image" src="figs/pipeline.jpg">
</div>


##  📢 News
- 2024.10.25 Update GenPercept [Huggingface]((https://huggingface.co/spaces/guangkaixu/GenPercept)) App demo.
- 2024.10.24 Release latest training and inference code, which is armed with the [accelerate](https://github.com/huggingface/accelerate) library and based on [Marigold](https://github.com/prs-eth/marigold).
- 2024.10.24 Release [arXiv v3 paper](https://arxiv.org/abs/2403.06090v3). We reorganize the structure of the paper and offer more detailed analysis.
- 2024.4.30: Release checkpoint weights of surface normal and dichotomous image segmentation.
- 2024.4.7:  Add [HuggingFace](https://huggingface.co/spaces/guangkaixu/GenPercept) App demo. 
- 2024.4.6:  Release inference code and depth checkpoint weight of GenPercept in the [GitHub](https://github.com/aim-uofa/GenPercept) repo.
- 2024.3.15: Release [arXiv v2 paper](https://arxiv.org/abs/2403.06090v2), with supplementary material.
- 2024.3.10: Release [arXiv v1 paper](https://arxiv.org/abs/2403.06090v1).


## 📚 Download Resource Summary

 - Space-Huggingface demo: https://huggingface.co/spaces/guangkaixu/GenPercept.
 - Models-all (including ablation study): https://huggingface.co/guangkaixu/genpercept-exps.
 - Models-main-paper: https://huggingface.co/guangkaixu/genpercept-models.
 - Models-depth: https://huggingface.co/guangkaixu/genpercept-depth.
 - Models-normal: https://huggingface.co/guangkaixu/genpercept-normal.
 - Models-dis: https://huggingface.co/guangkaixu/genpercept-dis.
 - Models-matting: https://huggingface.co/guangkaixu/genpercept-matting.
 - Models-seg: https://huggingface.co/guangkaixu/genpercept-seg.
 - Models-disparity: https://huggingface.co/guangkaixu/genpercept-disparity.
 - Models-disparity-dpt-head: https://huggingface.co/guangkaixu/genpercept-disparity-dpt-head.
 - Datasets-input demo: https://huggingface.co/datasets/guangkaixu/genpercept-input-demo.
 - Datasets-evaluation data: https://huggingface.co/datasets/guangkaixu/genpercept_datasets_eval.
 - Datasets-evaluation results: https://huggingface.co/datasets/guangkaixu/genpercept-exps-eval.


##  🖥️ Dependencies

```bash
conda create -n genpercept python=3.10
conda activate genpercept
pip install -r requirements.txt
pip install -e .
```

## 🚀 Inference
### Using Command-line Scripts
Download the [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) and [our trained models](https://huggingface.co/guangkaixu/genpercept-models) from HuggingFace and put the checkpoints under ```./pretrained_weights/``` and ```./weights/```, respectively. You can download them with the script ```script/download_sd21.sh``` and ```script/download_weights.sh```, or download the weights of [depth](https://huggingface.co/guangkaixu/genpercept-depth), [normal](https://huggingface.co/guangkaixu/genpercept-normal), [Dichotomous Image Segmentation](https://huggingface.co/guangkaixu/genpercept-dis), [matting](https://huggingface.co/guangkaixu/genpercept-matting), [segmentation](https://huggingface.co/guangkaixu/genpercept-seg), [disparity](https://huggingface.co/guangkaixu/genpercept-disparity), [disparity_dpt_head](https://huggingface.co/guangkaixu/genpercept-disparity-dpt-head) seperately.

Then, place images in the ```./input/``` dictionary. We offer demo images in [Huggingface](guangkaixu/genpercept-input-demo), and you can also download with the script ```script/download_sample_data.sh```. Then, run inference with scripts as below.

```bash
# Depth
source script/infer/main_paper/inference_genpercept_depth.sh
# Normal
source script/infer/main_paper/inference_genpercept_normal.sh
# Dis
source script/infer/main_paper/inference_genpercept_dis.sh
# Matting
source script/infer/main_paper/inference_genpercept_matting.sh
# Seg
source script/infer/main_paper/inference_genpercept_seg.sh
# Disparity
source script/infer/main_paper/inference_genpercept_disparity.sh
# Disparity_dpt_head
source script/infer/main_paper/inference_genpercept_disparity_dpt_head.sh
```

If you would like to change the input folder path, unet path, and output path, input these parameters like:
```bash
# Assign a values
input_rgb_dir=...
unet=...
output_dir=...
# Take depth as example
source script/infer/main_paper/inference_genpercept_depth.sh $input_rgb_dir $unet $output_dir
```
For a general inference script, please see ```script/infer/inference_general.sh``` in detail.

***Thanks to our one-step perception paradigm, the inference process runs much faster. (Around 0.4s for each image on an A800 GPU card.)***


### Using torch.hub

TODO

<!-- GenPercept models can be easily used with torch.hub for quick integration into your Python projects. Here's how to use the models for normal estimation, depth estimation, and segmentation:
#### Normal Estimation
```python
import torch
import cv2
import numpy as np

# Load the normal predictor model from torch hub
normal_predictor = torch.hub.load("hugoycj/GenPercept-hub", "GenPercept_Normal", trust_repo=True)

# Load the input image using OpenCV
image = cv2.imread("path/to/your/image.jpg", cv2.IMREAD_COLOR)

# Use the model to infer the normal map from the input image
with torch.inference_mode():
    normal = normal_predictor.infer_cv2(image)

# Save the output normal map to a file
cv2.imwrite("output_normal_map.png", normal)
```

#### Depth Estimation
```python
import torch
import cv2

# Load the depth predictor model from torch hub
depth_predictor = torch.hub.load("hugoycj/GenPercept-hub", "GenPercept_Depth", trust_repo=True)

# Load the input image using OpenCV
image = cv2.imread("path/to/your/image.jpg", cv2.IMREAD_COLOR)

# Use the model to infer the depth map from the input image
with torch.inference_mode():
    depth = depth_predictor.infer_cv2(image)

# Save the output depth map to a file
cv2.imwrite("output_depth_map.png", depth)
```

#### Segmentation
```python
import torch
import cv2

# Load the segmentation predictor model from torch hub
seg_predictor = torch.hub.load("hugoycj/GenPercept-hub", "GenPercept_Segmentation", trust_repo=True)

# Load the input image using OpenCV
image = cv2.imread("path/to/your/image.jpg", cv2.IMREAD_COLOR)

# Use the model to infer the segmentation map from the input image
with torch.inference_mode():
    segmentation = seg_predictor.infer_cv2(image)

# Save the output segmentation map to a file
cv2.imwrite("output_segmentation_map.png", segmentation)
``` -->

## 🔥 Train

NOTE: We implement the training with the [accelerate](https://github.com/huggingface/accelerate) library, but find a worse training accuracy with multi gpus compared to one gpu, with the same training ```effective_batch_size``` and ```max_iter```. Your assistance in resolving this issue would be greatly appreciated. Thank you very much!

### Preparation

Datasets: TODO

Place training datasets unser ```datasets/```

Download the [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) from HuggingFace and put the checkpoints under ```./pretrained_weights/```. You can also download with the script ```script/download_sd21.sh```.


### Start Training

The reproduction training scripts in [arxiv v3 paper](https://arxiv.org/abs/2403.06090v3) is released in ```script/```, whose configs are stored in ```config/```. Models with ```max_train_batch_size > 2``` are trained on an H100 and ```max_train_batch_size <= 2``` on an RTX 4090. Run the train script:

```bash
# Take depth training of main paper as an example
source script/train_sd21_main_paper/sd21_train_accelerate_genpercept_1card_ensure_depth_bs8_per_accu_pixel_mse_ssi_grad_loss.sh
```

## 🎖️ Eval

### Preparation

1. Download [evaluation datasets](https://huggingface.co/datasets/guangkaixu/genpercept_eval/tree/main) and place them in ```datasets_eval```.
2. Download [our trained models](https://huggingface.co/guangkaixu/genpercept-exps) of main paper and ablation study in Section 3 of [arxiv v3 paper](https://arxiv.org/abs/2403.06090v3), and place them in ```weights/genpercept-exps```.

### Start Evaluation

The evaluation scripts are stored in ```script/eval_sd21```.

```bash
# Take "ensemble1 + step1" as an example
source script/eval_sd21/eval_ensemble1_step1/0_infer_eval_all.sh
```



## 📖 Recommanded Works

- Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation. [arXiv](https://github.com/prs-eth/marigold), [GitHub](https://github.com/prs-eth/marigold).
- GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image. [arXiv](https://arxiv.org/abs/2403.12013), [GitHub](https://github.com/fuxiao0719/GeoWizard).
- FrozenRecon: Pose-free 3D Scene Reconstruction with Frozen Depth Models. [arXiv](https://arxiv.org/abs/2308.05733), [GitHub](https://github.com/aim-uofa/FrozenRecon).


## 👍 Results in Paper

### Depth and Surface Normal

<div align="center">
<img width="800" alt="image" src="figs/demo_depth_normal_new.jpg">
</div>

### Dichotomous Image Segmentation

<div align="center">
<img width="800" alt="image" src="figs/demo_dis_new.jpg">
</div>

### Image Matting

<div align="center">
<img width="800" alt="image" src="figs/demo_matting.jpg">
</div>

### Image Segmentation

<div align="center">
<img width="800" alt="image" src="figs/demo_seg.jpg">
</div>


## 🎫 License

For non-commercial academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 
For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).


## 🎓 Citation
```
@article{xu2024diffusion,
  title={What Matters When Repurposing Diffusion Models for General Dense Perception Tasks?},
  author={Xu, Guangkai and Ge, Yongtao and Liu, Mingyu and Fan, Chengxiang and Xie, Kangyang and Zhao, Zhiyue and Chen, Hao and Shen, Chunhua},
  journal={arXiv preprint arXiv:2403.06090},
  year={2024}
}
```
