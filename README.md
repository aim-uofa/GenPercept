<div align="center">

<h1> GenPercept: Diffusion Models Trained with Large Data Are Transferable Visual Models </h1>

[Guangkai Xu](https://github.com/guangkaixu/), &nbsp; 
[Yongtao Ge](https://yongtaoge.github.io/), &nbsp; 
[Mingyu Liu](https://mingyulau.github.io/), &nbsp;
[Chengxiang Fan](https://leaf1170124460.github.io/), &nbsp;
[Kangyang Xie](https://github.com/felix-ky), &nbsp;
[Zhiyue Zhao](https://github.com/ZhiyueZhau), &nbsp;
[Hao Chen](https://stan-haochen.github.io/), &nbsp;
[Chunhua Shen](https://cshen.github.io/), &nbsp;

Zhejiang University

### [HuggingFace](https://huggingface.co/spaces/guangkaixu/GenPercept) | [arXiv](https://arxiv.org/abs/2403.06090)

#### 🔥 Fine-tune diffusion models for perception tasks, and inference with only one step! ✈️

</div>

<div align="center">
<img width="800" alt="image" src="figs/pipeline.png">
</div>

##  Dependencies

```bash
conda create -n genpercept python=3.10
conda activate genpercept
pip install -r requirements.txt
pip install -e .
```

## Inference

Download the pre-trained depth model ```depth_v1.zip``` from [BaiduNetDisk](https://pan.baidu.com/s/11JUKU3W1rUqQiQ649Ieb7A?pwd=z938) (Extract code: z938) or [Rec Cloud Disk](https://rec.ustc.edu.cn/share/0ac5fb60-f4be-11ee-a23c-29229a5c4bc4). Put the package under ```./weights/``` and unzip it, the checkpoint will be stored under ```./weights/depth_v1/```.

Then, place images in the ```./input/``` dictionary, and run the following script. The output depth will be saved in ```./output/```.
```bash
source scripts/inference_depth.sh
```

Thanks to our one-step perception paradigm, the inference process runs much faster. (Around 0.4s for each image on an A800 GPU card.)

## Recommanded Works

- Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation. [arXiv](https://github.com/prs-eth/marigold), [GitHub](https://github.com/prs-eth/marigold).
- GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image. [arXiv](https://arxiv.org/abs/2403.12013), [GitHub](https://github.com/fuxiao0719/GeoWizard).
- FrozenRecon: Pose-free 3D Scene Reconstruction with Frozen Depth Models. [arXiv](https://arxiv.org/abs/2308.05733), [GitHub](https://github.com/aim-uofa/FrozenRecon).

## Results in Paper

### Depth and Surface Normal

<div align="center">
<img width="800" alt="image" src="figs/demo_depth_normal.png">
</div>

### Dichotomous Image Segmentation

<div align="center">
<img width="400" alt="image" src="figs/demo_dis.png">
</div>

### Image Matting

<div align="center">
<img width="800" alt="image" src="figs/demo_matting.png">
</div>

### Human Pose Estimation

<div align="center">
<img width="800" alt="image" src="figs/demo_keypoint.png">
</div>

## 🎫 License

For non-commercial use, this code is released under the [LICENSE](LICENSE).
For commercial use, please contact Chunhua Shen.

## 🖊️ Citation
```
@article{xu2024diffusion,
  title={Diffusion Models Trained with Large Data Are Transferable Visual Models},
  author={Xu, Guangkai and Ge, Yongtao and Liu, Mingyu and Fan, Chengxiang and Xie, Kangyang and Zhao, Zhiyue and Chen, Hao and Shen, Chunhua},
  journal={arXiv preprint arXiv:2403.06090},
  year={2024}
}
```
