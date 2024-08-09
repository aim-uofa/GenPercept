import torch
import numpy as np
import cv2
from PIL import Image
import os
import safetensors
from typing import Optional
from genpercept.pipeline_genpercept import GenPerceptPipeline
from genpercept.models import CustomUNet2DConditionModel
from diffusers import AutoencoderKL
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download
dependencies = ["torch", "numpy", "cv2", "PIL", "safetensors", "diffusers"]

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

class Predictor:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def infer_cv2(self, image, image_resolution=768):
        raw_image = HWC3(image)
        img = resize_image(raw_image, image_resolution)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.infer_pil(Image.fromarray(img))

    def infer_pil(self, image, image_resolution=768):
        with torch.no_grad():
            pipe_out = self.model(image,
                processing_res=image_resolution,
                match_input_res=True,
                batch_size=1,
                color_map="Spectral",
                show_progress_bar=True,
                mode='normal',
            )
            pred_normal = np.asarray(pipe_out.pred_colored)
            pred_normal = cv2.cvtColor(pred_normal, cv2.COLOR_RGB2BGR)
        return pred_normal

def GenPercept(local_dir: Optional[str] = None, device="cuda", repo_id = "guangkaixu/GenPercept"):
    unet_ckpt_path = hf_hub_download(repo_id=repo_id, filename='unet_normal_v1/diffusion_pytorch_model.safetensors', 
                      local_dir=local_dir)
    vae_ckpt_path = hf_hub_download(repo_id=repo_id, filename='vae/diffusion_pytorch_model.safetensors',
                      local_dir=local_dir)

    # Load UNet
    unet = CustomUNet2DConditionModel.from_config(repo_id, subfolder="unet_normal_v1")
    load_ckpt_unet = safetensors.torch.load_file(unet_ckpt_path)
    if not any('conv_out' in key for key in load_ckpt_unet.keys()):
        unet.conv_out = None
    if not any('conv_norm_out' in key for key in load_ckpt_unet.keys()):
        unet.conv_norm_out = None
    unet.load_state_dict(load_ckpt_unet)

    # Load VAE
    vae = AutoencoderKL.from_config(repo_id, subfolder="vae")
    load_ckpt_vae = safetensors.torch.load_file(vae_ckpt_path)
    if not any('decoder' in key for key in load_ckpt_vae.keys()):
        vae.decoder = None
    if not any('post_quant_conv' in key for key in load_ckpt_vae.keys()):
        vae.post_quant_conv = None
    vae.load_state_dict(load_ckpt_vae)

    # Load empty text embed
    empty_text_embed = torch.from_numpy(np.load('empty_text_embed.npy')).to(device, torch.float32)[None]

    genpercept_params_ckpt = dict(
        unet=unet,
        vae=vae,
        empty_text_embed=empty_text_embed,
        customized_head=None,
    )

    normal_predictor = GenPerceptPipeline(**genpercept_params_ckpt)
    normal_predictor = normal_predictor.to(device)

    return Predictor(normal_predictor, device)

def _test_run():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input image file")
    parser.add_argument("--output", "-o", type=str, required=True, help="output image file")
    parser.add_argument("--local_dir", type=str, help="use local model file")
    parser.add_argument("--pil", action="store_true", help="use PIL instead of OpenCV")
    args = parser.parse_args()

    predictor = torch.hub.load(".", "GenPercept", local_dir=args.local_dir,
                                source="local", trust_repo=True)

    if args.pil:
        from PIL import Image
        image = Image.open(args.input).convert("RGB")
        normal = predictor.infer_pil(image)
        Image.fromarray(normal).save(args.output)
    else:
        image = cv2.imread(args.input, cv2.IMREAD_COLOR)
        normal = predictor.infer_cv2(image)
        cv2.imwrite(args.output, normal)

if __name__ == "__main__":
    _test_run()