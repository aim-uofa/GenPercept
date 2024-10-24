# --------------------------------------------------------
# Diffusion Models Trained with Large Data Are Transferable Visual Models (https://arxiv.org/abs/2403.06090)
# Github source: https://github.com/aim-uofa/GenPercept
# Copyright (c) 2024 Zhejiang University
# Licensed under The CC0 1.0 License [see LICENSE for details]
# By Guangkai Xu
# Based on Marigold, diffusers codebases
# https://github.com/prs-eth/marigold
# https://github.com/huggingface/diffusers
# --------------------------------------------------------


import argparse
import os
import os.path as osp
from glob import glob
import logging
import cv2
import time
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset

import safetensors
from diffusers import UNet2DConditionModel, AutoencoderKL

from genpercept.util.seed_all import seed_all
from genpercept.util.image_util import ResizeHard
from genpercept.pipeline_genpercept import GenPerceptPipeline

from genpercept.models import CustomUNet2DConditionModel # DPTHead, 
from genpercept.models.dpt_head_elu import DPTNeckHeadForUnetAfterUpsample


EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="GenPercept inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path or hub name.",
    )
    
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )
    parser.add_argument(
        "--save_pcd",
        default=False,
        action="store_true",
        help="Save point cloud while evaluting depth.",
    )
    parser.add_argument(
        "--unet_ckpt_path",
        type=str,
        default=None,
        help="Checkpoint path for unet.",
    )
    parser.add_argument(
        "--vae_ckpt_path",
        type=str,
        default=None,
        help="Checkpoint path for vae.",
    )
    parser.add_argument(
        "--customized_head_name",
        type=str,
        default=None,
        help="Customized head to replace the VAE decoder",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['depth', 'seg', 'normal'],
        default="depth",
        help="inference mode.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, output label at resized operating resolution. Default: False.",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )

    # other settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Flag of running on Apple Silicon.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir

    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time
        seed = int(time.time())
        print('seed {}'.format(seed))

    seed_all(seed)

    # Output directories
    output_dir_rgb = os.path.join(output_dir, "rgb")
    output_dir_color = os.path.join(output_dir, "{}_colored".format(args.mode))
    output_dir_npy = os.path.join(output_dir, "{}_npy".format(args.mode))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    os.makedirs(output_dir_rgb, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{input_rgb_dir}'")
        exit(1)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32
    
    # unet
    unet_ckpt_path = args.unet_ckpt_path if args.unet_ckpt_path is not None else checkpoint_path
    unet = CustomUNet2DConditionModel.from_config(
        unet_ckpt_path, subfolder="unet", revision=args.non_ema_revision
    )
    try:
        load_ckpt_unet = safetensors.torch.load_file(osp.join(unet_ckpt_path, 'unet', 'diffusion_pytorch_model.safetensors'))
    except:
        load_ckpt_unet = safetensors.torch.load_file(osp.join(unet_ckpt_path, 'diffusion_pytorch_model.safetensors'))
    if not any('conv_out' in key for key in load_ckpt_unet.keys()):
        unet.conv_out = None
    if not any('conv_norm_out' in key for key in load_ckpt_unet.keys()):
        unet.conv_norm_out = None
    unet.load_state_dict(load_ckpt_unet)
    
    # vae
    vae_ckpt_path = args.vae_ckpt_path if args.vae_ckpt_path is not None else checkpoint_path
    vae = AutoencoderKL.from_config(
        vae_ckpt_path, subfolder="vae",
    )
    load_ckpt_vae = safetensors.torch.load_file(osp.join(vae_ckpt_path, 'vae', 'diffusion_pytorch_model.safetensors'))
    if not any('decoder' in key for key in load_ckpt_vae.keys()):
        vae.decoder = None
    if not any('post_quant_conv' in key for key in load_ckpt_vae.keys()):
        vae.post_quant_conv = None
    vae.load_state_dict(load_ckpt_vae)
    
    # customized head
    customized_head = None
    if args.customized_head_name is not None:
        if args.customized_head_name == 'dpt_head':
            cfgs = "configs_hf/dpt-sd2.1-unet-after-upsample"
            customized_head = DPTNeckHeadForUnetAfterUpsample.from_pretrained(checkpoint_path, subfolder="dpt_head")
        else:
            raise NotImplementedError
    
        customized_head = customized_head.to(device)
    
    empty_text_embed = torch.from_numpy(np.load("empty_text_embed.npy")).to(device, dtype)[None] # [1, 77, 1024]

    genpercept_params_ckpt = dict(
        unet=unet,
        vae=vae,
        empty_text_embed=empty_text_embed,
        customized_head=customized_head,
    )

    pipe = GenPerceptPipeline(**genpercept_params_ckpt)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers


    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for rgb_path in tqdm(rgb_filename_list, desc="Estimating {}".format(args.mode), leave=True):
            # Read input image
            input_image = Image.open(rgb_path).convert("RGB")

            pipe_out = pipe(
                input_image,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                color_map=color_map,
                show_progress_bar=True,
                mode=args.mode,
            )

            if args.mode == 'depth':

                depth_pred = pipe_out.pred_np
                depth_colored = pipe_out.pred_colored

                # Save as npy
                rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
                pred_name_base = rgb_name_base + "_depth_pred"
                npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
                if os.path.exists(npy_save_path):
                    logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
                np.save(npy_save_path, depth_pred)

                # Colorize
                colored_save_path = os.path.join(
                    output_dir_color, f"{pred_name_base}_colored.png"
                )
                if os.path.exists(colored_save_path):
                    logging.warning(
                        f"Existing file: '{colored_save_path}' will be overwritten"
                    )
                depth_colored.save(colored_save_path)

                output_save_path_rgb = os.path.join(
                    output_dir_rgb, f"{rgb_name_base}.png"
                )
                try:
                    input_image.save(output_save_path_rgb)
                except:
                    cv2.imwrite(output_save_path_rgb, rgb_img[:, :, ::-1])

            elif args.mode == 'seg':
                seg_colored: Image.Image = pipe_out.pred_colored
                rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
                pred_name_base = rgb_name_base + "_seg_pred"
                # Colorize
                colored_save_path = os.path.join(
                    output_dir_color, f"{pred_name_base}_colored.png"
                )
                if os.path.exists(colored_save_path):
                    logging.warning(
                        f"Existing file: '{colored_save_path}' will be overwritten"
                    )

                # seg_res = Image.fromarray(np.concatenate([np.array(input_image), np.array(seg_colored)],axis=1))
                # seg_res.save(colored_save_path)
                seg_colored.save(colored_save_path)
            elif args.mode == 'normal':
                normal_colored: Image.Image = pipe_out.pred_colored
                rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
                pred_name_base = rgb_name_base + "_normal_pred_with_image"
                # Colorize
                colored_save_path = os.path.join(
                    output_dir_color, f"{pred_name_base}_colored.png"
                )
                if os.path.exists(colored_save_path):
                    logging.warning(
                        f"Existing file: '{colored_save_path}' will be overwritten"
                    )

                normal_colored.save(colored_save_path)
                output_save_path_rgb = os.path.join(
                    output_dir_rgb, f"{rgb_name_base}.png"
                )
                input_image.save(output_save_path_rgb)

            else:
                raise NotImplementedError

    print('Finished inference~')