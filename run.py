# --------------------------------------------------------
# What Matters When Repurposing Diffusion Models for General Dense Perception Tasks? (https://arxiv.org/abs/2403.06090)
# Github source: https://github.com/aim-uofa/GenPercept
# Copyright (c) 2024, Advanced Intelligent Machines (AIM)
# Licensed under The BSD 2-Clause License [see LICENSE for details]
# Author: Guangkai Xu (https://github.com/guangkaixu/)
# --------------------------------------------------------------------------
# This code is based on Marigold and diffusers codebases
# https://github.com/prs-eth/marigold
# https://github.com/huggingface/diffusers
# --------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/aim-uofa/GenPercept#%EF%B8%8F-citation
# More information about the method can be found at https://github.com/aim-uofa/GenPercept
# --------------------------------------------------------------------------


import argparse
import logging
import os
import os.path as osp
import shutil
from pathlib import Path


import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from genpercept import GenPerceptPipeline
from src.util.seeding import seed_all
from src.dataset import (
    BaseDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)

from src.customized_modules.ddim import DDIMSchedulerCustomized
from safetensors.torch import load_model, save_model, load_file
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from diffusers import UNet2DConditionModel
from peft import LoraConfig
from transformers import DPTConfig
from genpercept.models.dpt_head import DPTNeckHeadForUnetAfterUpsample, DPTNeckHeadForUnetAfterUpsampleIdentity
from diffusers import AutoencoderKL
from genpercept.models.custom_unet import CustomUNet2DConditionModel


EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

def get_image_paths(folder_path):
    return [str(file) for file in Path(folder_path).rglob('*') if file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}]

def _replace_unet_conv_in(unet):
    # replace the first layer to accept 8 in_channels
    _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = unet.conv_in.bias.clone()  # [320]
    _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
    # half the activation magnitude
    _weight *= 0.5
    # new conv_in channel
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = Conv2d(
        8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    logging.info("Unet conv_in layer is replaced")
    # replace config
    unet.config["in_channels"] = 8
    logging.info("Unet config is updated")
    return unet

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-v1-0",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. For the original (DDIM) version, it's recommended to use 10-50 steps, while for LCM 1-4 steps.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )

    # # depth map colormap
    # parser.add_argument(
    #     "--color_map",
    #     type=str,
    #     default=None,
    #     help="Colormap used to render depth predictions.",
    # )

    # other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for unseeded inference.",
    )

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
    parser.add_argument(
        "--archs",
        type=str,
        default='marigold',
        choices=['marigold', 'genpercept', 'rgb_blending'],
        help="Flag of running on Apple Silicon.",
    )
    parser.add_argument(
        "--unet",
        type=str,
        default=None,
        help="Unet checkpoint path or hub name.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="Scheduler path or hub name.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='depth',
        choices=['depth', 'normal', 'matting', 'dis', 'seg'],
        help="",
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=0,
        help="",
    )

    args = parser.parse_args()

    mode = args.mode

    if args.archs == 'genpercept':
        args.denoise_steps = 1
        args.ensemble_size = 1
    
    if args.mode == "depth":
        args.color_map = 'Spectral'
    else:
        args.color_map = None

    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15 and args.archs != 'genpercept':
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

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
    # rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    # rgb_filename_list = [
    #     f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    # ]
    rgb_filename_list = get_image_paths(input_rgb_dir)
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
        variant = "fp16"
        logging.info(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    # NOTE: deal with guangkaixu/genpercept-models. It cannot detect whether customized head is used or not.
    if 'genpercept-models' in args.unet:
        unet_model_subfolder = ""
        if 'unet_disparity_dpt_head_v2' in args.unet:
            args.load_decoder_ckpt = osp.dirname(args.unet)
        else:
            args.load_decoder_ckpt = None
    else:
        unet_model_subfolder = 'unet'
        args.load_decoder_ckpt = args.unet

    pre_loaded_dict = dict()
    if args.load_decoder_ckpt: # NOTE: path to the checkpoint folder does not contain 'vae' or 'customized_head'
        if 'dpt_head_identity' in os.listdir(args.load_decoder_ckpt):
            sub_dir = "dpt_head_identity" 
            dpt_config = DPTConfig.from_pretrained("hf_configs/dpt-sd2.1-unet-after-upsample-general")
            loaded_model = DPTNeckHeadForUnetAfterUpsampleIdentity(config=dpt_config)
            load_model(loaded_model, osp.join(args.load_decoder_ckpt, sub_dir, 'model.safetensors'))
            pre_loaded_dict['customized_head'] = loaded_model.to(dtype=dtype).to(device=device)
        elif 'dpt_head' in os.listdir(args.load_decoder_ckpt):
            sub_dir = "dpt_head" 
            dpt_config = DPTConfig.from_pretrained("hf_configs/dpt-sd2.1-unet-after-upsample-general")
            loaded_model = DPTNeckHeadForUnetAfterUpsample(config=dpt_config)
            load_model(loaded_model, osp.join(args.load_decoder_ckpt, sub_dir, 'model.safetensors'))
            pre_loaded_dict['customized_head'] = loaded_model.to(dtype=dtype).to(device=device)
        elif 'vae_decoder' in os.listdir(args.load_decoder_ckpt) and 'vae_post_quant_conv' in os.listdir(args.load_decoder_ckpt):
            vae = AutoencoderKL.from_pretrained(checkpoint_path, subfolder='vae')
            load_model(vae.decoder, osp.join(args.load_decoder_ckpt, 'vae_decoder', 'model.safetensors'))
            load_model(vae.post_quant_conv, osp.join(args.load_decoder_ckpt, 'vae_post_quant_conv', 'model.safetensors'))
            pre_loaded_dict['vae'] = vae.to(dtype=dtype).to(device=device)
    
    if args.unet:
        if 'customized_head' in pre_loaded_dict.keys():
            unet = CustomUNet2DConditionModel.from_pretrained(checkpoint_path, subfolder='unet')
            del unet.conv_out
            del unet.conv_norm_out
        else:
            unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder='unet')
        
        if (8 != unet.config["in_channels"]) and (args.archs == 'marigold'):
            unet = _replace_unet_conv_in(unet)

        if osp.exists(osp.join(args.unet, unet_model_subfolder, 'diffusion_pytorch_model.bin')):
            unet_ckpt_path = osp.join(args.unet, unet_model_subfolder, 'diffusion_pytorch_model.bin')
        elif osp.exists(osp.join(args.unet, unet_model_subfolder, 'diffusion_pytorch_model.safetensors')):
            unet_ckpt_path = osp.join(args.unet, unet_model_subfolder, 'diffusion_pytorch_model.safetensors')
        else:
            print('Warning!!! the saved checkpoint does not contain U-Net. Load U-Net from pretrained models...')
            unet_ckpt_path = osp.join(checkpoint_path, 'unet', 'diffusion_pytorch_model.safetensors')

        ckpt = load_file(unet_ckpt_path)
        if 'customized_head' in pre_loaded_dict.keys():
            ckpt_new = {}
            for key in ckpt:
                if 'conv_out' in key:
                    continue
                if 'conv_norm_out' in key:
                    continue
                ckpt_new[key] = ckpt[key]
        else:
            ckpt_new = ckpt
        
        if args.lora_rank > 0:
            unet_lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            # Add adapter and make sure the trainable params are in float32.
            unet.add_adapter(unet_lora_config)
            unet.requires_grad_(False)

        unet.load_state_dict(ckpt_new)
        pre_loaded_dict['unet'] = unet.to(dtype=dtype).to(device=device)
    else:
        unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder='unet')
    
    if args.archs == 'marigold' or args.archs == 'rgb_blending':
        if args.scheduler is not None:
            pre_loaded_dict['scheduler'] = DDIMSchedulerCustomized.from_pretrained(args.scheduler, subfolder='scheduler')

        genpercept_pipeline = False
        pipe: GenPerceptPipeline = GenPerceptPipeline.from_pretrained(
            checkpoint_path, variant=variant, torch_dtype=dtype, rgb_blending=(args.archs != 'marigold'), genpercept_pipeline=genpercept_pipeline, **pre_loaded_dict
        )

    elif args.archs == 'genpercept':
        pre_loaded_dict['scheduler'] = DDIMSchedulerCustomized.from_pretrained('hf_configs/scheduler_beta_1.0_1.0')

        genpercept_pipeline = True
        pipe: GenPerceptPipeline = GenPerceptPipeline.from_pretrained(
            checkpoint_path, variant=variant, torch_dtype=dtype, genpercept_pipeline=genpercept_pipeline, **pre_loaded_dict
        )
    else:
        raise NotImplementedError

    del pre_loaded_dict

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    pipe = pipe.to(device)

    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}; "
        f"color_map = {color_map}."
    )

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for rgb_path in tqdm(rgb_filename_list, desc="Estimating depth", leave=True):
            
            # Output directories
            rgb_rel_folder = (osp.normpath(osp.dirname(rgb_path)) + '/').split(osp.normpath(input_rgb_dir) + '/')[-1]
            output_dir_i = osp.join(output_dir, rgb_rel_folder)
            os.makedirs(output_dir_i, exist_ok=True)

            # Read input image
            input_image = Image.open(rgb_path)

            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            pipe_out = pipe(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                color_map=color_map,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
                mode=mode,
            )

            pred_np: np.ndarray = pipe_out.pred_np
            pred_colored: Image.Image = pipe_out.pred_colored

            # Save as npy
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_pred"
            npy_save_path = os.path.join(output_dir_i, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, pred_np)

            png_save_path = os.path.join(output_dir_i, f"{pred_name_base}.png")
            if os.path.exists(png_save_path):
                logging.warning(f"Existing file: '{png_save_path}' will be overwritten")

            if mode in ['depth']:
                # Save as 16-bit uint png
                pred_to_save = (pred_np * 65535.0).astype(np.uint16)
                Image.fromarray(pred_to_save).save(png_save_path, mode="I;16")
            else:
                pred_to_save = (pred_np * 255.0).astype(np.uint8)
                Image.fromarray(pred_to_save).save(png_save_path)

            if pred_colored is not None:
                # Colorize
                colored_save_path = os.path.join(
                    output_dir_i, f"{pred_name_base}_colored.png"
                )
                if os.path.exists(colored_save_path):
                    logging.warning(
                        f"Existing file: '{colored_save_path}' will be overwritten"
                    )
                pred_colored.save(colored_save_path)
            
            # save rgb images
            shutil.copyfile(rgb_path, osp.join(output_dir_i, osp.basename(rgb_path)))
