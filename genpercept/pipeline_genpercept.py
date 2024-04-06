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

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from PIL import Image
from typing import List, Dict, Union
from torch.utils.data import DataLoader, TensorDataset

from diffusers import (
    DiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.utils import BaseOutput

from .util.image_util import chw2hwc, colorize_depth_maps, resize_max_res, norm_to_rgb, resize_res
from .util.batchsize import find_batch_size

class GenPerceptOutput(BaseOutput):

    pred_np: np.ndarray
    pred_colored: Image.Image

class GenPerceptPipeline(DiffusionPipeline):

    vae_scale_factor = 0.18215
    task_infos = {
        'depth':    dict(task_channel_num=1, interpolate='bilinear', ),
        'seg':      dict(task_channel_num=3, interpolate='nearest', ),
        'sr':       dict(task_channel_num=3, interpolate='nearest', ),
        'normal':   dict(task_channel_num=3, interpolate='bilinear', ),
    }

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        customized_head=None,
        empty_text_embed=None,
    ):
        super().__init__()

        self.empty_text_embed = empty_text_embed

        # register
        register_dict = dict(
            unet=unet,
            vae=vae,
            customized_head=customized_head,
        )
        self.register_modules(**register_dict)
    
    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        mode: str = 'depth',
        resize_hard = False,
        processing_res: int = 768,
        match_input_res: bool = True,
        batch_size: int = 0,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
    ) -> GenPerceptOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (Image):
                Input RGB (or gray-scale) image.
            processing_res (int, optional):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
                Defaults to 768.
            match_input_res (bool, optional):
                Resize depth prediction to match input resolution.
                Only valid if `limit_input_res` is not None.
                Defaults to True.
            batch_size (int, optional):
                Inference batch size.
                If set to 0, the script will automatically decide the proper batch size.
                Defaults to 0.
            show_progress_bar (bool, optional):
                Display a progress bar of diffusion denoising.
                Defaults to True.
            color_map (str, optional):
                Colormap used to colorize the depth map.
                Defaults to "Spectral".
        Returns:
            `GenPerceptOutput`
        """

        device = self.device

        task_channel_num = self.task_infos[mode]['task_channel_num']

        if not match_input_res:
            assert (
                processing_res is not None
            ), "Value error: `resize_output_back` is only valid with "
        assert processing_res >= 0

        # ----------------- Image Preprocess -----------------

        if type(input_image) == torch.Tensor: # [B, 3, H, W]            
            rgb_norm = input_image.to(device)
            input_size = input_image.shape[2:]
            bs_imgs = rgb_norm.shape[0]
            assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
            rgb_norm = rgb_norm.to(self.dtype)
        else:
            # if len(rgb_paths) > 0 and 'kitti' in rgb_paths[0]:
            #     # kb crop
            #     height = input_image.size[1]
            #     width = input_image.size[0]
            #     top_margin = int(height - 352)
            #     left_margin = int((width - 1216) / 2)
            #     input_image = input_image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # TODO: check the kitti evaluation resolution here.
            input_size = (input_image.size[1], input_image.size[0])
            # Resize image
            if processing_res > 0:
                if resize_hard:
                    input_image = resize_res(
                        input_image, max_edge_resolution=processing_res
                    )
                else:
                    input_image = resize_max_res(
                        input_image, max_edge_resolution=processing_res
                    )
            input_image = input_image.convert("RGB")
            image = np.asarray(input_image)

            # Normalize rgb values
            rgb = np.transpose(image, (2, 0, 1))  # [H, W, rgb] -> [rgb, H, W]
            rgb_norm = rgb / 255.0 * 2.0 - 1.0
            rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
            rgb_norm = rgb_norm[None].to(device)
            assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
            bs_imgs = 1

        # ----------------- Predicting depth -----------------

        single_rgb_dataset = TensorDataset(rgb_norm)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=1, 
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        pred_list = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        
        for batch in iterable:
            (batched_img, ) = batch
            pred = self.single_infer(
                rgb_in=batched_img,
                mode=mode,
            )
            pred_list.append(pred.detach().clone())
        preds = torch.concat(pred_list, axis=0).squeeze() # [bs_imgs, task_channel_num, H, W]
        preds = preds.view(bs_imgs, task_channel_num, preds.shape[-2], preds.shape[-1])
        
        if match_input_res:
            preds = F.interpolate(preds, input_size, mode=self.task_infos[mode]['interpolate'])

        # ----------------- Post processing -----------------
        if mode == 'depth':
            if len(preds.shape) == 4:
                preds = preds[:, 0] # [bs_imgs, H, W]
            # Scale prediction to [0, 1]
            min_d = preds.view(bs_imgs, -1).min(dim=1)[0]
            max_d = preds.view(bs_imgs, -1).max(dim=1)[0]
            preds = (preds - min_d[:, None, None]) / (max_d[:, None, None] - min_d[:, None, None])
            preds = preds.cpu().numpy().astype(np.float32)
            # Colorize
            pred_colored_img_list = []
            for i in range(bs_imgs):
                pred_colored_chw = colorize_depth_maps(
                    preds[i], 0, 1, cmap=color_map
                ).squeeze()  # [3, H, W], value in (0, 1)
                pred_colored_chw = (pred_colored_chw * 255).astype(np.uint8)
                pred_colored_hwc = chw2hwc(pred_colored_chw)
                pred_colored_img = Image.fromarray(pred_colored_hwc)
                pred_colored_img_list.append(pred_colored_img)

            return GenPerceptOutput(
                pred_np=np.squeeze(preds),
                pred_colored=pred_colored_img_list[0] if len(pred_colored_img_list) == 1 else pred_colored_img_list,
            )    

        elif mode == 'seg' or mode == 'sr':
            if not self.customized_head:
                # shift to [0, 1]
                preds = (preds + 1.0) / 2.0 
                # shift to [0, 255]
                preds = preds * 255
                # Clip output range
                preds = preds.clip(0, 255).cpu().numpy().astype(np.uint8)
            else:
                raise NotImplementedError

            pred_colored_img_list = []
            for i in range(preds.shape[0]):
                pred_colored_hwc = chw2hwc(preds[i])
                pred_colored_img = Image.fromarray(pred_colored_hwc)
                pred_colored_img_list.append(pred_colored_img)

            return GenPerceptOutput(
                pred_np=np.squeeze(preds),
                pred_colored=pred_colored_img_list[0] if len(pred_colored_img_list) == 1 else pred_colored_img_list,
            )

        elif mode == 'normal':
            if not self.customized_head:
                preds = preds.clip(-1, 1).cpu().numpy() # [-1, 1]
            else:
                raise NotImplementedError

            pred_colored_img_list = []
            for i in range(preds.shape[0]):
                pred_colored_chw = norm_to_rgb(preds[i])
                pred_colored_hwc = chw2hwc(pred_colored_chw)
                normal_colored_img_i = Image.fromarray(pred_colored_hwc)
                pred_colored_img_list.append(normal_colored_img_i)

            return GenPerceptOutput(
                pred_np=np.squeeze(preds),
                pred_colored=pred_colored_img_list[0] if len(pred_colored_img_list) == 1 else pred_colored_img_list,
            )

        else:
            raise NotImplementedError

    @torch.no_grad()
    def single_infer(
        self, 
        rgb_in: torch.Tensor, 
        mode: str = 'depth',
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (torch.Tensor):
                Input RGB image.
            num_inference_steps (int):
                Number of diffusion denoising steps (DDIM) during inference.
            show_pbar (bool):
                Display a progress bar of diffusion denoising.

        Returns:
            torch.Tensor: Predicted depth map.
        """
        device = rgb_in.device
        bs_imgs = rgb_in.shape[0]
        timesteps = torch.tensor([1]).long().repeat(bs_imgs).to(device)

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        batch_embed = self.empty_text_embed
        batch_embed = batch_embed.repeat((rgb_latent.shape[0], 1, 1)).to(device)   # [bs_imgs, 77, 1024]

        # Forward!
        if self.customized_head:
            unet_features = self.unet(rgb_latent, timesteps, encoder_hidden_states=batch_embed, return_feature_only=True)[0][::-1]
            pred = self.customized_head(unet_features)
        else:
            unet_output = self.unet(
                rgb_latent, timesteps, encoder_hidden_states=batch_embed
            )  # [bs_imgs, 4, h, w]
            unet_pred = unet_output.sample
            pred_latent = - unet_pred
            pred_latent.to(device)
            pred = self.decode_pred(pred_latent)
            if mode == 'depth':
                # mean of output channels
                pred = pred.mean(dim=1, keepdim=True)
            # clip prediction
            pred = torch.clip(pred, -1.0, 1.0)
        return pred


    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (torch.Tensor):
                Input RGB image to be encoded.

        Returns:
            torch.Tensor: Image latent
        """
        try:
            # encode
            h_temp = self.vae.encoder(rgb_in)
            moments = self.vae.quant_conv(h_temp)
        except:
            # encode
            h_temp = self.vae.encoder(rgb_in.float())
            moments = self.vae.quant_conv(h_temp.float())
            
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.vae_scale_factor
        return rgb_latent

    def decode_pred(self, pred_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode pred latent into pred label.

        Args:
            pred_latent (torch.Tensor):
                prediction latent to be decoded.

        Returns:
            torch.Tensor: Decoded prediction label.
        """
        # scale latent
        pred_latent = pred_latent / self.vae_scale_factor
        # decode
        z = self.vae.post_quant_conv(pred_latent)
        pred = self.vae.decoder(z)
        
        return pred
    