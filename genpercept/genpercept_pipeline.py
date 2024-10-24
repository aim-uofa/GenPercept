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


import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)

import matplotlib.pyplot as plt
from genpercept.models.dpt_head import DPTNeckHeadForUnetAfterUpsampleIdentity

class GenPerceptOutput(BaseOutput):
    """
    Output class for GenPercept general perception pipeline.

    Args:
        pred_np (`np.ndarray`):
            Predicted result, with values in the range of [0, 1].
        pred_colored (`PIL.Image.Image`):
            Colorized result, with the shape of [3, H, W] and values in [0, 1].
    """

    pred_np: np.ndarray
    pred_colored: Union[None, Image.Image]

class GenPerceptPipeline(DiffusionPipeline):
    """
    Pipeline for general perception using GenPercept: https://github.com/aim-uofa/GenPercept.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the perception latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and results
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        default_denoising_steps: Optional[int] = 10,
        default_processing_resolution: Optional[int] = 768,
        rgb_blending = False,
        customized_head = None,
        genpercept_pipeline = True,
    ):
        super().__init__()

        self.genpercept_pipeline = genpercept_pipeline

        if self.genpercept_pipeline:
            default_denoising_steps = 1
            rgb_blending = True

        self.register_modules(
            unet=unet,
            customized_head=customized_head,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
            rgb_blending=rgb_blending,
        )

        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution
        self.rgb_blending = rgb_blending

        self.text_embed = None

        self.customized_head = customized_head

        if self.customized_head:
            assert self.rgb_blending and self.scheduler.beta_start == 1 and self.scheduler.beta_end == 1
            assert self.genpercept_pipeline

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        mode = None,
        fix_timesteps = None,
        prompt = "",
    ) -> GenPerceptOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize perception result to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and perception results. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized result generation):
                Colormap used to colorize the result.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `GenPerceptOutput`: Output class for GenPercept general perception pipeline, including:
            - **pred_np** (`np.ndarray`) Predicted result, with values in the range of [0, 1]
            - **pred_colored** (`PIL.Image.Image`) Colorized result, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
        """
        assert mode is not None, "mode of GenPerceptPipeline can be chosen from ['depth', 'normal', 'seg', 'matting', 'dis']."
        self.mode = mode

        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        if self.genpercept_pipeline:
            assert ensemble_size == 1
            assert denoising_steps == 1
        else:
            # Check if denoising step is reasonable
            self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Perception Inference -----------------
        # Batch repeated input image
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict results (batched)
        pipe_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            pipe_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
                fix_timesteps=fix_timesteps,
                prompt=prompt,
            )
            pipe_pred_ls.append(pipe_pred_raw.detach())
        pipe_preds = torch.concat(pipe_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            pipe_pred, _ = ensemble_depth(
                pipe_preds,
                scale_invariant=True,
                shift_invariant=True,
                max_res=50,
                **(ensemble_kwargs or {}),
            )
        else:
            pipe_pred = pipe_preds

        # Resize back to original resolution
        if match_input_res:
            pipe_pred = resize(
                pipe_pred,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )

        # Convert to numpy
        pipe_pred = pipe_pred.squeeze()
        pipe_pred = pipe_pred.cpu().numpy()

        # Clip output range
        pipe_pred = pipe_pred.clip(0, 1)

        # Colorize
        if color_map is not None:
            assert self.mode in ['depth', 'disparity']
            pred_colored = colorize_depth_maps(
                pipe_pred, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            pred_colored = (pred_colored * 255).astype(np.uint8)
            pred_colored_hwc = chw2hwc(pred_colored)
            pred_colored_img = Image.fromarray(pred_colored_hwc)
        else:
            pred_colored_img = (pipe_pred * 255.0).astype(np.uint8)
            if len(pred_colored_img.shape) == 3 and pred_colored_img.shape[0] == 3:
                pred_colored_img = np.transpose(pred_colored_img, (1, 2, 0))
            pred_colored_img = Image.fromarray(pred_colored_img)
        
        if len(pipe_pred.shape) == 3 and pipe_pred.shape[0] == 3:
            pipe_pred = np.transpose(pipe_pred, (1, 2, 0))

        return GenPerceptOutput(
            pred_np=pipe_pred,
            pred_colored=pred_colored_img,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_text(self, prompt):
        """
        Encode text embedding for empty prompt
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        fix_timesteps = None,
        prompt = "",
    ) -> torch.Tensor:
        """
        Perform an individual perception result without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted result.
        """
        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        if fix_timesteps:
            timesteps = torch.tensor([fix_timesteps]).long().repeat(self.scheduler.timesteps.shape[0]).to(device)
        else:
            timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        if not (self.rgb_blending or self.genpercept_pipeline):
            # Initial result (noise)
            pred_latent = torch.randn(
                rgb_latent.shape,
                device=device,
                dtype=self.dtype,
                generator=generator,
            )  # [B, 4, h, w]
        else:
            pred_latent = rgb_latent

        # Batched empty text embedding
        if self.text_embed is None:
            self.encode_text(prompt)
        batch_text_embed = self.text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)
        
        if not self.customized_head:
            for i, t in iterable:
                if self.genpercept_pipeline and i > 0:
                    assert ValueError, "GenPercept only forward once."

                if not (self.rgb_blending or self.genpercept_pipeline):
                    unet_input = torch.cat(
                        [rgb_latent, pred_latent], dim=1
                    )  # this order is important
                else:
                    unet_input = pred_latent

                # predict the noise residual
                noise_pred = self.unet(
                    unet_input, t, encoder_hidden_states=batch_text_embed
                ).sample  # [B, 4, h, w]

                # compute the previous noisy sample x_t -> x_t-1
                step_output = self.scheduler.step(
                    noise_pred, t, pred_latent, generator=generator
                )
                pred_latent = step_output.prev_sample

            pred_latent = step_output.pred_original_sample # NOTE: for GenPercept, it is equivalent to "pred_latent = - noise_pred"

            pred = self.decode_pred(pred_latent)

            # clip prediction
            pred = torch.clip(pred, -1.0, 1.0)
            # shift to [0, 1]
            pred = (pred + 1.0) / 2.0

        elif isinstance(self.customized_head, DPTNeckHeadForUnetAfterUpsampleIdentity):
            unet_input = pred_latent
            model_pred_output = self.unet(
                unet_input, timesteps, encoder_hidden_states=batch_text_embed, return_feature=True
            )  # [B, 4, h, w]
            unet_features = model_pred_output.multi_level_feats[::-1]
            pred = self.customized_head(hidden_states=unet_features).prediction[:, None]
            # shift to [0, 1]
            pred = (pred - pred.min()) / (pred.max() - pred.min())
        else:
            raise ValueError

        return pred

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.latent_scale_factor
        return rgb_latent

    def decode_pred(self, pred_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode pred latent into result.

        Args:
            pred_latent (`torch.Tensor`):
                pred latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded result.
        """
        # scale latent
        pred_latent = pred_latent / self.latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(pred_latent)
        stacked = self.vae.decoder(z)
        if self.mode in ['depth', 'matting', 'dis', 'disparity']:
            # mean of output channels
            stacked = stacked.mean(dim=1, keepdim=True)
        return stacked
