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
import os
import os.path as osp
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
from diffusers import DDPMScheduler
from src.customized_modules.ddim import DDPMSchedulerCustomized
from omegaconf import OmegaConf
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from genpercept.genpercept_pipeline import GenPerceptPipeline, GenPerceptOutput
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dic_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.alignment import align_depth_least_square, depth2disparity, disparity2depth
from src.util.seeding import generate_seed_sequence

import accelerate
from packaging import version
from diffusers.utils.torch_utils import is_compiled_module
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file
from src.customized_modules.ddim import DDIMSchedulerCustomized
from genpercept.genpercept_pipeline import GenPerceptPipeline

from genpercept.models.dpt_head import DPTNeckHeadForUnetAfterUpsample, DPTNeckHeadForUnetAfterUpsampleIdentity
from genpercept.models.custom_unet import CustomUNet2DConditionModel
from genpercept.losses import L1Loss, ScaleAndShiftInvariantLoss, GradientLoss
from genpercept.losses.metric3d_losses import VNLoss, HDNRandomLoss, HDSNRandomLoss
from diffusers import AutoencoderKL

from src.util.alignment import compute_scale_and_shift
from peft import LoraConfig
from genpercept.losses.geometry_losses import angular_loss

from safetensors.torch import load_model, load_file
from transformers import DPTConfig

class StackedModels(nn.Module):
    def __init__(self, **kwargs):
        super(StackedModels, self).__init__()
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

class GenPerceptTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        unet,
        customized_head,
        vae,
        text_encoder,
        scheduler,
        tokenizer,
        train_dataloader: DataLoader,
        device,
        base_ckpt_dir,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
        accelerator=None,
        train_genpercept=False,
        device_seed=None,
        train_with_deepspeed=False,
    ):
        assert accelerator is not None
        self.accelerator = accelerator

        self.cfg: OmegaConf = cfg

        if 'depth' in self.cfg.gt_type:
            mode = 'depth'
        elif 'normal' in self.cfg.gt_type:
            mode = 'normal'
        elif 'matting' in self.cfg.gt_type:
            mode = 'matting'
        elif 'dis' in self.cfg.gt_type:
            mode = 'dis'
        elif 'seg' in self.cfg.gt_type:
            mode = 'seg'
        else:
            raise ValueError
        self.mode = mode
        self.vae = vae
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.tokenizer = tokenizer

        device = accelerator.device
        self.device = device
        self.seed = device_seed
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.base_ckpt_dir = base_ckpt_dir

        self.with_latent_loss = ('with_latent_loss' in self.cfg.loss.keys()) and (self.cfg.loss.with_latent_loss == True)

        # Loss
        if 'customized_loss' in cfg.loss.keys() and cfg.loss.customized_loss == True:
            if self.with_latent_loss:
                self.loss = torch.nn.MSELoss(**self.cfg.loss.kwargs)
            else:
                self.loss = None
            self.customized_loss = dict()
            loss_names = cfg.loss.name
            for loss_name in loss_names:
                if loss_name == 'l1_loss':
                    self.customized_loss[loss_name] = L1Loss(loss_weight=1.0)
                elif loss_name == "least_square_ssi_loss":
                    self.customized_loss[loss_name] = ScaleAndShiftInvariantLoss(align_type='least_square')
                elif loss_name == "medium_ssi_loss":
                    self.customized_loss[loss_name] = ScaleAndShiftInvariantLoss(align_type='medium')
                elif loss_name == "grad_loss":
                    self.customized_loss[loss_name] = GradientLoss(scales=1)
                elif loss_name == 'mse_loss':
                    self.customized_loss[loss_name] = get_loss(loss_name=loss_name, reduction='mean')
                elif loss_name == 'angular_loss':
                    assert self.mode == 'normal'
                    self.customized_loss[loss_name] = angular_loss
                elif loss_name == 'vnl_loss':
                    self.customized_loss[loss_name] = VNLoss(loss_weight=1.0, sample_ratio=0.2)
                elif loss_name == 'hdnr_loss':
                    self.customized_loss[loss_name] = HDNRandomLoss(loss_weight=0.5, random_num=10)
                elif loss_name == 'hdsnr_loss':
                    self.customized_loss[loss_name] = HDSNRandomLoss(loss_weight=0.5, random_num=20, batch_limit=4)
                else:
                    raise ValueError
                self.customized_loss[loss_name] = self.accelerator.prepare(self.customized_loss[loss_name])

        else:
            self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)
            self.customized_loss = None

        # Experiment setting
        self.rgb_blending = ('rgb_blending' in self.cfg.pipeline.kwargs.keys()) and (self.cfg.pipeline.kwargs.rgb_blending == True)
        self.train_genpercept = train_genpercept
        self.train_unet_lora = 'unet_lora_rank' in self.cfg.model.keys() and self.cfg.model.unet_lora_rank > 0
        self.train_unet = (not ('freeze_unet' in self.cfg.model.keys() and self.cfg.model.freeze_unet == True)) and (not self.train_unet_lora)
        self.train_vae_decoder = (not customized_head) and self.customized_loss and ('decoder_lr' in self.cfg.keys()) and (self.cfg.decoder_lr != 0)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        if not self.train_vae_decoder:
            self.vae = self.vae.to(dtype=self.weight_dtype)
        else:
            self.vae.encoder = self.vae.encoder.to(dtype=self.weight_dtype)
            self.vae.quant_conv = self.vae.quant_conv.to(dtype=self.weight_dtype)
            self.vae.decoder = self.vae.decoder.to(dtype=torch.float32)
            self.vae.post_quant_conv = self.vae.post_quant_conv.to(dtype=torch.float32)
            
        self.text_encoder = self.text_encoder.to(dtype=torch.float32) # Layernorm does not support fp16

        if not (self.train_unet or self.train_unet_lora):
            unet = unet.to(dtype=self.weight_dtype)

        # Adapt input layers
        if (8 != unet.config["in_channels"]) and (not self.train_genpercept) and (not self.rgb_blending):
            unet = self._replace_unet_conv_in(unet)
        
        if 'text_input' in self.cfg.model.keys() and self.cfg.model.text_input is not None:
            prompt = self.cfg.model.text_input
        else:
            prompt = ""
        
        self.prompt = prompt

        # Encode empty text prompt
        self.encode_text(prompt, unet)
        self.text_embed = self.text_embed.detach().clone().to(device)


        # Trainability
        self.text_encoder.requires_grad_(False)

        if self.train_unet:
            unet.requires_grad_(True)
        elif self.train_unet_lora:
            unet.requires_grad_(False)
            unet_lora_config = LoraConfig(
                r=self.cfg.model.unet_lora_rank,
                lora_alpha=self.cfg.model.unet_lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            # Add adapter and make sure the trainable params are in float32.
            unet.add_adapter(unet_lora_config)
            self.lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

        unet.enable_xformers_memory_efficient_attention()

        if customized_head is not None:
            customized_head.requires_grad_(True)
            self.vae.requires_grad_(False)
        elif self.train_vae_decoder:
            self.vae.encoder.requires_grad_(False)
            self.vae.quant_conv.requires_grad_(False)
            self.vae.decoder.requires_grad_(True)
            self.vae.post_quant_conv.requires_grad_(True)


        if train_with_deepspeed:
            model = StackedModels(unet=unet, customized_head=customized_head)
        
        print('-------------------- Training Settings --------------------')
        print('rgb_blending :', self.rgb_blending)
        print('train_unet_lora :', self.train_unet_lora)
        print('train_unet :', self.train_unet)
        print('train_vae_decoder :', self.train_vae_decoder)
        print('customized_head :', customized_head is not None)
        print('customized_loss :', self.customized_loss)
            
        # Optimizer !should be defined after input layer is adapted
        params = []
        if self.train_unet:
            print('Unet learning rate :', self.cfg.lr)
            params.append({"params": unet.parameters(), "lr": self.cfg.lr})
        elif self.train_unet_lora:
            print('Unet lora learning rate :', self.cfg.lr)
            params.append({"params": self.lora_layers, "lr": self.cfg.lr})
        if customized_head is not None:
            assert 'decoder_lr' in self.cfg.keys()
            print('Customized head learning rate :', self.cfg.decoder_lr)
            params.append({"params": customized_head.parameters(), "lr": self.cfg.decoder_lr})
        if self.train_vae_decoder:
            assert 'decoder_lr' in self.cfg.keys()
            print('VAE decoder learning rate :', self.cfg.decoder_lr)
            params.append({"params": vae.post_quant_conv.parameters(), "lr": self.cfg.decoder_lr})
            params.append({"params": vae.decoder.parameters(), "lr": self.cfg.decoder_lr})
        print('---------------- End of Training Settings ----------------')

        # params = [{"params": model.parameters(), "lr": self.cfg.lr}]
        self.optimizer = Adam(params)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        if train_with_deepspeed:
            # Accelerator prepare
            self.train_loader, model, self.optimizer, self.lr_scheduler = accelerator.prepare(
                self.train_loader, model, self.optimizer, self.lr_scheduler
            )
            self.unet = model.unet
            self.customized_head = model.customized_head
            self.vae = self.vae.to(accelerator.device)
        else:
            self.unet = unet
            self.customized_head = customized_head

            self.train_loader, self.optimizer, self.lr_scheduler = accelerator.prepare(
                self.train_loader, self.optimizer, self.lr_scheduler
            )

            if self.train_unet or self.train_unet_lora:
                self.unet = accelerator.prepare(self.unet)
            else:
                self.unet.to(accelerator.device)
            
            if self.customized_head:
                self.customized_head = accelerator.prepare(self.customized_head)
            
            if self.train_vae_decoder:
                self.vae.decoder, self.vae.post_quant_conv = accelerator.prepare(self.vae.decoder, self.vae.post_quant_conv)
                self.vae.encoder = self.vae.encoder.to(accelerator.device)
                self.vae.quant_conv = self.vae.quant_conv.to(accelerator.device)
            else:
                self.vae = self.vae.to(accelerator.device)
        
        if not self.train_genpercept:
            if 'scheduler_path' in self.cfg.model.keys() and self.cfg.model.scheduler_path is not None:
                # Training noise scheduler
                self.training_noise_scheduler: DDPMSchedulerCustomized = DDPMSchedulerCustomized.from_pretrained(self.cfg.model.scheduler_path)
            else:
                # Training noise scheduler
                self.training_noise_scheduler: DDPMSchedulerCustomized = DDPMSchedulerCustomized.from_pretrained(
                    os.path.join(
                        base_ckpt_dir,
                        cfg.trainer.training_noise_scheduler.pretrained_path,
                        "scheduler",
                    )
                )
        else:
            self.training_noise_scheduler: DDPMSchedulerCustomized = DDPMSchedulerCustomized.from_pretrained(
                os.path.join(
                    base_ckpt_dir,
                    cfg.trainer.training_noise_scheduler.pretrained_path,
                    "scheduler_beta_1.0_1.0",
                )
            )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.scheduler.config.prediction_type
        ), "Different prediction types"
        if self.train_genpercept:
            assert self.prediction_type == 'v_prediction'
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        # self.gradient_accumulation_steps = accumulation_steps
        self.gt_type = self.cfg.gt_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Multi-resolution noise
        self.apply_multi_res_noise = ('multi_res_noise' in self.cfg.keys()) and (self.cfg.multi_res_noise is not None) and (not self.train_genpercept) and (not self.rgb_blending)
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming
        self.progress_bar = tqdm(
            range(0, self.max_iter),
            initial=0,
            desc="Steps",
            disable=not accelerator.is_main_process,
        )

        self.latent_scale_factor = 0.18215
        
        if train_with_deepspeed:
            self.unet_dtype = self.unet.dtype
        else:
            self.unet_dtype = self.accelerator.unwrap_model(self.unet).dtype
            
        self.text_embed = self.text_embed.to(dtype=self.unet_dtype)

        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def unwrap_model(model):
                model = accelerator.unwrap_model(model)
                model = model._orig_mod if is_compiled_module(model) else model
                return model
            
            def save_model_hook(models, weights, output_dir):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    for i, model in enumerate(models):
                        if isinstance(unwrap_model(model), UNet2DConditionModel) or isinstance(unwrap_model(model), CustomUNet2DConditionModel):
                            sub_dir = 'unet'
                            file_name = 'diffusion_pytorch_model.safetensors'
                        elif isinstance(unwrap_model(model), DPTNeckHeadForUnetAfterUpsampleIdentity):
                            sub_dir = 'dpt_head_identity'
                            file_name = 'model.safetensors'
                        elif isinstance(unwrap_model(model), DPTNeckHeadForUnetAfterUpsample):
                            sub_dir = 'dpt_head'
                            file_name = 'model.safetensors'
                        elif self.train_vae_decoder and isinstance(unwrap_model(model), type(unwrap_model(self.vae.decoder))):
                            sub_dir = 'vae_decoder'
                            file_name = 'model.safetensors'
                        elif self.train_vae_decoder and isinstance(unwrap_model(model), type(unwrap_model(self.vae.post_quant_conv))):
                            sub_dir = 'vae_post_quant_conv'
                            file_name = 'model.safetensors'
                        else:
                            continue
                        
                        import pdb; pdb.set_trace()
                        state_dict = accelerator.get_state_dict(model)
                        os.makedirs(osp.join(output_dir, sub_dir), exist_ok=True)
                        save_file(state_dict, osp.join(output_dir, sub_dir, file_name))
                        try:
                            model.save_config(osp.join(output_dir, sub_dir))
                        except:
                            print('Warning! Fail to save the config of %s' %(type(model)))
                        del state_dict

                        # # make sure to pop weight so that corresponding model is not saved again
                        if weights:
                            weights.pop()

            def load_model_hook(models, input_dir):
                
                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()
                    if type(model) == type(accelerator.unwrap_model(model)):
                        if isinstance(model, UNet2DConditionModel) or isinstance(model, CustomUNet2DConditionModel):
                            sub_dir = "unet" 

                            if self.customized_head:
                                unet = CustomUNet2DConditionModel.from_pretrained(os.path.join(self.base_ckpt_dir, self.cfg.model.pretrained_path), subfolder=sub_dir)
                                del unet.conv_out
                                del unet.conv_norm_out
                            else:
                                unet = UNet2DConditionModel.from_pretrained(os.path.join(self.base_ckpt_dir, self.cfg.model.pretrained_path), subfolder=sub_dir)
                            
                            # Adapt input layers
                            if (8 != unet.config["in_channels"]) and (not self.train_genpercept) and (not self.rgb_blending):
                                unet = self._replace_unet_conv_in(unet)
                            
                            if osp.exists(osp.join(input_dir, sub_dir, 'diffusion_pytorch_model.bin')):
                                unet_ckpt_path = osp.join(input_dir, sub_dir, 'diffusion_pytorch_model.bin')
                            elif osp.exists(osp.join(input_dir, sub_dir, 'diffusion_pytorch_model.safetensors')):
                                unet_ckpt_path = osp.join(input_dir, sub_dir, 'diffusion_pytorch_model.safetensors')
                            else:
                                print('Warning!!! the saved checkpoint does not contain U-Net. Load U-Net from pretrained models...')
                                unet_ckpt_path = osp.join(os.path.join(self.base_ckpt_dir, self.cfg.model.pretrained_path), 'unet', 'diffusion_pytorch_model.safetensors')

                            ckpt = load_file(unet_ckpt_path)
                            if self.customized_head:
                                ckpt_new = {}
                                for key in ckpt:
                                    if 'conv_out' in key:
                                        continue
                                    if 'conv_norm_out' in key:
                                        continue
                                    ckpt_new[key] = ckpt[key]
                            else:
                                ckpt_new = ckpt
                            
                            if self.train_unet_lora:
                                unet_lora_config = LoraConfig(
                                    r=self.cfg.model.unet_lora_rank,
                                    lora_alpha=self.cfg.model.unet_lora_rank,
                                    init_lora_weights="gaussian",
                                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                                )
                                # Add adapter and make sure the trainable params are in float32.
                                unet.add_adapter(unet_lora_config)
                                unet.requires_grad_(False)

                            unet.load_state_dict(ckpt_new)

                            loaded_model = unet

                        elif isinstance(model, DPTNeckHeadForUnetAfterUpsampleIdentity):
                            sub_dir = "dpt_head_identity" 
                            dpt_config = DPTConfig.from_pretrained("hf_configs/dpt-sd2.1-unet-after-upsample-general")
                            loaded_model = DPTNeckHeadForUnetAfterUpsampleIdentity(config=dpt_config)
                            load_model(loaded_model, osp.join(input_dir, sub_dir, 'model.safetensors'))
                        elif isinstance(model, DPTNeckHeadForUnetAfterUpsample):
                            sub_dir = "dpt_head" 
                            dpt_config = DPTConfig.from_pretrained("hf_configs/dpt-sd2.1-unet-after-upsample-general")
                            loaded_model = DPTNeckHeadForUnetAfterUpsample(config=dpt_config)
                            load_model(loaded_model, osp.join(input_dir, sub_dir, 'model.safetensors'))
                        else:
                            print('Passing %s when loading from checkpoint...' %type(model))
                            continue
                            # raise NotImplementedError

                        try:
                            model.register_to_config(**loaded_model.config)
                        except:
                            pass
                        model.load_state_dict(loaded_model.state_dict())

                        del loaded_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)
        
        if 'freeze_unet' in self.cfg.model.keys() and self.cfg.model.freeze_unet == True:
            pass
        else:
            self.unet.train()

        if self.customized_head is not None:
            self.customized_head.train()
        elif self.train_vae_decoder:
            self.vae.post_quant_conv.train()
            self.vae.decoder.train()

    def _replace_unet_conv_in(self, unet):
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

    def train(self, t_end=None):
        logging.info("Start training")

        device = self.accelerator.device

        print('-------------------- Traning Dtype --------------------')
        print('unet_dtype :', self.accelerator.unwrap_model(self.unet).dtype)
        print('vae_dtype :', self.accelerator.unwrap_model(self.vae).dtype)
        print('---------------- End of Training Dtype ----------------')

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        self.train_metrics.reset()

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            accumulate_modules = []
            if self.train_unet:
                accumulate_modules.append(self.unet)
            if self.train_unet_lora:
                accumulate_modules.append(self.lora_layers)
            if self.customized_head:
                accumulate_modules.append(self.customized_head)
            elif self.train_vae_decoder:
                accumulate_modules.append(self.vae.decoder)
                accumulate_modules.append(self.vae.post_quant_conv)
            # for batch in self.train_loader:
            for batch in self.accelerator.prepare(skip_first_batches(self.train_loader, self.n_batch_in_epoch)):
                with self.accelerator.accumulate(accumulate_modules):

                    # globally consistent random generators
                    if self.seed is not None:
                        local_seed = self._get_next_seed()
                        rand_num_generator = torch.Generator(device=device)
                        rand_num_generator.manual_seed(local_seed)
                    else:
                        rand_num_generator = None

                    # Get data
                    rgb = batch["rgb_norm"].to(dtype=self.weight_dtype)
                    gt_for_latent = batch[self.gt_type].to(dtype=self.weight_dtype)

                    if self.gt_mask_type is not None:
                        valid_mask_for_latent = batch[self.gt_mask_type]
                        invalid_mask = ~valid_mask_for_latent
                        valid_mask_down = ~torch.max_pool2d(
                            invalid_mask.float(), 8, 8
                        ).bool()
                        valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))
                    else:
                        raise NotImplementedError

                    batch_size = rgb.shape[0]

                    with torch.no_grad():
                        # Encode image
                        rgb_latent = self.encode_rgb(rgb)  # [B, 4, h, w]
                        # Encode GT label
                        gt_latent = self.encode_perception(
                            gt_for_latent
                        )  # [B, 4, h, w]

                    if 'fix_timesteps' in self.cfg.model.keys():
                        timesteps = torch.tensor([self.cfg.model.fix_timesteps]).long().repeat(rgb.shape[0]).to(self.unet.device)
                    else:
                        # Sample a random timestep for each image
                        timesteps = torch.randint(
                            0,
                            self.scheduler_timesteps,
                            (batch_size,),
                            device=device,
                            generator=rand_num_generator,
                        ).long()  # [B]

                    # Sample noise
                    if not self.train_genpercept and (not self.rgb_blending):
                        if self.apply_multi_res_noise:
                            strength = self.mr_noise_strength
                            if self.annealed_mr_noise:
                                # calculate strength depending on t
                                strength = strength * (timesteps / self.scheduler_timesteps)
                            noise = multi_res_noise_like(
                                gt_latent,
                                strength=strength,
                                downscale_strategy=self.mr_noise_downscale_strategy,
                                generator=rand_num_generator,
                                device=device,
                            )
                        else:
                            noise = torch.randn(
                                gt_latent.shape,
                                device=device,
                                generator=rand_num_generator,
                            )  # [B, 4, h, w]
                    else: # NOTE: GenPercept take the rgb latent as noise.
                        noise = rgb_latent.clone()

                    # Add noise to the latents (diffusion forward process)
                    noisy_latents = self.training_noise_scheduler.add_noise(
                        gt_latent, noise, timesteps
                    )  # [B, 4, h, w]

                    # Text embedding
                    text_embed = self.text_embed.to(device).repeat(
                        (batch_size, 1, 1)
                    )  # [B, 77, 1024]

                    if not self.train_genpercept and (not self.rgb_blending):
                        # Concat rgb and noisy gt latents
                        cat_latents = torch.cat(
                            [rgb_latent, noisy_latents], dim=1
                        )  # [B, 8, h, w]
                    else:
                        cat_latents = noisy_latents
                    cat_latents = cat_latents.to(self.unet_dtype)

                    # Get the target for loss depending on the prediction type
                    if "sample" == self.prediction_type:
                        target = gt_latent
                    elif "epsilon" == self.prediction_type:
                        target = noise
                    elif "v_prediction" == self.prediction_type:
                        target = self.training_noise_scheduler.get_velocity(
                            gt_latent, noise, timesteps
                        )  # [B, 4, h, w]
                    else:
                        raise ValueError(f"Unknown prediction type {self.prediction_type}")

                    if self.customized_loss is None:
                        # Predict the noise residual
                        model_pred = self.unet(
                            cat_latents, timesteps, text_embed
                        ).sample  # [B, 4, h, w]
                        if torch.isnan(model_pred).any():
                            logging.warning("model_pred contains NaN.")
                        # Masked latent loss
                        if self.gt_mask_type is not None:
                            if valid_mask_down.sum() > 0:
                                latent_loss = self.loss(
                                    model_pred[valid_mask_down].float(),
                                    target[valid_mask_down].float(),
                                )
                            else:
                                latent_loss = model_pred.mean() * 0.
                        else:
                            latent_loss = self.loss(model_pred.float(), target.float())

                        loss = latent_loss.mean()
                        self.train_metrics.update("loss", loss.item())
                    else:
                        assert self.train_genpercept
                        loss_dict = {}
                        loss = 0

                        if self.customized_head is None: # Use VAE decoder
                            # Predict the noise residual
                            noise_pred = self.unet(
                                cat_latents, timesteps, text_embed
                            ).sample  # [B, 4, h, w]

                            if self.with_latent_loss:
                                # Masked latent loss
                                if self.gt_mask_type is not None:
                                    if valid_mask_down.sum() > 0:
                                        latent_loss = self.loss(
                                            noise_pred[valid_mask_down].float(),
                                            target[valid_mask_down].float(),
                                        )
                                    else:
                                        latent_loss = noise_pred.mean() * 0.
                                else:
                                    latent_loss = self.loss(noise_pred.float(), target.float())

                                loss += latent_loss.mean()
                            
                            model_pred = - noise_pred 
                            head_pred = self.decode_perception(model_pred.to(self.vae.dtype))
                        # TODO: customized head of other tasks besides depth.
                        elif isinstance(self.accelerator.unwrap_model(self.customized_head), DPTNeckHeadForUnetAfterUpsample) or isinstance(self.accelerator.unwrap_model(self.customized_head), DPTNeckHeadForUnetAfterUpsampleIdentity):
                            if self.with_latent_loss:
                                raise NotImplementedError
                            # Predict the noise residual
                            unet_features = self.unet(
                                cat_latents, timesteps, text_embed, return_feature=True
                            ).multi_level_feats[::-1]  # [B, 4, h, w]

                            head_pred = self.customized_head(hidden_states=unet_features).prediction[:, None]
                        else:
                            raise ValueError

                        if torch.isnan(head_pred).any():
                            logging.warning("head_pred contains NaN.")
                        
                        for loss_name, loss_func in self.customized_loss.items():
                            
                            if self.mode == 'depth':
                                prediction = head_pred.float() + 2 # NOTE: shift [-1, 1] to [1, 3] to avoid negative values
                                target = gt_for_latent[:, :1].float() + 2 # NOTE: shift [-1, 1] to [1, 3] to avoid negative values
                                mask = valid_mask_for_latent.bool()
                            else:
                                prediction = head_pred.repeat(1, 3, 1, 1).float() if head_pred.shape[1] == 1 else head_pred.float()
                                target = gt_for_latent.float()
                                mask = valid_mask_for_latent.repeat(1, 3, 1, 1).bool() if valid_mask_for_latent.shape[1] == 1 else valid_mask_for_latent.float()
                            data_dict = dict(
                                prediction=prediction,
                                target=target,
                                mask=mask,
                            )

                            if loss_name not in ['least_square_ssi_loss', 'medium_ssi_loss', 'grad_loss', 'mse_loss']:
                                if 'intrinsic' in batch.keys():
                                    data_dict['intrinsic'] = batch['intrinsic']
                                else:
                                    data_dict['intrinsic'] = None

                                loss_i = loss_func(**data_dict)
                                assert not (torch.isnan(loss_i) or torch.isinf(loss_i))
                                loss += loss_i
                                loss_dict[loss_name] = loss_i
                            elif loss_name == 'mse_loss':
                                loss_i = loss_func(prediction[mask], target[mask])
                                loss_weight = 1
                                loss += loss_i * loss_weight
                                loss_dict[loss_name] = loss_i
                            elif loss_name in ['least_square_ssi_loss', 'medium_ssi_loss']:
                                assert self.mode == 'depth'
                                loss_i = loss_func(**data_dict)
                                loss_weight = 0.5
                                loss += loss_i * loss_weight
                                loss_dict[loss_name] = loss_i
                            elif loss_name == 'grad_loss':
                                assert self.mode == 'depth'
                                loss_i = loss_func(**data_dict)
                                loss_weight = 2
                                loss += loss_i * loss_weight
                                loss_dict[loss_name] = loss_i
                            else:
                                raise NotImplementedError
                            self.train_metrics.update("loss", loss.item())
                    
                    # # NOTE: debug only
                    # viz_sample_id = 1
                    # decoded_neg_target_0 = self.decode_perception(-target.to(self.vae.dtype))[viz_sample_id].clip(-1, 1)
                    # decoded_neg_target_0 = (decoded_neg_target_0 + 1) / 2 * 255
                    # plt.imsave('temp_decoded_neg_target_0.png', decoded_neg_target_0.permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
                    # gt_0 = gt_for_latent[viz_sample_id].clip(-1, 1)
                    # gt_0 = (gt_0 + 1) / 2 * 255
                    # plt.imsave('temp_gt_0.png', gt_0.permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
                    # decoded_gt_0 = self.decode_perception(gt_latent.to(self.vae.dtype))[viz_sample_id].clip(-1, 1)
                    # decoded_gt_0 = (decoded_gt_0 + 1) / 2 * 255
                    # plt.imsave('temp_decoded_gt_0.png', decoded_gt_0.permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
                    # decoded_noisy_latent_0 = self.decode_rgb(noisy_latents.to(self.vae.dtype))[viz_sample_id].clip(-1, 1)
                    # decoded_noisy_latent_0 = (decoded_noisy_latent_0 + 1) / 2 * 255
                    # plt.imsave('temp_decoded_noisy_latent_0.png', decoded_noisy_latent_0.permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
                    # decoded_noise = self.decode_rgb(noise.to(self.vae.dtype))[viz_sample_id].clip(-1, 1)
                    # decoded_noise = (decoded_noise + 1) / 2 * 255
                    # plt.imsave('temp_decoded_noise_0.png', decoded_noise.permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
                    # import pdb; pdb.set_trace()

                    self.accelerator.backward(loss)

                    # print('self.accelerator.unwrap_model(self.unet).conv_in.bias.grad.sum() :', self.accelerator.unwrap_model(self.unet).conv_in.bias.grad.sum())

                    self.n_batch_in_epoch += 1

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Perform optimization step
                # if accumulated_step >= self.gradient_accumulation_steps:
                if self.accelerator.sync_gradients:
                    # accumulated_step = 0

                    self.effective_iter += 1
                    self.progress_bar.update(1)

                    if self.accelerator.is_main_process:
                        # Log to tensorboard
                        accumulated_loss = self.train_metrics.result()["loss"]
                        tb_logger.log_dic(
                            {
                                f"train/{k}": v
                                for k, v in self.train_metrics.result().items()
                            },
                            global_step=self.effective_iter,
                        )
                        tb_logger.writer.add_scalar(
                            "lr",
                            self.lr_scheduler.get_last_lr()[0],
                            global_step=self.effective_iter,
                        )
                        tb_logger.writer.add_scalar(
                            "n_batch_in_epoch",
                            self.n_batch_in_epoch,
                            global_step=self.effective_iter,
                        )
                        process_bar_dict = dict(loss=accumulated_loss)
                        self.progress_bar.set_postfix(**process_bar_dict)
                        self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.accelerator.wait_for_everyone()
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.accelerator.wait_for_everyone()
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    self.accelerator.wait_for_everyone()
                
                torch.cuda.empty_cache()
                # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0

        self.accelerator.wait_for_everyone()

        self.accelerator.end_training()

    def encode_text(self, prompt, unet):
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
        self.text_embed = self.text_encoder(text_input_ids)[0].to(unet.dtype)
    
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

    def decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        rgb_latent = rgb_latent / self.latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(rgb_latent)
        stacked = self.vae.decoder(z)
        return stacked
    
    def encode_perception(self, gt_in):
        if 'depth' in self.gt_type:
            # stack depth into 3-channel
            stacked = self.stack_depth_images(gt_in)
        else:
            stacked = gt_in
        # encode using VAE encoder
        gt_latent = self.encode_rgb(stacked)
        return gt_latent

    def decode_perception(self, perception_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode perception latent into perception map.

        Args:
            perception_latent (`torch.Tensor`):
                Perception latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded perception map.
        """
        # scale latent
        perception_latent = perception_latent / self.latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(perception_latent)
        stacked = self.vae.decoder(z)
        if self.mode in ['depth', 'matting', 'dis']:
            # mean of output channels
            stacked = stacked.mean(dim=1, keepdim=True)
        else:
            stacked = stacked
        return stacked

    @staticmethod
    def stack_depth_images(depth_in):
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1)
            stacked = depth_in.repeat(1, 3, 1, 1)
        return stacked

    def _train_step_callback(self):
        """Executed after every iteration"""
        # # Save backup (with a larger interval, without training states)
        # if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period and (self.accelerator.is_main_process):
        #     self.save_checkpoint(
        #         ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
        #     )

        # _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period and (self.accelerator.is_main_process):
            # self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            # self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            # _is_latest_saved = True
            self.validate()
            # self.in_evaluation = False
            # self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
        ):
            self.accelerator.wait_for_everyone()
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period and (self.accelerator.is_main_process):
            self.visualize()

    def validate(self):
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            val_metric_dic = self.validate_single_dataset(
                data_loader=val_loader, metric_tracker=self.val_metrics
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dic}"
            )
            tb_logger.log_dic(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dic.items()},
                global_step=self.effective_iter,
            )
            # save to file
            eval_text = eval_dic_to_text(
                val_metrics=val_metric_dic,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)

            # Update main eval metric
            if 0 == i:
                main_eval_metric = val_metric_dic[self.main_val_metric]
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )
                    # # Save a checkpoint
                    # self.save_checkpoint(
                    #     ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    # )

    def visualize(self):
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=None,
                save_to_dir=vis_out_dir,
            )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        if metric_tracker:
            metric_tracker.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

        # pipeline
        _pipeline_kwargs = self.cfg.pipeline.kwargs if self.cfg.pipeline.kwargs is not None else {}
        if self.cfg.pipeline.name == 'MarigoldPipeline':
            if 'scheduler_path' in self.cfg.model.keys() and self.cfg.model.scheduler_path is not None:
                customize_ddim_scheduler = DDIMSchedulerCustomized.from_pretrained(self.cfg.model.scheduler_path)
            else:
                customize_ddim_scheduler = DDIMSchedulerCustomized.from_pretrained(os.path.join(self.base_ckpt_dir, self.cfg.model.pretrained_path), subfolder='scheduler')

            genpercept_pipeline = False
            model = GenPerceptPipeline.from_pretrained(
                os.path.join(self.base_ckpt_dir, self.cfg.model.pretrained_path),
                unet=self.accelerator.unwrap_model(self.unet),
                scheduler=customize_ddim_scheduler,
                genpercept_pipeline=genpercept_pipeline,
                **_pipeline_kwargs,
            )
        elif self.cfg.pipeline.name == 'GenPerceptPipeline':
            customize_ddim_scheduler = DDIMSchedulerCustomized.from_pretrained('hf_configs/scheduler_beta_1.0_1.0')
            vae = AutoencoderKL.from_pretrained(os.path.join(self.base_ckpt_dir, self.cfg.model.pretrained_path), subfolder='vae')

            genpercept_pipeline = True
            model = GenPerceptPipeline.from_pretrained(
                os.path.join(self.base_ckpt_dir, self.cfg.model.pretrained_path),
                unet=self.accelerator.unwrap_model(self.unet),
                vae=vae.float(),
                scheduler=customize_ddim_scheduler,
                customized_head=self.accelerator.unwrap_model(self.customized_head) if self.customized_head is not None else None,
                genpercept_pipeline=genpercept_pipeline,
                **_pipeline_kwargs,
            )
        else:
            raise ValueError

        torch.cuda.empty_cache()
        model = model.to(self.device).to(dtype=torch.float32)

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            assert 1 == data_loader.batch_size
            # Read input image
            rgb_int = batch["rgb_int"].to(dtype=self.weight_dtype)  # [B, 3, H, W]
            if metric_tracker:
                # GT perception
                pred_raw_ts = batch[self.cfg.gt_type.replace('_norm', '_linear')].squeeze()
                pred_raw = pred_raw_ts.numpy()
                valid_mask_ts = batch[self.cfg.gt_mask_type].squeeze()
                valid_mask = valid_mask_ts.numpy()

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
            
            # Result
            pipe_out: GenPerceptOutput = model(
                rgb_int,
                denoising_steps=self.cfg.validation.denoising_steps,
                ensemble_size=self.cfg.validation.ensemble_size,
                processing_res=self.cfg.validation.processing_res,
                match_input_res=self.cfg.validation.match_input_res,
                generator=generator,
                batch_size=1,  # use batch size 1 to increase reproducibility
                color_map=None,
                show_progress_bar=False,
                resample_method=self.cfg.validation.resample_method,
                mode=self.mode,
                fix_timesteps=self.cfg.model.fix_timesteps if 'fix_timesteps' in self.cfg.model.keys() else None,
                prompt=self.prompt,
            )

            pred: np.ndarray = pipe_out.pred_np

            # TODO: other metrics for evaluation
            if metric_tracker:
                if self.mode == 'depth':
                    if "least_square" == self.cfg.eval.alignment:
                        pred, scale, shift = align_depth_least_square(
                            gt_arr=pred_raw,
                            pred_arr=pred,
                            valid_mask_arr=valid_mask,
                            return_scale_shift=True,
                            max_resolution=self.cfg.eval.align_max_res,
                        )
                    elif "least_square_disparity" == self.cfg.eval.alignment:
                        # convert GT depth -> GT disparity
                        gt_disparity, gt_non_neg_mask = depth2disparity(
                            depth=pred_raw, return_mask=True
                        )
                        # LS alignment in disparity space
                        pred_non_neg_mask = pred > 0
                        valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

                        disparity_pred, scale, shift = align_depth_least_square(
                            gt_arr=gt_disparity,
                            pred_arr=pred,
                            valid_mask_arr=valid_nonnegative_mask,
                            return_scale_shift=True,
                            max_resolution=self.cfg.eval.align_max_res,
                        )
                        # convert to depth
                        disparity_pred = np.clip(
                            disparity_pred, a_min=1e-3, a_max=None
                        )  # avoid 0 disparity
                        pred = disparity2depth(disparity_pred)
                    else:
                        raise RuntimeError(f"Unknown alignment type: {self.cfg.eval.alignment}")

                    # Clip to dataset min max
                    pred = np.clip(
                        pred,
                        a_min=data_loader.dataset.min_depth,
                        a_max=data_loader.dataset.max_depth,
                    )

                    # clip to d > 0 for evaluation
                    pred = np.clip(pred, a_min=1e-6, a_max=None)

                    # Evaluate
                    sample_metric = []
                    pred_ts = torch.from_numpy(pred)

                    for met_func in self.metric_funcs:
                        _metric_name = met_func.__name__
                        _metric = met_func(pred_ts, pred_raw_ts, valid_mask_ts).item()
                        sample_metric.append(_metric.__str__())
                        metric_tracker.update(_metric_name, _metric)

            # Save as 16-bit uint png
            if save_to_dir is not None:
                img_name = batch["rgb_relative_path"][0].replace("/", "_")
                png_save_path = os.path.join(save_to_dir, f"{img_name}.png")
                pred_to_save = (pipe_out.pred_np * 65535.0).astype(np.uint16)
                if self.mode in ['depth', 'matting', 'dis']:
                    Image.fromarray(pred_to_save).save(png_save_path, mode="I;16")
                elif self.mode in ['normal', 'seg']:
                    normal = pipe_out.pred_np * 255
                    normal = normal.astype(np.uint8)
                    Image.fromarray(normal).save(png_save_path)
                else:
                    raise ValueError

        self.unet = self.unet.to(self.unet_dtype)

        if metric_tracker:
            return metric_tracker.result()
        else:
            return None

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.accelerator.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        if self.accelerator.is_main_process:
            # Backup previous checkpoint
            temp_ckpt_dir = None
            if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
                temp_ckpt_dir = os.path.join(
                    os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
                )
                if os.path.exists(temp_ckpt_dir):
                    shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
                os.rename(ckpt_dir, temp_ckpt_dir)
                logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # # Save UNet
        # unet_path = os.path.join(ckpt_dir, "unet")
        # self.unet.save_pretrained(unet_path, safe_serialization=False)
        # logging.info(f"UNet is saved to: {unet_path}")

        # Save UNet
        self.accelerator.save_state(ckpt_dir)
        logging.info(f"Saved denoiser state to {ckpt_dir}") 

        # if save_train_state and self.accelerator.is_main_process:
        #     state = {
        #         "optimizer": self.optimizer.state_dict(),
        #         "lr_scheduler": self.lr_scheduler.state_dict(),
        #         "config": self.cfg,
        #         "effective_iter": self.effective_iter,
        #         "epoch": self.epoch,
        #         "n_batch_in_epoch": self.n_batch_in_epoch,
        #         "best_metric": self.best_metric,
        #         "in_evaluation": self.in_evaluation,
        #         "global_seed_sequence": self.global_seed_sequence,
        #     }
        #     train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
        #     torch.save(state, train_state_path)
        #     # iteration indicator
        #     f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
        #     f.close()

        #     logging.info(f"Trainer state is saved to: {train_state_path}")

        if self.accelerator.is_main_process:
            # Remove temp ckpt
            if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
                logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        # TODO: (guangkaixu) match with multi-gpu.
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        # _model_path = os.path.join(ckpt_path, "unet")
        # self.unet.load_state_dict(
        #     torch.load(_model_path, map_location=self.device)
        # )

        # import pdb; pdb.set_trace()
        self.accelerator.load_state(ckpt_path)
        logging.info(f"Parameters are loaded from {ckpt_path}")

        # # Load training states
        # if load_trainer_state:
        #     checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
        #     self.effective_iter = checkpoint["effective_iter"]
        #     self.epoch = checkpoint["epoch"]
        #     self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
        #     self.in_evaluation = checkpoint["in_evaluation"]
        #     self.global_seed_sequence = checkpoint["global_seed_sequence"]

        #     self.best_metric = checkpoint["best_metric"]

        #     self.optimizer.load_state_dict(checkpoint["optimizer"])
        #     logging.info(f"optimizer state is loaded from {ckpt_path}")

        #     if resume_lr_scheduler:
        #         self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        #         logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        # logging.info(
        #     f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        # )

        self.effective_iter = self.lr_scheduler.state_dict()['last_epoch']
        self.epoch = self.cfg.dataloader.effective_batch_size * self.effective_iter // (len(self.train_loader) * self.cfg.dataloader.max_train_batch_size) + 1
        self.n_batch_in_epoch = self.cfg.dataloader.effective_batch_size * self.effective_iter // self.cfg.dataloader.max_train_batch_size - len(self.train_loader) * (self.epoch - 1)
        # self.in_evaluation = ...
        # self.global_seed_sequence = ...
        # self.best_metric = ...

        self.progress_bar.n = self.effective_iter

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"