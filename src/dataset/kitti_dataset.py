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

import torch

from .base_dataset import BaseDataset, PerceptionFileNameMode


class KITTIDataset(BaseDataset):
    def __init__(
        self,
        kitti_bm_crop,  # Crop to KITTI benchmark size
        valid_mask_crop,  # Evaluation mask. [None, garg or eigen]
        **kwargs,
    ) -> None:
        super().__init__(
            # KITTI data parameter
            min_depth=1e-5,
            max_depth=80,
            has_filled_depth=False,
            name_mode=PerceptionFileNameMode.id,
            **kwargs,
        )
        self.kitti_bm_crop = kitti_bm_crop
        self.valid_mask_crop = valid_mask_crop
        assert self.valid_mask_crop in [
            None,
            "garg",  # set evaluation mask according to Garg  ECCV16
            "eigen",  # set evaluation mask according to Eigen NIPS14
        ], f"Unknown crop type: {self.valid_mask_crop}"

        # Filter out empty depth
        self.filenames = [f for f in self.filenames if "None" != f[1]]

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode KITTI depth
        if not self.is_exr_data:
            depth_decoded = depth_in / 256.0
        else:
            depth_decoded = depth_in
        return depth_decoded

    def _load_rgb_data(self, rgb_rel_path):
        rgb_data = super()._load_rgb_data(rgb_rel_path)
        if self.kitti_bm_crop:
            rgb_data = {k: self.kitti_benchmark_crop(v) for k, v in rgb_data.items()}
        return rgb_data

    def _load_depth_data(self, depth_rel_path, filled_rel_path, shape):
        depth_data = super()._load_depth_data(depth_rel_path, filled_rel_path, shape)
        if self.kitti_bm_crop:
            depth_data = {
                k: self.kitti_benchmark_crop(v) for k, v in depth_data.items()
            }
        return depth_data

    @staticmethod
    def kitti_benchmark_crop(input_img):
        """
        Crop images to KITTI benchmark size
        Args:
            `input_img` (torch.Tensor): Input image to be cropped.

        Returns:
            torch.Tensor:Cropped image.
        """
        KB_CROP_HEIGHT = 352
        KB_CROP_WIDTH = 1216

        height, width = input_img.shape[-2:]
        top_margin = int(height - KB_CROP_HEIGHT)
        left_margin = int((width - KB_CROP_WIDTH) / 2)
        if 2 == len(input_img.shape):
            out = input_img[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        elif 3 == len(input_img.shape):
            out = input_img[
                :,
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        return out

    def _get_valid_mask(self, depth: torch.Tensor):
        # reference: https://github.com/cleinc/bts/blob/master/pytorch/bts_eval.py
        valid_mask = super()._get_valid_mask(depth)  # [1, H, W]

        if self.valid_mask_crop is not None:
            eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
            gt_height, gt_width = eval_mask.shape

            if "garg" == self.valid_mask_crop:
                eval_mask[
                    int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
                ] = 1
            elif "eigen" == self.valid_mask_crop:
                eval_mask[
                    int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                ] = 1

            eval_mask.reshape(valid_mask.shape)
            valid_mask = torch.logical_and(valid_mask, eval_mask)
        return valid_mask
