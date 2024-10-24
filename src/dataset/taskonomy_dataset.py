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


from .base_dataset import BaseDataset, PerceptionFileNameMode
import numpy as np


class TaskonomyDataset(BaseDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # Hypersim data parameter
            min_depth=1e-5,
            max_depth=65.0,
            has_filled_depth=False,
            name_mode=PerceptionFileNameMode.rgb_i_d,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        if len(depth_in.shape) == 3 and depth_in.shape[2] == 3:
            assert np.all(depth_in[:, :, 0] == depth_in[:, :, 1])
            assert np.all(depth_in[:, :, 0] == depth_in[:, :, 2])
            depth_in = depth_in[:, :, 0]
        # Decode Hypersim depth
        if not self.is_exr_data:
            depth_decoded = depth_in / 512.0
        else:
            depth_decoded = depth_in
        return depth_decoded
