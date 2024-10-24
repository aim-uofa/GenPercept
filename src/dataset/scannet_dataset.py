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


class ScanNetDataset(BaseDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # ScanNet data parameter
            min_depth=1e-3,
            max_depth=10,
            has_filled_depth=False,
            name_mode=PerceptionFileNameMode.id,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode ScanNet depth
        if not self.is_exr_data:
            depth_decoded = depth_in / 1000.0
        else:
            depth_decoded = depth_in
        return depth_decoded
