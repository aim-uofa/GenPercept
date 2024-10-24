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
import tarfile
import os
import numpy as np

from .base_dataset import BaseDataset, PerceptionFileNameMode


class ETH3DDataset(BaseDataset):
    HEIGHT, WIDTH = 4032, 6048

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # ETH3D data parameter
            min_depth=1e-5,
            max_depth=torch.inf,
            has_filled_depth=False,
            name_mode=PerceptionFileNameMode.id,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        # Read special binary data: https://www.eth3d.net/documentation#format-of-multi-view-data-image-formats
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            binary_data = self.tar_obj.extractfile("./" + rel_path)
            binary_data = binary_data.read()

        else:
            depth_path = os.path.join(self.dataset_dir, rel_path)
            with open(depth_path, "rb") as file:
                binary_data = file.read()
        # Convert the binary data to a numpy array of 32-bit floats
        depth_decoded = np.frombuffer(binary_data, dtype=np.float32).copy()

        depth_decoded[depth_decoded == torch.inf] = 0.0

        depth_decoded = depth_decoded.reshape((self.HEIGHT, self.WIDTH))
        return depth_decoded
