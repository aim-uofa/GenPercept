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

import os
import tarfile
from io import BytesIO

import numpy as np
import torch

from .base_dataset import BaseDataset, PerceptionFileNameMode, DatasetMode


class DIODEDataset(BaseDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # DIODE data parameter
            min_depth=0.6,
            max_depth=350,
            has_filled_depth=False,
            name_mode=PerceptionFileNameMode.id,
            **kwargs,
        )

    def _read_npy_file(self, rel_path):
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            fileobj = self.tar_obj.extractfile("./" + rel_path)
            npy_path_or_content = BytesIO(fileobj.read())
        else:
            npy_path_or_content = os.path.join(self.dataset_dir, rel_path)
        data = np.load(npy_path_or_content).squeeze()[np.newaxis, :, :]
        return data

    def _read_depth_file(self, rel_path):
        depth = self._read_npy_file(rel_path)
        return depth

    def _get_data_path(self, index):
        return self.filenames[index]

    def _get_data_item(self, index):
        # Special: depth mask is read from data

        # rgb_rel_path, depth_rel_path, mask_rel_path, normal_rel_path = self._get_data_path(index=index)
        rgb_rel_path, depth_rel_path, mask_rel_path= self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, filled_rel_path=None
            )
            rasters.update(depth_data)

            # valid mask
            mask = self._read_npy_file(mask_rel_path).astype(bool)
            mask = torch.from_numpy(mask).bool()
            rasters["valid_mask_raw"] = mask.clone()
            rasters["valid_mask_filled"] = mask.clone()

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other
