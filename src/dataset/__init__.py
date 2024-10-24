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

from .base_dataset import BaseDataset, get_pred_name, DatasetMode  # noqa: F401
from .diode_dataset import DIODEDataset
from .eth3d_dataset import ETH3DDataset
from .hypersim_dataset import HypersimDataset
from .kitti_dataset import KITTIDataset
from .nyu_dataset import NYUDataset
from .scannet_dataset import ScanNetDataset
from .vkitti_dataset import VirtualKITTIDataset
from .p3m10k_dataset import P3M10KDataset
from .dis5k_dataset import DIS5KDataset
from .taskonomy_dataset import TaskonomyDataset
from .cityscapes_dataset import CityscapesDataset


dataset_name_class_dict = {
    "hypersim": HypersimDataset,
    "vkitti": VirtualKITTIDataset,
    "nyu_v2": NYUDataset,
    "kitti": KITTIDataset,
    "eth3d": ETH3DDataset,
    "diode": DIODEDataset,
    "scannet": ScanNetDataset,
    "p3m10k": P3M10KDataset,
    "dis5k": DIS5KDataset,
    "taskonomy": TaskonomyDataset,
    "cityscapes": CityscapesDataset,
}


def get_dataset(
    cfg_data_split, base_data_dir: str, mode: DatasetMode, **kwargs
) -> BaseDataset:
    if "mixed" == cfg_data_split.name:
        assert DatasetMode.TRAIN == mode, "Only training mode supports mixed datasets."
        dataset_ls = [
            get_dataset(_cfg, base_data_dir, mode, **kwargs)
            for _cfg in cfg_data_split.dataset_list
        ]
        return dataset_ls
    elif cfg_data_split.name in dataset_name_class_dict.keys():
        dataset_class = dataset_name_class_dict[cfg_data_split.name]
        dataset = dataset_class(
            mode=mode,
            filename_ls_path=cfg_data_split.filenames,
            dataset_dir=os.path.join(base_data_dir, cfg_data_split.dir),
            **cfg_data_split,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset
