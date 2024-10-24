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

import io
import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import os.path as osp
import pandas as pd
import random
import tarfile
from enum import Enum
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize

from src.util.depth_transform import DepthNormalizerBase


class DatasetMode(Enum):
    RGB_ONLY = "rgb_only"
    EVAL = "evaluate"
    TRAIN = "train"


class PerceptionFileNameMode(Enum):
    """Prediction file naming modes"""

    id = 1  # id.png
    rgb_id = 2  # rgb_id.png
    i_d_rgb = 3  # i_d_1_rgb.png
    rgb_i_d = 4


def read_image_from_tar(tar_obj, img_rel_path):
    image = tar_obj.extractfile("./" + img_rel_path)
    image = image.read()
    image = Image.open(io.BytesIO(image))


class BaseDataset(Dataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        dataset_dir: str,
        disp_name: str,
        min_depth: float = 0,
        max_depth: float = 1e8,
        has_filled_depth: bool = False,
        name_mode: PerceptionFileNameMode = PerceptionFileNameMode.id,
        depth_transform: Union[DepthNormalizerBase, None] = None,
        augmentation_args: dict = None,
        resize_to_hw=None,
        move_invalid_to_far_plane: bool = True,
        rgb_transform=lambda x: x / 255.0 * 2 - 1,  #  [0, 255] -> [-1, 1],
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        # dataset info
        self.filename_ls_path = filename_ls_path
        self.dataset_dir = dataset_dir
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset does not exist at: {self.dataset_dir}"
        self.disp_name = disp_name
        self.has_filled_depth = has_filled_depth
        self.name_mode: PerceptionFileNameMode = name_mode
        self.min_depth = min_depth
        self.max_depth = max_depth

        # training arguments
        self.depth_transform: DepthNormalizerBase = depth_transform
        self.augm_args = augmentation_args
        self.resize_to_hw = resize_to_hw
        self.rgb_transform = rgb_transform
        self.move_invalid_to_far_plane = move_invalid_to_far_plane

        if filename_ls_path.endswith('.txt'):
            # Load filenames
            with open(self.filename_ls_path, "r") as f:
                self.filenames = [
                    s.split() for s in f.readlines()
                ]  # [['rgb.png', 'depth.tif'], [], ...]
            self.is_exr_data = False
        elif osp.isdir(filename_ls_path): # diffusers dataset
            filenames = os.listdir(filename_ls_path)
            self.filenames = []
            # print('json files cnt :', len(filenames))
            for i, filename in enumerate(filenames):
                if not filename.endswith('.jsonl'):
                    continue
                metadata_path = osp.join(filename_ls_path, filename)
                metadata = pd.read_json(metadata_path, lines=True)

                for _, row in metadata.iterrows():
                    
                    image_path = row["image"]

                    if "depth_conditioning_image" in row.keys():
                        depth_path = row["depth_conditioning_image"]
                    else:      
                        depth_path = None

                    if "normal_conditioning_image" in row.keys():
                        normal_path = row["normal_conditioning_image"]
                    else:      
                        normal_path = None
                    
                    if "matting_conditioning_image" in row.keys():
                        matting_path = row["matting_conditioning_image"]
                    else:      
                        matting_path = None
                    
                    if "dis_conditioning_image" in row.keys():
                        dis_path = row["dis_conditioning_image"]
                    else:      
                        dis_path = None
                    
                    if "seg_conditioning_image" in row.keys():
                        seg_path = row["seg_conditioning_image"]
                    else:      
                        seg_path = None

                    data_row = [image_path, depth_path, None, normal_path, matting_path, dis_path, seg_path] # None for filled_rel_path
                    self.filenames.append(data_row)
            if depth_path is not None and depth_path.endswith('.exr'):
                self.is_exr_data = True
            else:
                self.is_exr_data = False
        else:
            raise NotImplementedError

        # Tar dataset
        self.tar_obj = None
        self.is_tar = (
            True
            if os.path.isfile(dataset_dir) and tarfile.is_tarfile(dataset_dir)
            else False
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, filled_rel_path, normal_rel_path, matting_rel_path, dis_rel_path, seg_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        if DatasetMode.RGB_ONLY != self.mode:
            # Depth
            if depth_rel_path is not None:
                depth_data = self._load_depth_data(
                    depth_rel_path=depth_rel_path, filled_rel_path=filled_rel_path, shape=(rasters["rgb_norm"].shape[1], rasters["rgb_norm"].shape[2]),
                )
                rasters.update(depth_data)

                # valid mask
                rasters["valid_mask_raw"] = self._get_valid_mask(
                    rasters["depth_raw_linear"]
                ).clone()

            if self.has_filled_depth:
                rasters["valid_mask_filled"] = self._get_valid_mask(
                    rasters["depth_filled_linear"]
                ).clone()

            # Normal
            if normal_rel_path is not None:
                normal_data = self._load_normal_data(
                    normal_rel_path=normal_rel_path, shape=(rasters["rgb_norm"].shape[1], rasters["rgb_norm"].shape[2]),
                )
                rasters.update(normal_data)
                rasters["valid_mask_raw_normal"] = self._get_valid_mask_normal(
                    rasters["normal_raw_linear"]
                ).clone()

            # Matting
            if matting_rel_path is not None:
                matting_data = self._load_matting_data(
                    matting_rel_path=matting_rel_path, shape=(rasters["rgb_norm"].shape[1], rasters["rgb_norm"].shape[2]),
                )
                rasters.update(matting_data)
                rasters["valid_mask_raw_matting"] = self._get_valid_mask_matting(
                    rasters["matting_raw_linear"]
                ).clone()

            # DIS
            if dis_rel_path is not None:
                dis_data = self._load_dis_data(
                    dis_rel_path=dis_rel_path, shape=(rasters["rgb_norm"].shape[1], rasters["rgb_norm"].shape[2]),
                )
                rasters.update(dis_data)
                rasters["valid_mask_raw_dis"] = self._get_valid_mask_dis(
                    rasters["dis_raw_linear"]
                ).clone()
            
            # Seg
            if seg_rel_path is not None:
                seg_data = self._load_seg_data(
                    seg_rel_path=seg_rel_path, shape=(rasters["rgb_norm"].shape[1], rasters["rgb_norm"].shape[2]),
                )
                rasters.update(seg_data)
                rasters["valid_mask_raw_seg"] = self._get_valid_mask_seg(
                    rasters["seg_raw_linear"]
                ).clone()

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other

    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

        outputs = {
            "rgb_int": torch.from_numpy(rgb).int(),
            "rgb_norm": torch.from_numpy(rgb_norm).float(),
        }
        return outputs

    def _load_depth_data(self, depth_rel_path, filled_rel_path, shape=None):
        # Read depth data
        outputs = {}
        try:
            depth_raw = self._read_depth_file(depth_rel_path).squeeze()
            depth_raw_linear = torch.from_numpy(depth_raw).float().unsqueeze(0)  # [1, H, W]
            outputs["depth_raw_linear"] = depth_raw_linear.clone()
        except:
            depth_raw = np.array([-1.,-1.,-1.])[:, None, None]
            depth_raw = np.repeat(depth_raw, shape[0], axis=1)
            depth_raw = np.repeat(depth_raw, shape[1], axis=2)
            depth_raw_linear = torch.from_numpy(depth_raw).float()  # [3, H, W]
            outputs["depth_raw_linear"] = depth_raw_linear.clone()

        if self.has_filled_depth:
            depth_filled = self._read_depth_file(filled_rel_path).squeeze()
            depth_filled_linear = torch.from_numpy(depth_filled).float().unsqueeze(0)
            outputs["depth_filled_linear"] = depth_filled_linear
        else:
            pass

        return outputs

    def _load_normal_data(self, normal_rel_path, shape):
        # Read depth data
        outputs = {}
        try:
            normal_raw = self._read_image(normal_rel_path).squeeze()
            normal_raw = np.transpose(normal_raw, (2, 0, 1))  # [rgb, H, W]
            normal_raw_linear = torch.from_numpy(normal_raw).float()  # [3, H, W]
            outputs["normal_raw_linear"] = normal_raw_linear.clone()
        except:
            normal_np = np.array([0.,0.,0.])[:, None, None]
            normal_np = np.repeat(normal_np, shape[0], axis=1)
            normal_raw = np.repeat(normal_np,  shape[1], axis=2)
            normal_raw_linear = torch.from_numpy(normal_raw).float()  # [3, H, W]
            outputs["normal_raw_linear"] = normal_raw_linear.clone()

        return outputs

    def _load_matting_data(self, matting_rel_path, shape):
        # Read depth data
        outputs = {}
        try:
            matting_raw = self._read_image(matting_rel_path).squeeze()
            if len(matting_raw.shape) == 2:
                matting_raw = np.repeat(matting_raw[None], 3, axis=0)
            else:
                matting_raw = np.transpose(matting_raw, (2, 0, 1))  # [rgb, H, W]
            matting_raw_linear = torch.from_numpy(matting_raw).float()  # [3, H, W]
            outputs["matting_raw_linear"] = matting_raw_linear.clone()
        except:
            matting_np = np.array([-1.,-1.,-1.])[:, None, None]
            matting_np = np.repeat(matting_np, shape[0], axis=1)
            matting_raw = np.repeat(matting_np,  shape[1], axis=2)
            matting_raw_linear = torch.from_numpy(matting_raw).float()  # [3, H, W]
            outputs["matting_raw_linear"] = matting_raw_linear.clone()

        return outputs
    
    def _load_dis_data(self, dis_rel_path, shape):
        # Read depth data
        outputs = {}
        try:
            dis_raw = self._read_image(dis_rel_path).squeeze()
            if len(dis_raw.shape) == 2:
                dis_raw = np.repeat(dis_raw[None], 3, axis=0)
            else:
                dis_raw = np.transpose(dis_raw, (2, 0, 1))  # [rgb, H, W]
            dis_raw_linear = torch.from_numpy(dis_raw).float()  # [3, H, W]
            outputs["dis_raw_linear"] = dis_raw_linear.clone()
        except:
            dis_np = np.array([-1.,-1.,-1.])[:, None, None]
            dis_np = np.repeat(dis_np, shape[0], axis=1)
            dis_raw = np.repeat(dis_np,  shape[1], axis=2)
            dis_raw_linear = torch.from_numpy(dis_raw).float()  # [3, H, W]
            outputs["dis_raw_linear"] = dis_raw_linear.clone()

        return outputs

    def _load_seg_data(self, seg_rel_path, shape):
        # Read depth data
        outputs = {}
        try:
            seg_raw = self._read_image(seg_rel_path, convert_rgb=True).squeeze()
            seg_raw = np.transpose(seg_raw, (2, 0, 1))  # [rgb, H, W]
            seg_raw_linear = torch.from_numpy(seg_raw).float()  # [3, H, W]
            outputs["seg_raw_linear"] = seg_raw_linear.clone()
        except:
            seg_np = np.array([-1.,-1.,-1.])[:, None, None]
            seg_np = np.repeat(seg_np, shape[0], axis=1)
            seg_raw = np.repeat(seg_np,  shape[1], axis=2)
            seg_raw_linear = torch.from_numpy(seg_raw).float()  # [3, H, W]
            outputs["seg_raw_linear"] = seg_raw_linear.clone()

        return outputs

    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        rgb_rel_path = filename_line[0]

        depth_rel_path, filled_rel_path, normal_rel_path, matting_rel_path, dis_rel_path, seg_rel_path = None, None, None, None, None, None
        if DatasetMode.RGB_ONLY != self.mode:
            depth_rel_path = filename_line[1]
            if self.has_filled_depth:
                filled_rel_path = filename_line[2]
            
            if len(filename_line) > 3:
                normal_rel_path = filename_line[3]
                matting_rel_path = filename_line[4]
                dis_rel_path = filename_line[5]
                seg_rel_path = filename_line[6]

        return rgb_rel_path, depth_rel_path, filled_rel_path, normal_rel_path, matting_rel_path, dis_rel_path, seg_rel_path

    def _read_image(self, img_rel_path, convert_rgb=False) -> np.ndarray:
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            image_to_read = self.tar_obj.extractfile("./" + img_rel_path)
            image_to_read = image_to_read.read()
            image_to_read = io.BytesIO(image_to_read)
        else:
            image_to_read = os.path.join(self.dataset_dir, img_rel_path)
        if image_to_read.endswith('.exr'):
            image = cv2.imread(image_to_read, cv2.IMREAD_UNCHANGED)
            if len(image.shape) == 2:
                image = np.repeat(image[:, :, None], 3, axis=2).copy()
            elif len(image.shape) == 3 and image.shape[2] < 10:
                image = image[...,:3][...,::-1].copy()
            else:
                raise ValueError
        else:
            image = Image.open(image_to_read)  # [H, W, rgb]
            if convert_rgb:
                image = image.convert("RGB")
            image = np.asarray(image)
        return image

    def _read_rgb_file(self, rel_path) -> np.ndarray:
        rgb = self._read_image(rel_path)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [rgb, H, W]
        return rgb

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        if len(depth_in.shape) == 3 and depth_in.shape[2] == 3:
            assert np.all(depth_in[:, :, 0] == depth_in[:, :, 1])
            assert np.all(depth_in[:, :, 0] == depth_in[:, :, 2])
            depth_in = depth_in[:, :, 0]
        #  Replace code below to decode depth according to dataset definition
        depth_decoded = depth_in

        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = torch.logical_and(
            (depth > self.min_depth), (depth < self.max_depth)
        ).bool()
        return valid_mask

    def _get_valid_mask_normal(self, normal: torch.Tensor):
        valid_mask_normal = (normal != 0).any(dim=(0))[None]
        return valid_mask_normal
    
    def _get_valid_mask_matting(self, matting: torch.Tensor):
        valid_mask_matting = (matting != -1).any(dim=(0))[None]
        return valid_mask_matting

    def _get_valid_mask_dis(self, dis: torch.Tensor):
        valid_mask_dis = (dis != -1).any(dim=(0))[None]
        return valid_mask_dis

    def _get_valid_mask_seg(self, seg: torch.Tensor):
        valid_mask_seg = (seg != -1).any(dim=(0))[None]
        return valid_mask_seg

    def _training_preprocess(self, rasters):
        # Augmentation
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)

        # Depth
        if "depth_raw_linear" in rasters.keys():
            # Depth Normalization
            rasters["depth_raw_norm"] = self.depth_transform(
                rasters["depth_raw_linear"], rasters["valid_mask_raw"]
            ).clone()
            del rasters["depth_raw_linear"]

        if self.has_filled_depth:
            rasters["depth_filled_norm"] = self.depth_transform(
                rasters["depth_filled_linear"], rasters["valid_mask_filled"]
            ).clone()
            del rasters["depth_filled_linear"]
        
        # Normal
        if "normal_raw_linear" in rasters.keys():
            # Normal Normalization
            if (rasters["valid_mask_raw"][0] == False).sum() > 0:
                rasters["normal_raw_linear"][:, ~rasters["valid_mask_raw"][0]] = torch.tensor([0.,0.,0.]).view(3, 1).float()
            norm = torch.sqrt(torch.sum(rasters["normal_raw_linear"] ** 2, dim=0, keepdim=True))
            norm = torch.clamp(norm, min=1e-8)
            rasters["normal_raw_norm"] = rasters["normal_raw_linear"] / norm
            del rasters["normal_raw_linear"]
            assert rasters["normal_raw_norm"].min() >= -1 and rasters["normal_raw_norm"].max() <= 1, print('min :', rasters["normal_raw_norm"].min(), 'max', rasters["normal_raw_norm"].max())

        # Matting
        if "matting_raw_linear" in rasters.keys():
            # Matting Normalization
            rasters["matting_raw_norm"] = (rasters["matting_raw_linear"] - rasters["matting_raw_linear"].min()) / torch.clamp((rasters["matting_raw_linear"].max() - rasters["matting_raw_linear"].min()), 1e-8) # [0, 1]
            rasters["matting_raw_norm"] = (rasters["matting_raw_norm"] - 0.5) * 2
            del rasters["matting_raw_linear"]
            assert rasters["matting_raw_norm"].min() >= -1 and rasters["matting_raw_norm"].max() <= 1, print('min :', rasters["matting_raw_norm"].min(), 'max', rasters["matting_raw_norm"].max())
        
        # DIS
        if "dis_raw_linear" in rasters.keys():
            # DIS Normalization
            rasters["dis_raw_norm"] = (rasters["dis_raw_linear"] - rasters["dis_raw_linear"].min()) / torch.clamp((rasters["dis_raw_linear"].max() - rasters["dis_raw_linear"].min()), 1e-8) # [0, 1]
            rasters["dis_raw_norm"] = (rasters["dis_raw_norm"] - 0.5) * 2
            del rasters["dis_raw_linear"]
            assert rasters["dis_raw_norm"].min() >= -1 and rasters["dis_raw_norm"].max() <= 1, print('min :', rasters["dis_raw_norm"].min(), 'max', rasters["dis_raw_norm"].max())

        # Seg
        if "seg_raw_linear" in rasters.keys():
            # Seg Normalization
            rasters["seg_raw_norm"] = (rasters["seg_raw_linear"] - rasters["seg_raw_linear"].min()) / torch.clamp((rasters["seg_raw_linear"].max() - rasters["seg_raw_linear"].min()), 1e-8) # [0, 1]
            rasters["seg_raw_norm"] = (rasters["seg_raw_norm"] - 0.5) * 2
            del rasters["seg_raw_linear"]
            assert rasters["seg_raw_norm"].min() >= -1 and rasters["seg_raw_norm"].max() <= 1, print('min :', rasters["seg_raw_norm"].min(), 'max', rasters["seg_raw_norm"].max())

        # Set invalid pixel to far plane
        if self.move_invalid_to_far_plane and self.has_filled_depth:
            if self.depth_transform.far_plane_at_max:
                rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_max
                )
            else:
                rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_min
                )

        # Resize
        if self.resize_to_hw is not None:

            for k, v in rasters.items():
                if 'matting' in k or 'dis' in k:
                    interpolation = interpolation=InterpolationMode.BILINEAR
                else:
                    try:
                        interpolation=InterpolationMode.NEAREST_EXACT
                    except:
                        interpolation=InterpolationMode.NEAREST
                resize_transform = Resize(
                    size=self.resize_to_hw, interpolation=interpolation
                )
                rasters[k] = resize_transform(v)

        return rasters

    def _augment_data(self, rasters_dict):
        # lr flipping
        lr_flip_p = self.augm_args.lr_flip_p
        if random.random() < lr_flip_p:
            rasters_dict = {k: v.flip(-1) for k, v in rasters_dict.items()}
            if "normal_raw_linear" in rasters_dict.keys():
                rasters_dict["normal_raw_linear"][0, :, :] = -rasters_dict["normal_raw_linear"][0, :, :]

        return rasters_dict

    def __del__(self):
        if hasattr(self, "tar_obj") and self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None


def get_pred_name(rgb_basename, name_mode, suffix=".png"):
    if PerceptionFileNameMode.rgb_id == name_mode:
        pred_basename = "pred_" + rgb_basename.split("_")[1]
    elif PerceptionFileNameMode.i_d_rgb == name_mode:
        pred_basename = rgb_basename.replace("_rgb.", "_pred.")
    elif PerceptionFileNameMode.id == name_mode:
        pred_basename = "pred_" + rgb_basename
    elif PerceptionFileNameMode.rgb_i_d == name_mode:
        pred_basename = "pred_" + "_".join(rgb_basename.split("_")[1:])
    else:
        raise NotImplementedError
    # change suffix
    pred_basename = os.path.splitext(pred_basename)[0] + suffix

    return pred_basename
