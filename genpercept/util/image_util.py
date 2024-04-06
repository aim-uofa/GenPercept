import matplotlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def norm_to_rgb(norm):
    # norm: (3, H, W), range from [-1, 1]
    norm_rgb = ((norm + 1) * 0.5) * 255
    norm_rgb = np.clip(norm_rgb, a_min=0, a_max=255)
    norm_rgb = norm_rgb.astype(np.uint8)
    return norm_rgb

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = np.squeeze(depth_map.copy())
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = np.squeeze(valid_mask)  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


def resize_max_res(img: Image.Image, max_edge_resolution: int) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio

    Args:
        img (Image.Image): Image to be resized
        max_edge_resolution (int): Maximum edge length (px).

    Returns:
        Image.Image: Resized image.
    """
    original_width, original_height = img.size
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = img.resize((new_width, new_height))
    return resized_img

def resize_max_res_integer_16(img: Image.Image, max_edge_resolution: int) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio

    Args:
        img (Image.Image): Image to be resized
        max_edge_resolution (int): Maximum edge length (px).

    Returns:
        Image.Image: Resized image.
    """
    original_width, original_height = img.size
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor) // 16 * 16 # make sure it is integer multiples of 16, used for pixart
    new_height = int(original_height * downscale_factor) // 16 * 16 # make sure it is integer multiples of 16, used for pixart

    resized_img = img.resize((new_width, new_height))
    return resized_img

def resize_res(img: Image.Image, max_edge_resolution: int) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio

    Args:
        img (Image.Image): Image to be resized
        max_edge_resolution (int): Maximum edge length (px).

    Returns:
        Image.Image: Resized image.
    """

    resized_img = img.resize((max_edge_resolution, max_edge_resolution))
    return resized_img

class ResizeLongestEdge:
    def __init__(self, max_size, interpolation=transforms.InterpolationMode.BILINEAR):
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img):

        scale = self.max_size / max(img.width, img.height)
        new_size = (int(img.height * scale), int(img.width * scale))

        return transforms.functional.resize(img, new_size, self.interpolation)

class ResizeShortestEdge:
    def __init__(self, min_size, interpolation=transforms.InterpolationMode.BILINEAR):
        self.min_size = min_size
        self.interpolation = interpolation

    def __call__(self, img):

        scale = self.min_size / min(img.width, img.height)
        new_size = (int(img.height * scale), int(img.width * scale))

        return transforms.functional.resize(img, new_size, self.interpolation)

class ResizeHard:
    def __init__(self, size, interpolation=transforms.InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):

        new_size = (int(self.size), int(self.size))

        return transforms.functional.resize(img, new_size, self.interpolation)


class ResizeLongestEdgeInteger:
    def __init__(self, max_size, interpolation=transforms.InterpolationMode.BILINEAR, integer=16):
        self.max_size = max_size
        self.interpolation = interpolation
        self.integer = integer

    def __call__(self, img):

        scale = self.max_size / max(img.width, img.height)
        new_size_h = int(img.height * scale) // self.integer * self.integer
        new_size_w = int(img.width * scale) // self.integer * self.integer
        new_size = (new_size_h, new_size_w)

        return transforms.functional.resize(img, new_size, self.interpolation)