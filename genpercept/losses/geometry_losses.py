"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn
import numpy as np

COS_EPS = 1e-7

def align_scale_median_torch_batch(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    
    B, H, W = pred.shape
    # if len(mask.shape) == 3:
    #     mask = mask[:, None]
    #     mask = mask.repeat(1, target.shape[1], 1, 1)

    mask = mask & (target > 0)

    # Reshape for processing
    pred = pred.view(B, -1)
    target = target.view(B, -1)
    mask = mask.view(B, -1)

    # Applying mask
    pred_masked = torch.where(mask, pred, torch.tensor(float('nan'), device=pred.device))
    target_masked = torch.where(mask, target, torch.tensor(float('nan'), device=target.device))

    # Computing the parameters using median
    pred_median = torch.nanmedian(pred_masked, dim=1).values
    target_median = torch.nanmedian(target_masked, dim=1).values

    # Handle cases where mask is all False (all values in pred_median or target_median are NaN)
    median_nan_mask = torch.isnan(pred_median) | torch.isnan(target_median)
    pred_median = torch.where(median_nan_mask, torch.tensor(1.0, device=pred.device), pred_median)
    target_median = torch.where(median_nan_mask, torch.tensor(1.0, device=target.device), target_median)

    scale = target_median / (pred_median + 1e-8)

    return scale.view(B, 1, 1) #.repeat(1, C, H, W)


def compute_scale_and_shift(prediction, target, mask):
    try:
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    except Exception as e:
        x_0 = torch.zeros(prediction.shape[0]).to(prediction.dtype).to(prediction.device)
        x_1 = torch.zeros(prediction.shape[0]).to(prediction.dtype).to(prediction.device)
        # import pdb; pdb.set_trace()
        print('warning! There exists compute scale shift invald data!!! continue...')
        print(e)

    return x_0, x_1


def match_scale_and_shift(prediction, target):
    # based on implementation from
    # https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0

    assert len(target.shape) == len(prediction.shape)
    if len(target.shape) == 4:
        four_chan = True
        target = target.squeeze(dim=1)
        prediction = prediction.squeeze(dim=1)
    else:
        four_chan = False

    mask = (target > 0).float()

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 *
    # a_10) . b
    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    # compute scale and shift
    scale = torch.ones_like(b_0)
    shift = torch.zeros_like(b_1)
    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    scale = scale.view(-1, 1, 1).detach()
    shift = shift.view(-1, 1, 1).detach()
    prediction = prediction * scale + shift

    return prediction[:, None, :, :] if four_chan else prediction


def align_scale_median_torch_batch(pred: torch.Tensor, target: torch.Tensor, mask):
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    
    B, C, H, W = pred.shape
    if len(mask.shape) == 3:
        mask = mask[:, None]
        mask = mask.repeat(1, target.shape[1], 1, 1)

    mask = mask & (target > 0)

    # Reshape for processing
    pred = pred.view(B, -1)
    target = target.view(B, -1)
    mask = mask.view(B, -1)

    # Applying mask
    pred_masked = torch.where(mask, pred, torch.tensor(float('nan'), device=pred.device))
    target_masked = torch.where(mask, target, torch.tensor(float('nan'), device=target.device))

    # Computing the parameters using median
    pred_median = torch.nanmedian(pred_masked, dim=1).values
    target_median = torch.nanmedian(target_masked, dim=1).values

    # Handle cases where mask is all False (all values in pred_median or target_median are NaN)
    median_nan_mask = torch.isnan(pred_median) | torch.isnan(target_median)
    pred_median = torch.where(median_nan_mask, torch.tensor(1.0, device=pred.device), pred_median)
    target_median = torch.where(median_nan_mask, torch.tensor(1.0, device=target.device), target_median)

    scale = target_median / (pred_median + 1e-8)

    return scale.view(B, 1, 1, 1).repeat(1, C, H, W)


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, align_type='medium'):
        super().__init__()
        self.name = "SSILoss"
        self.align_type = align_type
        self.eps = 1e-6

    def ssi_mae(self, target, prediction, mask):
        valid_pixes = torch.sum(mask) + self.eps

        bs = target.shape[0]
        gt_median = torch.median(target.reshape(bs,-1),dim=1)[0]
        gt_s = torch.abs(target - gt_median[...,None,None]).reshape(bs, -1).sum(1) / (mask.reshape(bs, -1).sum(1) + self.eps)
        gt_trans = (target - gt_median[...,None,None]) / (gt_s[...,None,None] + self.eps)

        pred_median = torch.median(prediction.reshape(bs,-1),dim=1)[0]
        pred_s = torch.abs(prediction - pred_median[...,None,None]).reshape(bs, -1).sum(1) / (mask.reshape(bs, -1).sum(1) + self.eps)
        pred_trans = (prediction - pred_median[...,None,None]) / (pred_s[...,None,None] + self.eps)

        # gt_median = torch.median(target) if target.numel() else 0
        # gt_s = torch.abs(target - gt_median).sum() / valid_pixes
        # import pdb;pdb.set_trace()
        # gt_trans = (target - gt_median) / (gt_s + self.eps)

        # pred_median = torch.median(prediction) if prediction.numel() else 0
        # pred_s = torch.abs(prediction - pred_median).sum() / valid_pixes
        # pred_trans = (prediction - pred_median) / (pred_s + self.eps)

        return pred_trans, gt_trans
        
        # ssi_mae_sum = torch.sum(torch.abs(gt_trans - pred_trans))
        # return ssi_mae_sum, valid_pixes


    def forward(
        self, 
        prediction, 
        target, 
        mask, 
        interpolate=True, 
        return_interpolated=False,
        **kwargs):
        
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction

        # prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        B, C, H, W = prediction.shape
        assert C == 1

        if self.align_type == 'least_square':

            scale, shift = compute_scale_and_shift(prediction[:, 0], target[:, 0], mask[:, 0])
            scale = scale.view(-1, 1, 1, 1).repeat(1, C, H, W)
            shift = shift.view(-1, 1, 1, 1).repeat(1, C, H, W)
            # scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
            scale_median = align_scale_median_torch_batch(prediction, target, mask)  # [B, C, H, W]
            median_mask = (scale <= 0)
            scaled_prediction = prediction.clone()
            scaled_prediction[~median_mask] = scaled_prediction[~median_mask] * scale[~median_mask] + shift[~median_mask]
            scaled_prediction[median_mask] = scaled_prediction[median_mask] * scale_median[median_mask]
            loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])

        elif self.align_type == 'medium':
            prediction = prediction[:, 0]
            target = target[:, 0]
            mask = mask[:, 0]

            pred_trans, gt_trans = self.ssi_mae(prediction, target, mask)
            loss = nn.functional.l1_loss(pred_trans[mask], gt_trans[mask])
            # import pdb;pdb.set_trace()

        else:
            raise NotImplementedError

        return loss
        # if not return_interpolated:
        #     return loss, scaled_prediction

        # return loss, intr_input


# loss_name = "L1"
def l1_loss(norm_out, gt_norm, gt_norm_mask): 
    """ norm_out:       (B, 3, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)
    """
    pred_norm = norm_out[:, 0:3, ...]

    l1 = torch.sum(torch.abs(gt_norm - pred_norm), dim=1, keepdim=True)      # (B, 1, ...)
    l1 = l1[gt_norm_mask]
    return torch.mean(l1)


# loss_name = "L2"
def l2_loss(norm_out, gt_norm, gt_norm_mask): 
    """ norm_out:       (B, 3, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)   
    """
    pred_norm = norm_out[:, 0:3, ...]

    l2 = torch.sum(torch.square(gt_norm - pred_norm), dim=1, keepdim=True)   # (B, 1, ...)
    l2 = l2[gt_norm_mask]
    return torch.mean(l2)


# loss_name = "AL"
def angular_loss_org(norm_out, gt_norm, gt_norm_mask): 
    """ norm_out:       (B, 3, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)   

    """
    pred_norm = norm_out[:, 0:3, ...]
    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1).unsqueeze(1)    
    valid_mask = torch.logical_and(gt_norm_mask, torch.abs(dot.detach()) < 1-COS_EPS)
    angle = torch.acos(dot[valid_mask])
    return torch.mean(angle)


# loss_name = "NLL_vonmf"
def nll_vonmf(dot, pred_kappa):
    loss = - torch.log(pred_kappa) \
            - (pred_kappa * (dot - 1)) \
            + torch.log(1 - torch.exp(- 2 * pred_kappa))
    return loss


def vonmf_loss(norm_out, gt_norm, gt_norm_mask):
    """ norm_out:       (B, 4, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)   
    """
    pred_norm, pred_kappa = norm_out[:, 0:3, ...], norm_out[:, 3:, ...]

    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1).unsqueeze(1)    
    valid_mask = torch.logical_and(gt_norm_mask, torch.abs(dot.detach()) < 1-COS_EPS)

    # compute the loss
    nll = nll_vonmf(dot[valid_mask], pred_kappa[valid_mask])
    return torch.mean(nll)

# loss_name = "NLL_angmf"
def nll_angmf(dot, pred_kappa):
    loss = - torch.log(torch.square(pred_kappa) + 1) \
            + pred_kappa * torch.acos(dot) \
            + torch.log(1 + torch.exp(-pred_kappa * np.pi))
    return loss


def angmf_loss(norm_out, gt_norm, gt_norm_mask):
    """ norm_out:       (B, 4, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)   
    """
    pred_norm, pred_kappa = norm_out[:, 0:3, ...], norm_out[:, 3:, ...]

    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1).unsqueeze(1)
    valid_mask = torch.logical_and(gt_norm_mask, torch.abs(dot.detach()) < 1-COS_EPS)

    # compute the loss
    nll = nll_angmf(dot[valid_mask], pred_kappa[valid_mask])

    loss_mean = torch.mean(nll)

    if loss_mean != loss_mean:
        breakpoint()

    return loss_mean


def depth_si_loss(depth_pr, depth_gt, alpha=10, lambda_scale=0.85, eps=1e-5):
    """
    Based on the loss proposed by Eigen et al (NeurIPS 2014). This differs from the
    implementation used by PixelFormer in that the sqrt is applied per image before
    mean as opposed to compute the mean loss before square-root.
    """
    assert depth_pr.shape == depth_gt.shape, f"{depth_pr.shape} != {depth_gt.shape}"

    valid = (depth_gt > 0).detach().float()
    num_valid = valid.sum(dim=(-1, -2)).clamp(min=1)

    depth_pr = depth_pr.clamp(min=eps).log()
    depth_gt = depth_gt.clamp(min=eps).log()
    diff = (depth_pr - depth_gt) * valid
    diff_mean = diff.pow(2).sum(dim=(-2, -1)) / num_valid
    diff_var = diff.sum(dim=(-2, -1)).pow(2) / num_valid.pow(2)
    loss = alpha * (diff_mean - lambda_scale * diff_var).sqrt().mean()

    return loss


def sig_loss(depth_pr, depth_gt, sigma=0.85, eps=0.001, only_mean=False):
    """
    SigLoss
        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.
        adapated from DINOv2 code

    Args:
        depth_pr (FloatTensor): predicted depth
        depth_gt (FloatTensor): groundtruth depth
        eps (float): to avoid exploding gradient
    """
    # ignore invalid depth pixels
    valid = depth_gt > 0
    depth_pr = depth_pr[valid]
    depth_gt = depth_gt[valid]

    g = torch.log(depth_pr + eps) - torch.log(depth_gt + eps)

    loss = g.pow(2).mean() - sigma * g.mean().pow(2)
    loss = loss.sqrt()
    return loss


def gradient_loss(depth_pr, depth_gt, eps=0.001):
    """GradientLoss.

    Adapted from https://www.cs.cornell.edu/projects/megadepth/ and DINOv2 repo

    Args:
        depth_pr (FloatTensor): predicted depth
        depth_gt (FloatTensor): groundtruth depth
        eps (float): to avoid exploding gradient
    """
    # import pdb;pdb.set_trace()
    depth_pr_downscaled = [depth_pr] + [
        depth_pr[:, :, :: 2 * i, :: 2 * i] for i in range(1, 3)
    ]
    depth_gt_downscaled = [depth_gt] + [
        depth_gt[:, :, :: 2 * i, :: 2 * i] for i in range(1, 3)
    ]

    # import pdb;pdb.set_trace()

    gradient_loss = 0
    for depth_pr, depth_gt in zip(depth_pr_downscaled, depth_gt_downscaled):

        # ignore invalid depth pixels
        valid = depth_gt > 0
        N = torch.sum(valid)

        depth_pr_log = torch.log(depth_pr + eps)
        depth_gt_log = torch.log(depth_gt + eps)
        log_d_diff = depth_pr_log - depth_gt_log

        log_d_diff = torch.mul(log_d_diff, valid)

        v_gradient = torch.abs(log_d_diff[0:-2, :] - log_d_diff[2:, :])
        v_valid = torch.mul(valid[0:-2, :], valid[2:, :])
        v_gradient = torch.mul(v_gradient, v_valid)

        h_gradient = torch.abs(log_d_diff[:, 0:-2] - log_d_diff[:, 2:])
        h_valid = torch.mul(valid[:, 0:-2], valid[:, 2:])
        h_gradient = torch.mul(h_gradient, h_valid)

        gradient_loss += (torch.sum(h_gradient) + torch.sum(v_gradient)) / N

    return gradient_loss


def sig_loss_wo_log(depth_pr, depth_gt, sigma=0.85, eps=0.001, only_mean=False):
    """
    SigLoss
        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.
        adapated from DINOv2 code

    Args:
        depth_pr (FloatTensor): predicted depth
        depth_gt (FloatTensor): groundtruth depth
        eps (float): to avoid exploding gradient
    """
    # ignore invalid depth pixels
    valid = depth_gt > 0
    depth_pr = depth_pr[valid]
    depth_gt = depth_gt[valid]

    g = depth_pr - depth_gt

    # g = torch.log(depth_pr + eps) - torch.log(depth_gt + eps)
    loss = g.pow(2).mean() - sigma * g.mean().pow(2)
    loss = loss.sqrt()
    return loss


def gradient_loss_wo_log(depth_pr_log, depth_gt_log, eps=0.001):
    """GradientLoss.

    Adapted from https://www.cs.cornell.edu/projects/megadepth/ and DINOv2 repo

    Args:
        depth_pr (FloatTensor): predicted depth
        depth_gt (FloatTensor): groundtruth depth
        eps (float): to avoid exploding gradient
    """
    depth_pr_downscaled = [depth_pr_log] + [
        depth_pr_log[:, :, :: 2 * i, :: 2 * i] for i in range(1, 3)
    ]
    depth_gt_downscaled = [depth_gt_log] + [
        depth_gt_log[:, :, :: 2 * i, :: 2 * i] for i in range(1, 3)
    ]

    gradient_loss = 0
    for depth_pr_log, depth_gt_log in zip(depth_pr_downscaled, depth_gt_downscaled):

        # ignore invalid depth pixels
        valid = depth_gt_log > 0
        N = torch.sum(valid)

        # depth_pr_log = torch.log(depth_pr + eps)
        # depth_gt_log = torch.log(depth_gt + eps)

        log_d_diff = depth_pr_log - depth_gt_log
        log_d_diff = torch.mul(log_d_diff, valid)

        v_gradient = torch.abs(log_d_diff[0:-2, :] - log_d_diff[2:, :])
        v_valid = torch.mul(valid[0:-2, :], valid[2:, :])
        v_gradient = torch.mul(v_gradient, v_valid)

        h_gradient = torch.abs(log_d_diff[:, 0:-2] - log_d_diff[:, 2:])
        h_valid = torch.mul(valid[:, 0:-2], valid[:, 2:])
        h_gradient = torch.mul(h_gradient, h_valid)

        gradient_loss += (torch.sum(h_gradient) + torch.sum(v_gradient)) / N

    return gradient_loss


class DepthLoss(nn.Module):
    def __init__(self, weight_sig=2.0, weight_grad=0.5, max_depth=10, enable_log_normalization=True):
        super().__init__()
        self.sig_w = weight_sig
        self.grad_w = weight_grad
        self.max_depth = max_depth
        self.enable_log_normalization = enable_log_normalization

    def forward(self, pred, target, valid_mask):

        pred = (pred + 1) / 2 * self.max_depth
        target = (target + 1) / 2 * self.max_depth
        target[valid_mask == 0] = 0
        # 0 out max depth so it gets ignored
        # target[target > self.max_depth] = 0
        # import pdb;pdb.set_trace()
        # if self.enable_log_normalization:
        loss_s = self.sig_w * sig_loss_wo_log(pred, target)
        loss_g = self.grad_w * gradient_loss_wo_log(pred, target)            
        # else:
        #     loss_s = self.sig_w * sig_loss(pred, target)
        #     loss_g = self.grad_w * gradient_loss(pred, target)

        return loss_s, loss_g


class Probe3DDepthLoss(nn.Module):
    def __init__(self, weight_sig=10.0, weight_grad=0.5, max_depth=10):
        '''
        https://github.com/mbanani/probe3d/blob/main/evals/utils/losses.py
        '''
        super().__init__()
        self.sig_w = weight_sig
        self.grad_w = weight_grad
        self.max_depth = max_depth

    def forward(self, pred, target, valid_mask):
        # 0 out max depth so it gets ignored
        # target[target > self.max_depth] = 0
        target[valid_mask == 0] = 0

        loss_s = self.sig_w * sig_loss(pred, target)
        loss_g = self.grad_w * gradient_loss(pred, target)

        return loss_s + loss_g


def angular_loss(prediction, target, mask, uncertainty_aware=False, eps=1e-4, **kwargs):
    """
    Angular loss with uncertainty aware component based on Bae et al.
    """
    # ensure mask is float and batch x height x width
    assert mask.ndim == 4, f"mask should be (batch x height x width) not {mask.shape}"
    if mask.shape[1] != 1:
        mask = mask[:, 0]
    else:
        mask = mask.squeeze(1).float()

    # import pdb;pdb.set_trace()
    if prediction.shape[1] == 4:
        uncertainty_aware=True

    # import pdb;pdb.set_trace()
    # compute correct loss
    if uncertainty_aware:
        assert prediction.shape[1] == 4
        loss_ang = torch.cosine_similarity(prediction[:, :3], target, dim=1)
        loss_ang = loss_ang.clamp(min=-1 + eps, max=1 - eps).acos()

        # apply elu and add 1.01 to have a min kappa of 0.01 (similar to paper)
        kappa = torch.nn.functional.elu(prediction[:, 3]) + 1.01
        kappa_reg = (1 + (-kappa * torch.pi).exp()).log() - (kappa.pow(2) + 1).log()

        loss = kappa_reg + kappa * loss_ang
    else:
        assert prediction.shape[1] == 3
        loss_ang = torch.cosine_similarity(prediction, target, dim=1)
        loss = loss_ang.clamp(min=-1 + eps, max=1 - eps).acos()

        # valid_mask = torch.logical_and(gt_norm_mask, torch.abs(dot.detach()) < 1-COS_EPS)

    # compute loss over valid position
    loss_mean = loss[mask.bool()].mean()

    if loss_mean != loss_mean:
        breakpoint()

    return loss_mean


def snorm_l1_loss(prediction, target, mask, eps=1e-4):
    """
    Angular loss with uncertainty aware component based on Bae et al.
    """
    # ensure mask is float and batch x height x width
    assert mask.ndim == 4, f"mask should be (batch x height x width) not {mask.shape}"
    mask = mask.squeeze(1).float()

    assert prediction.shape[1] == 3
    loss = torch.nn.functional.l1_loss(prediction, target, reduction="none")
    loss = loss.mean(dim=1)

    # compute loss over valid position
    loss_mean = loss[mask.bool()].mean()
    if loss_mean != loss_mean:
        breakpoint()
    return loss_mean