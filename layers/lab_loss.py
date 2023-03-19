from typing import Tuple

import torch
import torch.nn as nn
from kornia.color import rgb_to_lab


class LabLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(LabLoss, self).__init__()
        self.criterion = torch.nn.SmoothL1Loss(reduction=reduction)

    def __call__(self, fakeIm, realIm, mask=None, return_lab=True):
        fakeIm_lab = rgb_to_lab(fakeIm.contiguous())
        fakeIm_lab = self.normalize_lab(fakeIm_lab)  # [0, 1] NCHW
        realIm_lab = rgb_to_lab(realIm.contiguous())
        realIm_lab = self.normalize_lab(realIm_lab)  # [0, 1], NCHW
        fakeIm_lab_vis = fakeIm_lab.detach().clone()
        realIm_lab_vis = realIm_lab.detach().clone()

        # drop L channel, no lighting consideration
        fakeIm_lab_vis[:, 0] = realIm_lab_vis[:, 0]
        loss = self.criterion(fakeIm_lab[:, 1:], realIm_lab[:, 1:])

        # masking
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        if return_lab:
            return loss, fakeIm_lab_vis, realIm_lab_vis
        else:
            return loss

    @staticmethod
    def normalize_lab(lab: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lab: NCHW, normalize to [0,1]
        """
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-127, 127], not exact
        # [0, 100] => [0, 1],  ~[-127, 127] => [0, 1]
        _min = torch.tensor([0.0, -127.0, -127.0]).view(3, 1, 1).to(lab)
        _max = torch.tensor([100.0, 127.0, 127.0]).view(3, 1, 1).to(lab)
        lab_normed = (lab - _min) / (_max - _min)
        return lab_normed
