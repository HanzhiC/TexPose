# code modified based on GRAF: https://github.com/autonomousvision/graf
import torch
import math
import torch.nn.functional as F
import camera


class RaySampler(object):
    def __init__(self, opt, intrinsics=None):
        self.intrinsics = intrinsics

    @staticmethod
    def get_image(opt, coords, image, H=None, W=None):
        with torch.no_grad():
            batch_size, h, w, _ = coords.shape  # [B, h, w, 2]
            if H is None and W is None:
                _H, _W = opt.H, opt.W
            else:
                _H, _W = H, W
            image_sampled = F.grid_sample(image, coords, mode='bilinear', align_corners=True)
        return image_sampled

    @staticmethod
    def get_bounds(opt, coords, z_near, z_far, H=None, W=None):
        with torch.no_grad():
            batch_size, h, w, _ = coords.shape  # [B, h, w, 2]
            if H is None and W is None:
                _H, _W = opt.H, opt.W
            else:
                _H, _W = H, W

            # sampling near bound and far bound
            z_near = z_near.view(batch_size, _H, _W, 1).permute(0, 3, 1, 2)  # [B, 1, H, W]
            z_far = z_far.view(batch_size, _H, _W, 1).permute(0, 3, 1, 2)  # [B, 1, H, W]
            z_near_sampled = F.grid_sample(z_near, coords, mode='bilinear', align_corners=True)[:, 0]  # [B, h, w]
            z_far_sampled = F.grid_sample(z_far, coords, mode='bilinear', align_corners=True)[:, 0]  # [B, h, w]
        return z_near_sampled, z_far_sampled

    @staticmethod
    def get_rays(opt, intrinsics, coords, pose, H=None, W=None):
        with torch.no_grad():
            batch_size, h, w, _ = coords.shape  # [B, h, w, 2]
            if H is None and W is None:
                _H, _W = opt.H, opt.W
            else:
                _H, _W = H, W

            # sample 2D coordinates of the rays
            x_range = torch.arange(_W, dtype=torch.float32, device=pose.device)
            y_range = torch.arange(_H, dtype=torch.float32, device=pose.device)
            Y, X = torch.meshgrid(y_range, x_range)  # [H, W]
            Y = Y[None, None].repeat(batch_size, 1, 1, 1)  # [B, 1, H, W]
            X = X[None, None].repeat(batch_size, 1, 1, 1)  # [B, 1, H, W]

            u = F.grid_sample(X, coords, mode='bilinear', align_corners=True)[:, 0]  # [B, h, w]
            v = F.grid_sample(Y, coords, mode='bilinear', align_corners=True)[:, 0]  # [B, h, w]
            xy_grid = torch.stack([u, v], dim=-1).view(batch_size, h * w, -1)  # [B, hw, 2]

            # transform from camera to world coordinates, poses with shape [B, 3, 4]
            grid_3D = camera.img2cam(camera.to_hom(xy_grid), intrinsics)  # [B, hw, 3]
            center_3D = torch.zeros_like(grid_3D)  # [B, hw, 3]
            grid_3D = camera.cam2world(grid_3D, pose)  # [B, hw, 3]
            center_3D = camera.cam2world(center_3D, pose)  # [B, hw, 3]

            # shape back
            grid_3D = grid_3D.view(batch_size, h, w, -1)  # [B, h, w, 3]
            center_3D = center_3D.view(batch_size, h, w, -1)  # [B, h, w, 3]
            ray = grid_3D - center_3D  # [B, h, w, 3]
        return center_3D, ray
