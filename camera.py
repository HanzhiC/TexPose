import numpy as np
import os, sys, time
import torch
import torch.nn.functional as torch_F
import collections
from easydict import EasyDict as edict
import torch.nn.functional as F

import util
from util import log, debug


class Pose():
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self, R=None, t=None):
        # construct a camera pose from the given R and/or t
        assert (R is not None or t is not None)
        if R is None:
            if not isinstance(t, torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1], device=R.device)
        else:
            if not isinstance(R, torch.Tensor): R = torch.tensor(R)
            if not isinstance(t, torch.Tensor): t = torch.tensor(t)
        assert (R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R, t[..., None]], dim=-1)  # [...,3,4]
        assert (pose.shape[-2:] == (3, 4))
        return pose

    def invert(self, pose, use_inverse=False):
        # invert a camera pose
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new


class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self, w):  # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I + A * wx + B * wx @ wx
        return R

    def SO3_to_so3(self, R, eps=1e-7):  # [...,3,3]
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[
                    ..., None, None] % np.pi  # ln(R) will explode if theta==pi
        lnR = 1 / (2 * self.taylor_A(theta) + 1e-8) * (R - R.transpose(-2, -1))  # FIXME: wei-chiu finds it weird
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    def se3_to_SE3(self, wu):  # [...,3]
        w, u = wu.split([3, 3], dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I + A * wx + B * wx @ wx
        V = I + B * wx + C * wx @ wx
        Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
        return Rt

    def SE3_to_se3(self, Rt, eps=1e-8):  # [...,3,4]
        R, t = Rt.split([3, 1], dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta ** 2 + eps) * wx @ wx
        u = (invV @ t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu

    def skew_symmetric(self, w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O, -w2, w1], dim=-1),
                          torch.stack([w2, O, -w0], dim=-1),
                          torch.stack([-w1, w0, O], dim=-1)], dim=-2)
        return wx

    def taylor_A(self, x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            if i > 0: denom *= (2 * i) * (2 * i + 1)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_B(self, x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            denom *= (2 * i + 1) * (2 * i + 2)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_C(self, x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            denom *= (2 * i + 2) * (2 * i + 3)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans


class Quaternion():

    def q_to_R(self, q):
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        qa, qb, qc, qd = q.unbind(dim=-1)
        R = torch.stack(
            [torch.stack([1 - 2 * (qc ** 2 + qd ** 2), 2 * (qb * qc - qa * qd), 2 * (qa * qc + qb * qd)], dim=-1),
             torch.stack([2 * (qb * qc + qa * qd), 1 - 2 * (qb ** 2 + qd ** 2), 2 * (qc * qd - qa * qb)], dim=-1),
             torch.stack([2 * (qb * qd - qa * qc), 2 * (qa * qb + qc * qd), 1 - 2 * (qb ** 2 + qc ** 2)], dim=-1)],
            dim=-2)
        return R

    def R_to_q(self, R, eps=1e-8):  # [B,3,3]
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # FIXME: this function seems a bit problematic, need to double-check
        row0, row1, row2 = R.unbind(dim=-2)
        R00, R01, R02 = row0.unbind(dim=-1)
        R10, R11, R12 = row1.unbind(dim=-1)
        R20, R21, R22 = row2.unbind(dim=-1)
        t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        r = (1 + t + eps).sqrt()
        qa = 0.5 * r
        qb = (R21 - R12).sign() * 0.5 * (1 + R00 - R11 - R22 + eps).sqrt()
        qc = (R02 - R20).sign() * 0.5 * (1 - R00 + R11 - R22 + eps).sqrt()
        qd = (R10 - R01).sign() * 0.5 * (1 - R00 - R11 + R22 + eps).sqrt()
        q = torch.stack([qa, qb, qc, qd], dim=-1)
        for i, qi in enumerate(q):
            if torch.isnan(qi).any():
                K = torch.stack([torch.stack([R00 - R11 - R22, R10 + R01, R20 + R02, R12 - R21], dim=-1),
                                 torch.stack([R10 + R01, R11 - R00 - R22, R21 + R12, R20 - R02], dim=-1),
                                 torch.stack([R20 + R02, R21 + R12, R22 - R00 - R11, R01 - R10], dim=-1),
                                 torch.stack([R12 - R21, R20 - R02, R01 - R10, R00 + R11 + R22], dim=-1)], dim=-2) / 3.0
                K = K[i]
                eigval, eigvec = torch.linalg.eigh(K)
                V = eigvec[:, eigval.argmax()]
                q[i] = torch.stack([V[3], V[0], V[1], V[2]])
        return q

    def invert(self, q):
        qa, qb, qc, qd = q.unbind(dim=-1)
        norm = q.norm(dim=-1, keepdim=True)
        q_inv = torch.stack([qa, -qb, -qc, -qd], dim=-1) / norm ** 2
        return q_inv

    def product(self, q1, q2):  # [B,4]
        q1a, q1b, q1c, q1d = q1.unbind(dim=-1)
        q2a, q2b, q2c, q2d = q2.unbind(dim=-1)
        hamil_prod = torch.stack([q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d,
                                  q1a * q2b + q1b * q2a + q1c * q2d - q1d * q2c,
                                  q1a * q2c - q1b * q2d + q1c * q2a + q1d * q2b,
                                  q1a * q2d + q1b * q2c - q1c * q2b + q1d * q2a], dim=-1)
        return hamil_prod


class Continuous6D():
    """
    Continuous 6D representation for pose
    # Code from Pytorch3D
    # https://pytorch3d.readthedocs.io/
    """

    def rotation_6d_to_matrix(self, d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6)

        Returns:
            batch of rotation matrices of size (*, 3, 3)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def matrix_to_rotation_6d(self, matrix: torch.Tensor) -> torch.Tensor:
        return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

    def pose_9d_to_matrix(self, d9):
        rot_6d = d9[:, :6]  # B x 6
        t_3d = d9[:, 6:].unsqueeze(-1)  # B x 3 x 1
        R = self.rotation_6d_to_matrix(rot_6d)  # B x 3 x 3
        Rt = torch.cat([R, t_3d], dim=-1)  # B x 3 x 4
        return Rt


pose = Pose()
lie = Lie()
quaternion = Quaternion()
continuous6d = Continuous6D()


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom


# basic operations of transforming 3D points between world/camera/image coordinates
def world2cam(X, pose):  # [B,N,3]
    X_hom = to_hom(X)
    return X_hom @ pose.transpose(-1, -2)


def cam2img(X, cam_intr):
    return X @ cam_intr.transpose(-1, -2)


def img2cam(X, cam_intr):
    return X @ cam_intr.inverse().transpose(-1, -2)


def cam2world(X, pose):
    # Input pose is to transform the points from world to camera frame;
    # So we need to inverse at first to ensure the pose is to transform
    # the from camera frame to world frame
    # This pose is essentially annotated pose of the objects
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom @ pose_inv.transpose(-1, -2)


def angle_to_rotation_matrix(a, axis):
    # get the rotation matrix from Euler angle around specific axis
    roll = dict(X=1, Y=2, Z=0)[axis]
    O = torch.zeros_like(a)
    I = torch.ones_like(a)
    M = torch.stack([torch.stack([a.cos(), -a.sin(), O], dim=-1),
                     torch.stack([a.sin(), a.cos(), O], dim=-1),
                     torch.stack([O, O, I], dim=-1)], dim=-2)
    M = M.roll((roll, roll), dims=(-2, -1))
    return M


def get_center_and_ray(opt, pose, intr=None, H=None, W=None):  # [HW,2]
    # given the intrinsic/extrinsic matrices, get the camera center and ray directions]
    assert (opt.camera.model == "perspective")
    with torch.no_grad():
        # compute image coordinate grid
        if H is None and W is None:
            _H, _W = opt.H, opt.W
        else:
            _H, _W = H, W
        y_range = torch.arange(_H, dtype=torch.float32, device=pose.device).add_(0.5)
        x_range = torch.arange(_W, dtype=torch.float32, device=pose.device).add_(0.5)
        Y, X = torch.meshgrid(y_range, x_range)  # [H,W]
        xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
    # compute center and ray
    batch_size = len(pose)
    xy_grid = xy_grid.repeat(batch_size, 1, 1)  # [B,HW,2]
    grid_3D = img2cam(to_hom(xy_grid), intr)  # [B,HW,3]
    center_3D = torch.zeros_like(grid_3D)  # [B,HW,3]
    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D, pose)  # [B,HW,3]
    center_3D = cam2world(center_3D, pose)  # [B,HW,3]
    ray = grid_3D - center_3D  # [B,HW,3]
    return center_3D, ray


def get_3D_points_from_depth(opt, center, ray, depth, multi_samples=False):
    if multi_samples:
        center, ray = center[:, :, None], ray[:, :, None]  # [B, HW, 1, 3]
    # x = c+dv
    points_3D = center + ray * depth  # [B,HW,3]/[B,HW,N,3]/[N,3]
    return points_3D


def convert_NDC(opt, center, ray, intr, near=1):
    # shift camera center (ray origins) to near plane (z=1)
    # (unlike conventional NDC, we assume the cameras are facing towards the +z direction)
    center = center + (near - center[..., 2:]) / ray[..., 2:] * ray
    # projection
    cx, cy, cz = center.unbind(dim=-1)  # [B,HW]
    rx, ry, rz = ray.unbind(dim=-1)  # [B,HW]
    scale_x = intr[:, 0, 0] / intr[:, 0, 2]  # [B]
    scale_y = intr[:, 1, 1] / intr[:, 1, 2]  # [B]
    cnx = scale_x[:, None] * (cx / cz)
    cny = scale_y[:, None] * (cy / cz)
    cnz = 1 - 2 * near / cz
    rnx = scale_x[:, None] * (rx / rz - cx / cz)
    rny = scale_y[:, None] * (ry / rz - cy / cz)
    rnz = 2 * near / cz
    center_ndc = torch.stack([cnx, cny, cnz], dim=-1)  # [B,HW,3]
    ray_ndc = torch.stack([rnx, rny, rnz], dim=-1)  # [B,HW,3]
    return center_ndc, ray_ndc


def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1 @ R2.transpose(-2, -1)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()  # numerical stability near -1/+1
    return angle


def procrustes_analysis(X0, X1):  # [N,3]
    # translation
    t0 = X0.mean(dim=0, keepdim=True)
    t1 = X1.mean(dim=0, keepdim=True)
    X0c = X0 - t0
    X1c = X1 - t1
    # scale
    s0 = (X0c ** 2).sum(dim=-1).mean().sqrt()
    s1 = (X1c ** 2).sum(dim=-1).mean().sqrt()
    X0cs = X0c / s0
    X1cs = X1c / s1
    # rotation (use double for SVD, float loses precision)
    U, S, V = (X0cs.t() @ X1cs).double().svd(some=True)
    R = (U @ V.t()).float()
    if R.det() < 0: R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0], t1=t1[0], s0=s0, s1=s1, R=R)
    return sim3


def get_novel_view_poses(opt, pose_anchor, N=60, scale=1, motion='wild'):
    # create circular viewpoints (small oscillations)
    theta = torch.arange(N) / N * 2 * np.pi
    if motion == 'wild':
        R_x = angle_to_rotation_matrix((theta.sin() * 0.3).asin(), "X")
        R_y = angle_to_rotation_matrix((theta.cos() * 0.3).asin(), "Y")
        pose_shift = pose(t=[0, 0, 3 * scale])
        pose_shift2 = pose(t=[0, 0, -1 * scale])
    elif motion == 'gentle':
        R_x = angle_to_rotation_matrix((theta.sin() * 0.05).asin(), "X")
        R_y = angle_to_rotation_matrix((theta.cos() * 0.05).asin(), "Y")
        pose_shift = pose(t=[0, 0, -4 * scale])
        pose_shift2 = pose(t=[0, 0, 4 * scale])
    else:
        raise NotImplementedError
    pose_rot = pose(R=R_y @ R_x)
    pose_oscil = pose.compose([pose_shift, pose_rot, pose_shift2])
    pose_novel = pose.compose([pose_oscil, pose_anchor.cpu()[None]])
    return pose_novel


def get_novel_view_poses_obj(opt, pose_anchor, N=10):
    theta = torch.arange(-N / 2, N / 2) / N * 0.5 * np.pi

    R_z = angle_to_rotation_matrix(theta, "Z")

    pose_rot = pose(R=R_z)
    pose_novel = pose.compose([pose_rot, pose_anchor.cpu()])

    return pose_novel


def compose_pose_residual(pose_refine, pose_source):
    rot = pose_source[:, :3, :3]
    pose_rot = pose(R=rot)
    pose_rot_T = pose(R=torch.transpose(rot, 1, 2))
    # pose_out = pose.compose([pose_source, pose_rot_T, pose_refine, pose_rot])
    pose_out = pose.compose([pose_rot, pose_refine, pose_rot_T, pose_source])

    return pose_out


def aabb_ray_intersection(aabb_min, aabb_max, ray_o, ray_d):
    B, HW, _ = ray_o.shape

    inv_d = torch.reciprocal(ray_d)
    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    t0 = torch.minimum(t_min, t_max)  # B, HW, 3
    t1 = torch.maximum(t_min, t_max)  # B, HW, 3

    t_near, _ = torch.max(t0, dim=2)  # B, HW
    t_far, _ = torch.min(t1, dim=2)  # B, HW

    valid = (t_far > 0) * (t_far > t_near)  # B, HW

    t_near = t_near.view(B, HW)
    t_far = t_far.view(B, HW)

    return t_near, t_far, valid


def enlarge_diagonal(aabb_min, aabb_max, alpha=0.25):
    direction = aabb_max - aabb_min
    aabb_max_n = aabb_max + direction * alpha / 2
    aabb_min_n = aabb_min - direction * alpha / 2
    return aabb_min_n, aabb_max_n


def back_project(pix_coord, depth, cam_intr):
    batch_size, HW, _ = depth.shape

    points = (pix_coord * depth.float()) @ cam_intr.inverse().transpose(-1, -2)  # B x HW x 3
    return points


def generate_pix_coord(batch_size, H=240, W=320, homo=False):
    y_range = torch.arange(H, dtype=torch.float32).add_(0.5)
    x_range = torch.arange(W, dtype=torch.float32).add_(0.5)
    Y, X = torch.meshgrid(y_range, x_range)  # [H,W]
    xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2).cuda()  # [HW,2]
    xy_grid = xy_grid.repeat(batch_size, 1, 1)  # [B,HW,2]
    if homo:
        xy_grid = to_hom(xy_grid)  # B x HW x 3
    return xy_grid


from typing import Union
import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.loss.chamfer import _validate_chamfer_reduction_inputs, _handle_pointcloud_input


def p2p_distance(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
):
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x
    cham_normals = cham_norm_x if return_normals else None

    return cham_dist, cham_normals
