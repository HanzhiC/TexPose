import torch
import torch.nn as nn
import functools
import inspect
import math
from typing import Union, Tuple, List, Dict, NamedTuple
import torch
import numpy as np

from pytorch3d.renderer import PerspectiveCameras

from pytorch3d.renderer import RasterizationSettings, MeshRenderer, MeshRasterizer, \
    SoftSilhouetteShader, SoftPhongShader, AmbientLights
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments

from pytorch3d.renderer.blending import BlendParams
from kornia.geometry.conversions import angle_axis_to_rotation_matrix
import numpy as np
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures import Meshes
from pytorch3d.io import IO, ply_io, load_ply
from pytorch3d.renderer import HardPhongShader, SoftPhongShader
from pytorch3d.renderer.blending import softmax_rgb_blend

import torch
from typing import NamedTuple, Sequence, Union
from pytorch3d.renderer.blending import BlendParams
import torch.nn.functional as F

class MVRenderer(nn.Module):
    def __init__(self, cad_mesh, height, width, batch_size, cam_K=None, mode='complex'):
        super(MVRenderer, self).__init__()
        self.cad_mesh = cad_mesh.extend(batch_size)
        self.device = self.cad_mesh.device
        self.height = height
        self.width = width
        self.batch_size = batch_size
        if cam_K is None:
            cam_K = torch.eye(3, dtype=torch.float32)
        print(cam_K.shape)
        f_x, f_y = cam_K[0, 0], cam_K[1, 1]
        p_x, p_y = cam_K[0, 2], cam_K[1, 2]
        self.f = torch.stack((f_x, f_y), dim=0).unsqueeze(0).type(torch.float32).repeat(batch_size, 1)
        self.c = torch.stack((p_x, p_y), dim=0).unsqueeze(0).type(torch.float32).repeat(batch_size, 1)
        self.T_calib = Pose.from_aa(torch.from_numpy(np.array([0., 0., np.pi], dtype=np.float32))[None],
                                    torch.zeros(1, 3).type(torch.float32))

        self.cameras = PerspectiveCameras(
            focal_length=self.f,
            principal_point=self.c,
            image_size=((self.height, self.width),),
            device=self.device,
            in_ndc=False)

        if mode == 'simplified':
            self.raster_settings = RasterizationSettings(
                image_size=(self.height, self.width),
                faces_per_pixel=1,
            )
        elif mode == 'complex':
            self.raster_settings = RasterizationSettings(
                image_size=(self.height, self.width),
                faces_per_pixel=1,
                max_faces_per_bin=40000
            )
        else:
            raise NotImplemented

        self.mask_raster_settings = RasterizationSettings(
            image_size=(self.height, self.width),
            blur_radius=0.0001,
            faces_per_pixel=70,
            # bin_size=0
        )

        self.nocs_renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings),

            shader=SoftPhongNOCSShader(
                device=self.device,
                cameras=self.cameras,
                blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
                # blend_params=BlendParams(background_color=(0.0, 0.0, 0.0))

            )
        )

        self.mask_renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.mask_raster_settings),
            shader=SoftSilhouetteShader(
                blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
            )
        )

        self.color_renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=AmbientLights(device=self.device),
                blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
            )
        )

        self.feature_renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings),
            shader=SoftPhongFeatureShader(
                device=self.device,
                cameras=self.cameras,
                blend_params=BlendParams(sigma=1e-4, gamma=1e-4)
            )
        )

        self.normal_renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings),
            shader=SoftPhongNormalShader(
                device=self.device,
                cameras=self.cameras,
                blend_params=BlendParams(sigma=1e-4, gamma=1e-4)
            )
        )

        self.renderers = dict(nocs=self.nocs_renderer,
                              color=self.color_renderer,
                              mask=self.mask_renderer,
                              normal=self.normal_renderer,
                              feature=self.feature_renderer)

    @staticmethod
    def calibrate_pose(pose, T_calib):
        T_calib = T_calib.to(pose.device)
        T = T_calib @ pose
        R = torch.transpose(T.R, -1, -2)
        R_6d = matrix_to_rotation_6d(R)
        R = rotation_6d_to_matrix(R_6d).type(torch.float32)
        t = T.t.type(torch.float32)
        return R, t

    def forward(self, pose, K=None, mode='feature', return_depth=True):
        R, t = self.calibrate_pose(pose, self.T_calib)
        R = R.to(self.device)
        t = t.to(self.device)

        if K is not None:
            f_x, f_y = K[..., 0, 0], K[..., 1, 1]
            c_x, c_y = K[..., 0, 2], K[..., 1, 2]
            f = torch.stack((f_x, f_y), dim=-1)
            c = torch.stack((c_x, c_y), dim=-1)
        else:
            f, c = self.f, self.c
        # Rendered results
        rendered_results, fragments = self.renderers[mode](self.cad_mesh,
                                                           R=R,
                                                           T=t,
                                                           focal_length=f,
                                                           principal_point=c,
                                                           device=self.device,
                                                           znear=0.0,
                                                           zfar=10000.0
                                                           )
        rendered_results = rendered_results[..., :-1].permute(0, 3, 1, 2).contiguous()
        if return_depth:
            depth = fragments.zbuf[..., 0].contiguous()
            return rendered_results, depth
        return rendered_results

    # Deprecated

    # def render_nocs(self, pose, K=None, return_depth=False):
    #     # Make sure the coordinate system is aligned between ours and pytorch3d
    #     R, t = self.calibrate_pose(pose, self.T_calib)
    #     R = R.to(self.device)
    #     t = t.to(self.device)
    #     if K is not None:
    #         f_x, f_y = K[..., 0, 0], K[..., 1, 1]
    #         c_x, c_y = K[..., 0, 2], K[..., 1, 2]
    #         f = torch.stack((f_x, f_y), dim=-1)
    #         c = torch.stack((c_x, c_y), dim=-1)
    #     else:
    #         f, c = self.f, self.c
    #     # Rendered results
    #     rendered_results, fragments = self.nocs_renderer(self.cad_mesh,
    #                                                      R=R,
    #                                                      T=t,
    #                                                      focal_length=f,
    #                                                      principal_point=c,
    #                                                      device=self.device,
    #                                                      znear=0.0,
    #                                                      zfar=10000.0
    #                                                      )
    #     rendered_results = rendered_results[..., :-1].permute(0, 3, 1, 2).contiguous()
    #     if return_depth:
    #         depth = fragments.zbuf[..., 0].contiguous()
    #         return rendered_results, depth
    #     return rendered_results
    #
    # def render_mask(self, pose, K):
    #     # Make sure the coordinate system is aligned between ours and pytorch3d
    #     R, t = self.calibrate_pose(pose, self.T_calib)
    #     R = R.to(self.device)
    #     t = t.to(self.device)
    #     if K is not None:
    #         f_x, f_y = K[..., 0, 0], K[..., 1, 1]
    #         c_x, c_y = K[..., 0, 2], K[..., 1, 2]
    #         f = torch.stack((f_x, f_y), dim=-1)
    #         c = torch.stack((c_x, c_y), dim=-1)
    #     else:
    #         f, c = self.f, self.c
    #     # Rendered results
    #     rendered_results = self.mask_renderer(self.cad_mesh,
    #                                           R=R,
    #                                           T=t,
    #                                           focal_length=f,
    #                                           principal_point=c,
    #                                           device=self.device,
    #                                           znear=0.0,
    #                                           zfar=10000.0
    #                                           )[..., 3]
    #
    #     return rendered_results
    #
    # def render_color(self, pose, K):
    #     # Make sure the coordinate system is aligned between ours and pytorch3d
    #     R, t = self.calibrate_pose(pose, self.T_calib)
    #     R = R.to(self.device)
    #     t = t.to(self.device)
    #     if K is not None:
    #         f_x, f_y = K[..., 0, 0], K[..., 1, 1]
    #         c_x, c_y = K[..., 0, 2], K[..., 1, 2]
    #         f = torch.stack((f_x, f_y), dim=-1)
    #         c = torch.stack((c_x, c_y), dim=-1)
    #     else:
    #         f, c = self.f, self.c
    #     # Rendered results
    #     rendered_results = self.color_renderer(self.cad_mesh,
    #                                            R=R,
    #                                            T=t,
    #                                            focal_length=f,
    #                                            principal_point=c,
    #                                            device=self.device,
    #                                            znear=0.0,
    #                                            zfar=10000.0
    #                                            )
    #     rendered_results = rendered_results[..., :-1].permute(0, 3, 1, 2).contiguous()
    #     return rendered_results
    #
    # def render_feature(self, pose, K=None, return_depth=False):
    #     # Make sure the coordinate system is aligned between ours and pytorch3d
    #     R, t = self.calibrate_pose(pose, self.T_calib)
    #     R = R.to(self.device)
    #     t = t.to(self.device)
    #     if K is not None:
    #         f_x, f_y = K[..., 0, 0], K[..., 1, 1]
    #         c_x, c_y = K[..., 0, 2], K[..., 1, 2]
    #         f = torch.stack((f_x, f_y), dim=-1)
    #         c = torch.stack((c_x, c_y), dim=-1)
    #     else:
    #         f, c = self.f, self.c
    #
    #     # Rendered results
    #     rendered_results, fragments = self.feature_renderer(self.cad_mesh,
    #                                                         R=R,
    #                                                         T=t,
    #                                                         focal_length=f,
    #                                                         principal_point=c,
    #                                                         device=self.device,
    #                                                         znear=0.0,
    #                                                         zfar=10000.0
    #                                                         )
    #     rendered_results = rendered_results[..., :-1].permute(0, 3, 1, 2).contiguous()
    #     if return_depth:
    #         depth = fragments.zbuf[..., 0].contiguous()
    #         return rendered_results, depth
    #     return rendered_results
    #
    # def crop_rendered_results(self, images, target_centers, target_scales, target_res=128):
    #     images = images.unsqueeze(1) if (len(images.shape) == 3) else images
    #     device = images.device
    #     B, C, _, _ = images.shape
    #     results = []
    #     for bi in range(B):
    #         image = images[bi]
    #         scale = target_scales[bi]
    #         center = (target_centers[bi][0].item(), target_centers[bi][1].item())
    #         upper = max(0, int(center[0] - scale / 2. + 0.5))
    #         left = max(0, int(center[1] - scale / 2. + 0.5))
    #         bottom = min(self.height, int(center[0] - scale / 2. + 0.5) + int(scale))
    #         right = min(self.width, int(center[1] - scale / 2. + 0.5) + int(scale))
    #         crop_ht = float(bottom - upper)
    #         crop_wd = float(right - left)
    #
    #         # Determine resizing parameters
    #         if crop_ht > crop_wd:
    #             resize_ht = target_res
    #             resize_wd = int(target_res / crop_ht * crop_wd + 0.5)
    #         elif crop_ht < crop_wd:
    #             resize_wd = target_res
    #             resize_ht = int(target_res / crop_wd * crop_ht + 0.5)
    #         else:
    #             resize_wd = resize_ht = target_res
    #
    #         # Begin do resizing
    #         extracted = image[:, upper:bottom, left:right].unsqueeze(0)
    #         # print(extracted.shape)
    #         image_resized = F.interpolate(extracted, size=(resize_ht, resize_wd), mode='bilinear',
    #                                       align_corners=False).squeeze(0)
    #
    #         image_out = torch.zeros((C, target_res, target_res)).float()
    #         image_out[:, int(target_res / 2.0 - resize_ht / 2.0 + 0.5):(int(target_res / 2.0 -
    #                                                                         resize_ht / 2.0 + 0.5) + resize_ht),
    #         int(target_res / 2.0 - resize_wd / 2.0 + 0.5):(
    #                 int(target_res / 2.0 - resize_wd / 2.0 + 0.5) + resize_wd)] = image_resized
    #         results.append(image_out)
    #
    #     results = torch.stack(results, dim=0).to(device)
    #     results = results.squeeze(1) if C == 1 else results
    #     # results = results.to(device)
    #     return results

def so3exp_map(w, eps: float = 1e-7):
    """Compute rotation matrices from batched twists.
    Args:
        w: batched 3D axis-angle vectors of size (..., 3).
    Returns:
        A batch of rotation matrices of size (..., 3, 3).
    """
    theta = w.norm(p=2, dim=-1, keepdim=True)
    small = theta < eps
    div = torch.where(small, torch.ones_like(theta), theta)
    W = skew_symmetric(w / div)
    theta = theta[..., None]  # ... x 1 x 1
    res = W * torch.sin(theta) + (W @ W) * (1 - torch.cos(theta))
    res = torch.where(small[..., None], W, res)  # first-order Taylor approx
    return torch.eye(3).to(W) + res
def skew_symmetric(v):
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3).
    """
    z = torch.zeros_like(v[..., 0])
    M = torch.stack([
        z, -v[..., 2], v[..., 1],
        v[..., 2], z, -v[..., 0],
        -v[..., 1], v[..., 0], z,
    ], dim=-1).reshape(v.shape[:-1] + (3, 3))
    return M
def autocast(func):
    """Cast the inputs of a TensorWrapper method to PyTorch tensors
       if they are numpy arrays. Use the device and dtype of the wrapper.
    """
    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device('cpu')
        dtype = None
        if isinstance(self, TensorWrapper):
            if self._data is not None:
                device = self.device
                dtype = self.dtype
        elif not inspect.isclass(self) or not issubclass(self, TensorWrapper):
            raise ValueError(self)

        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)
        return func(self, *cast_args)

    return wrap


class TensorWrapper:
    _data = None

    @autocast
    def __init__(self, data: torch.Tensor):
        self._data = data

    @property
    def shape(self):
        return self._data.shape[:-1]

    @property
    def device(self):
        return self._data.device

    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, index):
        return self.__class__(self._data[index])

    def __setitem__(self, index, item):
        self._data[index] = item.data

    def to(self, *args, **kwargs):
        return self.__class__(self._data.to(*args, **kwargs))

    def cpu(self):
        return self.__class__(self._data.cpu())

    def cuda(self):
        return self.__class__(self._data.cuda())

    def pin_memory(self):
        return self.__class__(self._data.pin_memory())

    def float(self):
        return self.__class__(self._data.float())

    def double(self):
        return self.__class__(self._data.double())

    def detach(self):
        return self.__class__(self._data.detach())

    @classmethod
    def stack(cls, objects: List, dim=0, *, out=None):
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.stack:
            return self.stack(*args, **kwargs)
        else:
            return NotImplemented


class Pose(TensorWrapper):
    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] == 12
        super().__init__(data)

    @classmethod
    @autocast
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor):
        '''Pose from a rotation matrix and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            t: translation vector with shape (..., 3).
        '''
        assert R.shape[-2:] == (3, 3)
        assert t.shape[-1] == 3
        assert R.shape[:-2] == t.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), t], -1)
        return cls(data)

    @classmethod
    @autocast
    def from_aa(cls, aa: torch.Tensor, t: torch.Tensor):
        '''Pose from an axis-angle rotation vector and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            aa: axis-angle rotation vector with shape (..., 3).
            t: translation vector with shape (..., 3).
        '''
        assert aa.shape[-1] == 3
        assert t.shape[-1] == 3
        assert aa.shape[:-1] == t.shape[:-1]
        return cls.from_Rt(so3exp_map(aa), t)

    @classmethod
    def from_4x4mat(cls, T: torch.Tensor):
        '''Pose from an SE(3) transformation matrix.
        Args:
            T: transformation matrix with shape (..., 4, 4).
        '''
        assert T.shape[-2:] == (4, 4)
        R, t = T[..., :3, :3], T[..., :3, 3]
        return cls.from_Rt(R, t)

    @classmethod
    def from_colmap(cls, image: NamedTuple):
        '''Pose from a COLMAP Image.'''
        return cls.from_Rt(image.qvec2rotmat(), image.tvec)

    @property
    def R(self) -> torch.Tensor:
        '''Underlying rotation matrix with shape (..., 3, 3).'''
        rvec = self._data[..., :9]
        return rvec.reshape(rvec.shape[:-1]+(3, 3))

    @property
    def t(self) -> torch.Tensor:
        '''Underlying translation vector with shape (..., 3).'''
        return self._data[..., -3:]

    def inv(self) -> 'Pose':
        '''Invert an SE(3) pose.'''
        R = self.R.transpose(-1, -2)
        t = -(R @ self.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    def compose(self, other: 'Pose') -> 'Pose':
        '''Chain two SE(3) poses: T_B2C.compose(T_A2B) -> T_A2C.'''
        R = self.R @ other.R
        t = self.t + (self.R @ other.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    @autocast
    def transform(self, p3d: torch.Tensor) -> torch.Tensor:
        '''Transform a set of 3D points.
        Args:
            p3d: 3D points, numpy array or PyTorch tensor with shape (..., 3).
        '''
        assert p3d.shape[-1] == 3
        # assert p3d.shape[:-2] == self.shape  # allow broadcasting
        return p3d @ self.R.transpose(-1, -2) + self.t.unsqueeze(-2)

    def __mul__(self, p3D: torch.Tensor) -> torch.Tensor:
        '''Transform a set of 3D points: T_A2B * p3D_A -> p3D_B.'''
        return self.transform(p3D)

    def __matmul__(self, other: 'Pose') -> 'Pose':
        '''Chain two SE(3) poses: T_B2C @ T_A2B -> T_A2C.'''
        return self.compose(other)

    def numpy(self) -> Tuple[np.ndarray]:
        return self.R.numpy(), self.t.numpy()

    def magnitude(self) -> Tuple[torch.Tensor]:
        '''Magnitude of the SE(3) transformation.
        Returns:
            dr: rotation anngle in degrees.
            dt: translation distance in meters.
        '''
        trace = torch.diagonal(self.R, dim1=-1, dim2=-2).sum(-1)
        cos = torch.clamp((trace - 1) / 2, -1, 1)
        dr = torch.acos(cos).abs() / math.pi * 180
        dt = torch.norm(self.t, dim=-1)
        return dr, dt

    def __repr__(self):
        return f'Pose: {self.shape} {self.dtype} {self.device}'

# from .rasterizer import Fragments


def softmax_feature_blend(
        features: torch.Tensor,
        fragments,
        blend_params: BlendParams,
        znear: Union[float, torch.Tensor] = 1.0,
        zfar: Union[float, torch.Tensor] = 100,
) -> torch.Tensor:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
        features: (N, H, W, K, C) RGB color for each of the top K faces per pixel.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.
            - background_color: (3) element list/tuple/torch.Tensor specifying
              the RGB values for the background color.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction

    Returns:
        RGBA pixel_colors: (N, H, W, 4)

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """
    C = features.shape[-1]
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_features = torch.ones((N, H, W, C + 1), dtype=features.dtype, device=features.device)
    # background_ = blend_params.background_color
    # if not isinstance(background_, torch.Tensor):
    #     background = torch.tensor(background_, dtype=torch.float32, device=device)
    # else:
    #     background = background_.to(device)

    background = torch.zeros(C, dtype=torch.float32, device=device)
    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.

    # Reshape to be compatible with (N, H, W, K) values in fragments
    if torch.is_tensor(zfar):
        # pyre-fixme[16]
        zfar = zfar[:, None, None, None]
    if torch.is_tensor(znear):
        # pyre-fixme[16]: Item `float` of `Union[float, Tensor]` has no attribute
        #  `__getitem__`.
        znear = znear[:, None, None, None]

    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

    # Also apply exp normalize trick for the background color weight.
    # Clamp to ensure delta is never 0.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta

    # Sum: weights * textures + background color
    weighted_features = (weights_num[..., None] * features).sum(dim=-2)
    weighted_background = delta * background
    pixel_features[..., :-1] = (weighted_features + weighted_background) / denom
    pixel_features[..., -1] = 1.0 - alpha

    return pixel_features


class SoftPhongNOCSShader(SoftPhongShader):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def phong_NOCS_shading(self, meshes, fragments, cameras) -> torch.Tensor:
        """
        Apply per pixel shading. First interpolate the vertex normals and
        vertex coordinates using the barycentric coordinates to get the position
        and normal at each pixel. Then compute the illumination for each pixel.
        The pixel color is obtained by multiplying the pixel textures by the ambient
        and diffuse illumination and adding the specular component.

        Args:
            meshes: Batch of meshes
            fragments: Fragments named tuple with the outputs of rasterization
            lights: Lights class containing a batch of lights
            cameras: Cameras class containing a batch of cameras
            materials: Materials class containing a batch of material properties
            texels: texture per pixel of shape (N, H, W, K, 3)

        Returns:
            colors: (N, H, W, K, 3)
        """

        verts = meshes.verts_packed()  # (V, 3)
        faces = meshes.faces_packed()  # (F, 3)

        x = verts[:, 0]
        y = verts[:, 1]
        z = verts[:, 2]

        x_ct = verts[:, 0].mean()
        y_ct = verts[:, 1].mean()
        z_ct = verts[:, 2].mean()

        x_abs = torch.max(torch.abs(x - x_ct))
        y_abs = torch.max(torch.abs(y - y_ct))
        z_abs = torch.max(torch.abs(z - z_ct))

        x_norm = (x - x_ct) / x_abs
        y_norm = (y - y_ct) / y_abs
        z_norm = (z - z_ct) / z_abs

        x_norm = (x_norm + 1) / 2
        y_norm = (y_norm + 1) / 2
        z_norm = (z_norm + 1) / 2

        verts_norm = torch.zeros_like(verts)
        verts_norm[:, 0] = x_norm
        verts_norm[:, 1] = y_norm
        verts_norm[:, 2] = z_norm

        faces_verts = verts_norm[faces]  # (F, 3, 3) For each face, give the 3 vertices associated

        pixel_coords = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts
        )

        colors = pixel_coords
        return colors

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardPhongShader"
            raise ValueError(msg)
        blend_params = kwargs.get("blend_params", self.blend_params)

        colors = self.phong_NOCS_shading(
            meshes=meshes,
            fragments=fragments,
            cameras=cameras,
        )

        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        NOCS = softmax_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar
        )
        return NOCS

class SoftPhongNormalShader(SoftPhongShader):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def phong_normal_shading(self, meshes, fragments, cameras) -> torch.Tensor:
        """
        Apply per pixel shading. First interpolate the vertex normals and
        vertex coordinates using the barycentric coordinates to get the position
        and normal at each pixel. Then compute the illumination for each pixel.
        The pixel color is obtained by multiplying the pixel textures by the ambient
        and diffuse illumination and adding the specular component.

        Args:
            meshes: Batch of meshes
            fragments: Fragments named tuple with the outputs of rasterization
            lights: Lights class containing a batch of lights
            cameras: Cameras class containing a batch of cameras
            materials: Materials class containing a batch of material properties
            texels: texture per pixel of shape (N, H, W, K, 3)

        Returns:
            colors: (N, H, W, K, 3)
        """

        verts = meshes.verts_packed()  # (V, 3)
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_verts = verts[faces]
        faces_normals = vertex_normals[faces]
        # pixel_coords = interpolate_face_attributes(
        #     fragments.pix_to_face, fragments.bary_coords, faces_verts
        # )
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals  # torch.ones_like()
        )
        pix_to_face = fragments.pix_to_face
        batch_size = pix_to_face.shape[0]

        tangent = self.compute_tangent(verts[faces])
        # Smoothe the tangent map by interpolating per vertex tangent
        tangent_map = self.interpolate_face_average_attributes(
            tangent, fragments, verts, faces, batch_size
        )

        pixel_normals = F.normalize(pixel_normals, dim=-1)
        bitangent_map = torch.cross(pixel_normals, tangent_map, dim=-1)
        bitangent_map = F.normalize(bitangent_map, dim=-1)
        tangent_map = torch.cross(bitangent_map, pixel_normals, dim=-1)
        tangent_map = F.normalize(tangent_map, dim=-1)

        # pixel-wise TBN matrix - flip to get correct direction
        TBN = torch.stack(
            (-tangent_map, -bitangent_map, pixel_normals), dim=4
        )
        nm = self.normal_map.sample_textures(fragments)
        nm = F.normalize(
            torch.matmul(
                TBN.transpose(-1, -2).reshape(-1, 3, 3), nm.reshape(-1, 3, 1)
            ).reshape(pixel_normals.shape),
            dim=-1,
        )
        return nm

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardPhongShader"
            raise ValueError(msg)
        blend_params = kwargs.get("blend_params", self.blend_params)

        normal = self.phong_normal_shading(
            meshes=meshes,
            fragments=fragments,
            cameras=cameras,
        )
        return normal



class SoftPhongNOCSShader_minmax(SoftPhongShader):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def phong_NOCS_shading(self, meshes, fragments, cameras) -> torch.Tensor:
        """
        Apply per pixel shading. First interpolate the vertex normals and
        vertex coordinates using the barycentric coordinates to get the position
        and normal at each pixel. Then compute the illumination for each pixel.
        The pixel color is obtained by multiplying the pixel textures by the ambient
        and diffuse illumination and adding the specular component.

        Args:
            meshes: Batch of meshes
            fragments: Fragments named tuple with the outputs of rasterization
            lights: Lights class containing a batch of lights
            cameras: Cameras class containing a batch of cameras
            materials: Materials class containing a batch of material properties
            texels: texture per pixel of shape (N, H, W, K, 3)

        Returns:
            colors: (N, H, W, K, 3)
        """

        verts = meshes.verts_packed()  # (V, 3)
        faces = meshes.faces_packed()  # (F, 3)

        x = verts[:, 0]
        y = verts[:, 1]
        z = verts[:, 2]

        x_min, x_max = torch.min(x), torch.max(x)
        y_min, y_max = torch.min(y), torch.max(y)
        z_min, z_max = torch.min(z), torch.max(z)

        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)
        z_norm = (z - z_min) / (z_max - z_min)

        verts_norm = torch.zeros_like(verts)
        verts_norm[:, 0] = x_norm
        verts_norm[:, 1] = y_norm
        verts_norm[:, 2] = z_norm

        faces_verts = verts_norm[faces]  # (F, 3, 3) For each face, give the 3 vertices associated

        pixel_coords = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts
        )

        colors = pixel_coords
        return colors

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardPhongShader"
            raise ValueError(msg)
        blend_params = kwargs.get("blend_params", self.blend_params)

        colors = self.phong_NOCS_shading(
            meshes=meshes,
            fragments=fragments,
            cameras=cameras,
        )

        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        NOCS = softmax_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar
        )
        return NOCS

class SoftPhongFeatureShader(SoftPhongShader):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        blend_params = kwargs.get("blend_params", self.blend_params)

        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        featxels = meshes.sample_textures(fragments)
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_feature_blend(
            featxels, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images
