import json
import os

import PIL
import imageio
import numpy as np
import torch
import cv2

import camera
from util import readlines
from . import base
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
from .cad_model import CAD_Model
from util import log, debug

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Dataset(base.Dataset):
    def __init__(self, opt, split="train", subset=None, multi_obj=False):
        self.raw_H, self.raw_W = 480, 640
        super().__init__(opt, split)
        self.data_path = os.path.join(opt.data.root, opt.data.dataset)  # Data path to the bop dataset

        self.split_path = os.path.join("splits", opt.data.dataset, str(opt.data.object),
                                       opt.data.scene, "{}.txt".format(split))
        self.list = readlines(self.split_path)
        self.multi_obj = multi_obj

        if subset: self.list = self.list[:subset]
        if self.multi_obj: assert not opt.data.preload
        self.initialize_meta(opt)

    def initialize_meta(self, opt):
        line = self.list[0].split(' ')
        model_name, folder = line[0], line[1]
        scene_obj_path = os.path.join(self.data_path, folder, 'scene_object.json')

        # load scene obj
        if self.multi_obj:
            with open(scene_obj_path) as f_obj:
                self.scene_obj_all = json.load(f_obj)
                f_obj.close()
            del f_obj

        # load scene info
        if self.split != 'test' and opt.data.pose_source == 'predicted':
            if opt.data.scene_info_source is None:
                scene_info_path = os.path.join(self.data_path, folder, 'scene_pred_info.json')
            else:
                info_names = dict(gt='scene_gt_info.json', predicted='scene_pred_info.json')
                scene_info_path = os.path.join(self.data_path, folder, info_names[opt.data.scene_info_source])
        else:
            scene_info_path = os.path.join(self.data_path, folder, 'scene_gt_info.json')

        with open(scene_info_path) as f_info:
            self.scene_info_all = json.load(f_info)
            f_info.close()
        # load scene pose
        scene_gt_path = os.path.join(self.data_path, folder, 'scene_gt.json')
        scene_cam_path = os.path.join(self.data_path, folder, 'scene_camera.json')
        scene_pred_path = os.path.join(self.data_path, folder, 'scene_pred_{}.json'.format(opt.data.pose_loop))
        log.info("Use predicted pose from {} ...".format(scene_pred_path))

        with open(scene_gt_path) as f_gt:
            self.scene_gt_all = json.load(f_gt)
            f_gt.close()
        with open(scene_cam_path) as f_cam:
            self.scene_cam_all = json.load(f_cam)
            f_cam.close()
        if self.split == 'train' and opt.data.pose_source == 'predicted':
            with open(scene_pred_path) as f_pred:
                self.scene_pred_all = json.load(f_pred)
                f_pred.close()
            del f_pred
        del f_cam, f_gt, f_info,
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt, self.get_image)
            self.cameras = self.preload_threading(opt, self.get_camera, data_str="cameras")

    def prefetch_all_data(self, opt):
        assert (not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def get_all_camera_poses(self, opt, source='gt'):
        # Read the json file at first for this item
        scene_pose_all = self.scene_gt_all if source == 'gt' else self.scene_pred_all
        pose_raw_all = []
        for idx, sample in enumerate(self.list):
            line = sample.split(' ')
            model_name = line[0]
            frame_index = int(line[2])
            if self.multi_obj:
                obj_scene_id = int(self.scene_obj_all[str(frame_index)][model_name])
            else:
                obj_scene_id = 0
            rot = scene_pose_all[str(frame_index)][obj_scene_id]['cam_R_m2c']
            tra = scene_pose_all[str(frame_index)][obj_scene_id]['cam_t_m2c']
            pose_raw = torch.eye(4)
            pose_raw[:3, :3] = torch.from_numpy(np.array(rot).reshape(3, 3).astype(np.float32))
            pose_raw[:3, 3] = torch.from_numpy(np.array(tra).astype(np.float32)) / 1000
            pose_raw_all.append(pose_raw)
        pose_canon_all = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
        return pose_canon_all

    def __getitem__(self, idx):
        line = self.list[idx].split()
        frame_index = int(line[2])

        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        model_name = line[0]
        if self.multi_obj:
            obj_scene_id = int(self.scene_obj_all[str(frame_index)][model_name])
        else:
            obj_scene_id = 0
        image = self.images[idx] if opt.data.preload else self.get_image(opt, idx, obj_scene_id=obj_scene_id)
        _, intr, pose_gt, pose_init = self.cameras[idx] if opt.data.preload else self.get_camera(opt, idx,
                                                                                                 obj_scene_id=obj_scene_id)
        z_near, z_far = self.get_range(opt, idx, obj_scene_id=obj_scene_id)
        obj_mask = self.get_obj_mask(opt, idx, obj_scene_id=obj_scene_id)

        if self.opt.data.scene != 'scene_all':
            depth_gt = self.get_depth(opt, idx, obj_scene_id=obj_scene_id)
        else:
            depth_gt = np.ones_like(obj_mask)

        if opt.data.bgcolor is not None:
            image = torch.where(obj_mask[None].repeat(3, 1, 1) > 0, image, opt.data.bgcolor * torch.ones_like(image))
        frame_index = torch.tensor(int(self.list[idx].split()[2]))
        sample.update(
            image=image,
            intr=intr,
            pose=pose_gt,
            pose_init=pose_init,
            z_near=z_near,
            z_far=z_far,
            obj_mask=obj_mask,
            depth_gt=depth_gt,
            frame_index=frame_index)
        if opt.data.erode_mask_loss is not None:
            erode_mask = self.get_obj_mask(opt, idx, return_erode=True)
            sample.update(erode_mask=erode_mask)
        if opt.loss_weight.feat or opt.gan:
            if self.split == 'train':
                image_syn, mask_syn = self.get_predicted_synthetic_image(opt, idx, obj_scene_id=obj_scene_id)
                sample.update(image_syn=image_syn, mask_syn=mask_syn)
        if self.split == 'train' and opt.gan is not None:
            nocs_pred = self.get_predicted_nocs(opt, idx, obj_scene_id=obj_scene_id)
            normal_pred = self.get_predicted_normal(opt, idx, obj_scene_id=obj_scene_id)
            sample.update(nocs_pred=nocs_pred, normal_pred=normal_pred)
        return sample

    def get_2d_bbox(self, opt, idx, obj_scene_id):
        # Parse the split text
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        assert opt.H == opt.W
        bbox = self.scene_info_all[str(frame_index)][obj_scene_id]["bbox_obj"]
        if opt.data.box_format is None:
            x_ul, y_ul, h, w = bbox
        elif opt.data.box_format == 'hw':
            x_ul, y_ul, h, w = bbox
        elif opt.data.box_format == 'wh':
            x_ul, y_ul, w, h = bbox
        else:
            raise NotImplementedError

        center = np.array([int(y_ul + h / 2), int(x_ul + w / 2)])
        scale = int(1.5 * max(h, w))  # detected bb size
        resize = opt.H / scale
        return center, scale, resize

    def get_image(self, opt, idx, ext='.png', obj_scene_id=0):
        # Parse the split text
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}{}".format(frame_index, ext)
        image_fname = os.path.join(self.data_path, folder, 'rgb', file)

        center, scale, resize = self.get_2d_bbox(opt, idx, obj_scene_id=obj_scene_id)
        image = cv2.imread(image_fname, -1)[:, :, [2, 1, 0]]
        image = self.Crop_by_Pad(image, center, scale, opt.H, channel=3).astype(np.uint8)
        image = torchvision_F.to_tensor(image)
        return image

    def get_predicted_synthetic_image(self, opt, idx, ext='.png', obj_scene_id=0):
        # Parse the split text
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        if not self.multi_obj:
            file = "{:06d}{}".format(frame_index, ext)
        else:
            file = "{:06d}_{:06d}{}".format(frame_index, obj_scene_id, ext)

        if opt.data.pose_source == 'predicted' and self.split == 'train':
            assert opt.data.pose_loop is not None
            source = 'rgbsyn_{}'.format(opt.data.pose_loop)
        else:
            source = 'rgbsyn_GT'
        rgba_fname = os.path.join(self.data_path, folder, source, file)
        rgba = cv2.imread(rgba_fname, -1)
        image = torchvision_F.to_tensor(rgba[..., :3][..., [2, 1, 0]])
        alpha = torch.from_numpy((rgba[..., 3] > 0).astype(np.float32))
        return image, alpha

    def get_predicted_nocs(self, opt, idx, ext='.png', obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        if not self.multi_obj:
            file = "{:06d}{}".format(frame_index, ext)
        else:
            file = "{:06d}_{:06d}{}".format(frame_index, obj_scene_id, ext)
        if opt.data.pose_source == 'predicted' and self.split == 'train':
            assert opt.data.pose_loop is not None
            source = 'nocs_{}'.format(opt.data.pose_loop)
        else:
            source = 'nocs_GT'
        nocs_fname = os.path.join(self.data_path, folder, source, file)
        nocs = cv2.imread(nocs_fname, -1).astype(np.float32)[..., [2, 1, 0]]
        nocs = self.smooth_geo(nocs / 255)
        nocs = torch.from_numpy(nocs).permute(2, 0, 1)
        return nocs

    def get_predicted_normal(self, opt, idx, ext='.npz', obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        if not self.multi_obj:
            file = "{:06d}{}".format(frame_index, ext)
        else:
            file = "{:06d}_{:06d}{}".format(frame_index, obj_scene_id, ext)
        if opt.data.pose_source == 'predicted' and self.split == 'train':
            assert opt.data.pose_loop is not None
            source = 'normal_{}'.format(opt.data.pose_loop)
        else:
            source = 'normal_GT'
        normal_fname = os.path.join(self.data_path, folder, source, file)
        normal = np.load(normal_fname, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
        normal = self.smooth_geo(normal)
        normal = torch.from_numpy(normal).permute(2, 0, 1)
        return normal

    def get_depth(self, opt, idx, ext='.png', obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}{}".format(frame_index, ext)
        depth_scale = self.scene_cam_all[str(frame_index)]["depth_scale"]

        depth_fname = os.path.join(self.data_path, folder, 'depth', file)
        center, scale, resize = self.get_2d_bbox(opt, idx, obj_scene_id=obj_scene_id)
        depth = cv2.imread(depth_fname, -1) / 1000.
        depth = self.Crop_by_Pad(depth, center, scale, opt.H, channel=1).astype(np.float32)
        depth = torch.from_numpy(depth).float().squeeze(-1)
        mask = self.get_obj_mask(opt, idx, obj_scene_id=obj_scene_id)
        depth *= (opt.nerf.depth.scale * depth_scale)
        depth *= mask.float()
        return depth

    def get_obj_mask(self, opt, idx, ext='.png', return_visib=True, return_erode=False, obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}_{:06d}{}".format(frame_index, obj_scene_id, ext)
        center, scale, resize = self.get_2d_bbox(opt, idx, obj_scene_id=obj_scene_id)

        # Full mask through rasterization-based rendering
        mask_fname_full = os.path.join(self.data_path, folder, 'mask', file)
        mask_full = cv2.imread(mask_fname_full, -1)
        mask_full = self.Crop_by_Pad(mask_full, center, scale, opt.H, 1, cv2.INTER_LINEAR)
        mask_full = mask_full.astype(np.float32)

        # Visible region
        if self.split == 'train':
            visib_source = opt.data.mask_visib_source if 'adapt_st' in opt.model else 'mask_visib'
            mask_fname_visib = os.path.join(self.data_path, folder, visib_source, file)
            mask_visib = cv2.imread(mask_fname_visib, -1)
            if mask_visib.shape[0] != opt.H:
                mask_visib = self.Crop_by_Pad(mask_visib, center, scale, opt.H, 1, cv2.INTER_LINEAR)
            if opt.data.erode_mask:
                # erode the mask during training
                d_kernel = np.ones((3, 3))
                mask_visib = cv2.erode(mask_visib, kernel=d_kernel, iterations=1)
            mask_visib = mask_visib.astype(np.float32)
        else:
            mask_visib = np.ones_like(mask_full)

        if return_visib:
            if self.split == 'train':
                obj_mask = mask_visib > 0
            else:
                obj_mask = mask_full > 0
        else:
            if self.split == 'train':
                obj_mask = mask_visib > 0
            else:
                obj_mask = mask_full > 0
        obj_mask = obj_mask.astype(np.float32)
        if return_erode:
            obj_mask = cv2.erode(obj_mask, kernel=np.ones((3, 3)), iterations=1)
        obj_mask = torch.from_numpy(obj_mask).squeeze(-1)
        return obj_mask

    def get_range(self, opt, idx, obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])

        depth_min_bg, depth_max_bg = opt.nerf.depth.range
        depth_min_bg *= opt.nerf.depth.scale
        depth_max_bg *= opt.nerf.depth.scale
        depth_min_bg = torch.Tensor([depth_min_bg]).float().expand(opt.H * opt.W)
        depth_max_bg = torch.Tensor([depth_max_bg]).float().expand(opt.H * opt.W)
        mask = self.get_obj_mask(opt, idx, return_visib=False, obj_scene_id=obj_scene_id).float()
        if opt.nerf.depth.range_source == 'box':
            if opt.data.pose_source == 'predicted' and self.split in ['train', 'val']:
                box_source = opt.nerf.depth.box_source
            else:
                box_source = 'gt_box'
            # Get meta-information, pose and intrinsics
            if self.multi_obj:
                file = "{:06d}_{:06d}{}".format(frame_index, obj_scene_id, '.npz')
            else:
                file = "{:06d}{}".format(frame_index, '.npz')
            box_fname = os.path.join(self.data_path, folder, box_source, file)
            box_range = np.load(box_fname, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
            box_range = box_range.astype(np.float32).transpose((1, 2, 0))
            center, scale, resize = self.get_2d_bbox(opt, idx, obj_scene_id=obj_scene_id)
            box_range = self.Crop_by_Pad(box_range, center, scale, opt.H, channel=2).astype(np.float32)
            box_range = torch.from_numpy(box_range)
            if opt.nerf.depth.box_mask:
                box_range *= mask[..., None]
            box_range = box_range.permute(2, 0, 1).view(2, opt.H * opt.W)
            box_range = (box_range / 1000) * opt.nerf.depth.scale

            z_near, z_far = box_range[0], box_range[1]
            z_near = torch.where(z_near > 0, z_near, depth_min_bg)
            z_far = torch.where(z_far > 0, z_far, depth_max_bg)

        elif opt.nerf.depth.range_source == 'render':
            depth_gt = self.get_depth(opt, idx).view(opt.H * opt.W)
            z_near, z_far = depth_gt * 0.8, depth_gt * 1.2
            z_near = torch.where(z_near > 0, z_near, depth_min_bg)
            z_far = torch.where(z_far > 0, z_far, depth_max_bg)

        elif opt.nerf.depth.range_source is None:
            z_near = depth_min_bg
            z_far = depth_max_bg
        else:
            raise NotImplementedError

        assert z_near.shape[0] == opt.H * opt.W and z_far.shape[0] == opt.H * opt.W
        return z_near, z_far

    def get_camera(self, opt, idx, obj_scene_id=0):
        # Parse the split text
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])

        center, scale, resize = self.get_2d_bbox(opt, idx, obj_scene_id=obj_scene_id)
        center_offset = self.get_center_offset(center, scale, self.raw_H, self.raw_W)
        cam_K = self.scene_cam_all[str(frame_index)]["cam_K"]
        cam_K = torch.from_numpy(np.array(cam_K, dtype=np.float32).reshape(3, 3))
        cam_intr = cam_K.clone()
        intr = self.preprocess_intrinsics(cam_intr, resize, center + center_offset, res=opt.H)

        # Load GT pose if required
        rot = self.scene_gt_all[str(frame_index)][obj_scene_id]['cam_R_m2c']
        tra = self.scene_gt_all[str(frame_index)][obj_scene_id]['cam_t_m2c']
        pose_gt_raw = torch.eye(4)
        pose_gt_raw[:3, :3] = torch.from_numpy(np.array(rot).reshape(3, 3).astype(np.float32))
        pose_gt_raw[:3, 3] = torch.from_numpy(np.array(tra).astype(np.float32)) / 1000  # scale in meter
        pose_gt = self.parse_raw_camera(opt, pose_gt_raw)

        # Load predicted pose
        if self.split == 'train' and opt.data.pose_source == 'predicted':
            # TODO: save predicted pose before training
            rot = self.scene_pred_all[str(frame_index)][obj_scene_id]['cam_R_m2c']
            tra = self.scene_pred_all[str(frame_index)][obj_scene_id]['cam_t_m2c']
            pose_init_raw = torch.eye(4)
            pose_init_raw[:3, :3] = torch.from_numpy(np.array(rot).reshape(3, 3).astype(np.float32))
            pose_init_raw[:3, 3] = torch.from_numpy(np.array(tra).astype(np.float32)) / 1000  # scale in meter
            pose_init = self.parse_raw_camera(opt, pose_init_raw) 
        else:
            pose_init = pose_gt

        return cam_K, intr, pose_gt, pose_init

    @staticmethod
    def parse_raw_camera(opt, pose_raw):
        # Acquire the pose transform point from world to camera
        pose_eye = camera.pose(R=torch.diag(torch.tensor([1, 1, 1])))
        pose = camera.pose.compose([pose_eye, pose_raw[:3]])
        # Scaling
        pose[:, 3] *= opt.nerf.depth.scale
        return pose

    @staticmethod
    def preprocess_intrinsics(cam_K, resize, crop_center, res):
        # This implementation is tested faithfully. Results in PnP with 0.02% drop.
        K = cam_K

        # First resize from original size to the target size
        K[0, 0] = K[0, 0] * resize
        K[1, 1] = K[1, 1] * resize
        K[0, 2] = (K[0, 2] + 0.5) * resize - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * resize - 0.5

        # Then crop the image --> need to modify the optical center,
        # remember that current top left is the coordinates measured in resized results
        # And its information is vu instead of uv
        top_left = crop_center * resize - res / 2
        K[0, 2] = K[0, 2] - top_left[1]
        K[1, 2] = K[1, 2] - top_left[0]
        return K

    @staticmethod
    def get_center_offset(center, scale, ht, wd):
        upper = max(0, int(center[0] - scale / 2. + 0.5))
        left = max(0, int(center[1] - scale / 2. + 0.5))
        bottom = min(ht, int(center[0] - scale / 2. + 0.5) + int(scale))
        right = min(wd, int(center[1] - scale / 2. + 0.5) + int(scale))

        if upper == 0:
            h_offset = - int(center[0] - scale / 2. + 0.5) / 2
        elif bottom == ht:
            h_offset = - (int(center[0] - scale / 2. + 0.5) + int(scale) - ht) / 2
        else:
            h_offset = 0

        if left == 0:
            w_offset = - int(center[1] - scale / 2. + 0.5) / 2
        elif right == wd:
            w_offset = - (int(center[1] - scale / 2. + 0.5) + int(scale) - wd) / 2
        else:
            w_offset = 0
        center_offset = np.array([h_offset, w_offset])
        return center_offset

    @staticmethod
    # code adopted from CDPN: https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
    def Crop_by_Pad(img, center, scale, res=None, channel=3, interpolation=cv2.INTER_LINEAR, resize=True):
        # Code from CDPN
        ht, wd = img.shape[0], img.shape[1]

        upper = max(0, int(center[0] - scale / 2. + 0.5))
        left = max(0, int(center[1] - scale / 2. + 0.5))
        bottom = min(ht, int(center[0] - scale / 2. + 0.5) + int(scale))
        right = min(wd, int(center[1] - scale / 2. + 0.5) + int(scale))
        crop_ht = float(bottom - upper)
        crop_wd = float(right - left)

        if resize:
            if crop_ht > crop_wd:
                resize_ht = res
                resize_wd = int(res / crop_ht * crop_wd + 0.5)
            elif crop_ht < crop_wd:
                resize_wd = res
                resize_ht = int(res / crop_wd * crop_ht + 0.5)
            else:
                resize_wd = resize_ht = int(res)

        if channel <= 3:
            tmpImg = img[upper:bottom, left:right]
            if not resize:
                outImg = np.zeros((int(scale), int(scale), channel))
                outImg[int(scale / 2.0 - (bottom - upper) / 2.0 + 0.5):(
                        int(scale / 2.0 - (bottom - upper) / 2.0 + 0.5) + (bottom - upper)),
                int(scale / 2.0 - (right - left) / 2.0 + 0.5):(
                        int(scale / 2.0 - (right - left) / 2.0 + 0.5) + (right - left)), :] = tmpImg
                return outImg

            resizeImg = cv2.resize(tmpImg, (resize_wd, resize_ht), interpolation=interpolation)
            if len(resizeImg.shape) < 3:
                resizeImg = np.expand_dims(resizeImg, axis=-1)  # for depth image, add the third dimension
            outImg = np.zeros((res, res, channel))
            outImg[int(res / 2.0 - resize_ht / 2.0 + 0.5):(int(res / 2.0 - resize_ht / 2.0 + 0.5) + resize_ht),
            int(res / 2.0 - resize_wd / 2.0 + 0.5):(int(res / 2.0 - resize_wd / 2.0 + 0.5) + resize_wd), :] = resizeImg

        else:
            raise NotImplementedError
        return outImg

    @staticmethod
    # code adopted from GDRN: https://github.com/THU-DA-6D-Pose-Group/GDR-Net
    def get_edge(mask, bw=1, out_channel=3):
        if len(mask.shape) > 2:
            channel = mask.shape[2]
        else:
            channel = 1
        if channel == 3:
            mask = mask[:, :, 0] != 0
        edges = np.zeros(mask.shape[:2])
        edges[:-bw, :] = np.logical_and(mask[:-bw, :] == 1, mask[bw:, :] == 0) + edges[:-bw, :]
        edges[bw:, :] = np.logical_and(mask[bw:, :] == 1, mask[:-bw, :] == 0) + edges[bw:, :]
        edges[:, :-bw] = np.logical_and(mask[:, :-bw] == 1, mask[:, bw:] == 0) + edges[:, :-bw]
        edges[:, bw:] = np.logical_and(mask[:, bw:] == 1, mask[:, :-bw] == 0) + edges[:, bw:]
        if out_channel == 3:
            edges = np.dstack((edges, edges, edges))
        return edges

    def smooth_geo(self, x):
        """smooth the edge areas to reduce noise."""
        x = np.asarray(x, np.float32)
        x_blur = cv2.medianBlur(x, 3)
        edges = self.get_edge(x)
        x[edges != 0] = x_blur[edges != 0]
        return x
