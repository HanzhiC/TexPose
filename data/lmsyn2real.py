import numpy as np
import cv2
import os, sys, time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import json
import pickle

from . import base
import camera
from util import log, debug, readlines


class Dataset(base.Dataset):

    def __init__(self, opt, split="train", subset=None, multi_obj=False):
        self.raw_H, self.raw_W = 480, 640
        super().__init__(opt, split)
        assert opt.H / self.raw_H == opt.W / self.raw_W

        self.data_path = os.path.join(opt.data.root, opt.data.dataset)  # Data path to the bop dataset
        self.split_path = os.path.join("splits", opt.data.dataset, str(opt.data.object),
                                       opt.data.scene, "{}.txt".format(split))
        self.list = readlines(self.split_path)
        self.multi_obj = multi_obj

        if subset: self.list = self.list[:subset]
        if self.multi_obj: assert not opt.data.preload

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
        if self.split == 'train' and 'adapt_st' in opt.model and opt.data.pose_source == 'predicted':
            scene_info_path = os.path.join(self.data_path, folder, 'scene_pred_info.json')
        else:
            scene_info_path = os.path.join(self.data_path, folder, 'scene_gt_info.json')
        with open(scene_info_path) as f_info:
            self.scene_info_all = json.load(f_info)
            f_info.close()
        # load scene pose
        scene_gt_path = os.path.join(self.data_path, folder, 'scene_gt.json')
        scene_cam_path = os.path.join(self.data_path, folder, 'scene_camera.json')
        scene_pred_path = os.path.join(self.data_path, folder, 'scene_pred_{}.json'.format(opt.data.pose_loop))
        with open(scene_gt_path) as f_gt:
            self.scene_gt_all = json.load(f_gt)
            f_gt.close()
        with open(scene_cam_path) as f_cam:
            self.scene_cam_all = json.load(f_cam)
            f_cam.close()
        if 'adapt_st' in opt.model and self.split == 'train' and opt.data.pose_source == 'predicted':
            with open(scene_pred_path) as f_pred:
                self.scene_pred_all = json.load(f_pred)
                f_pred.close()
            del f_pred
        else:
            self.scene_pred_all = self.scene_gt_all
        del f_cam, f_gt, f_info
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
        intr, pose_gt, pose_init = self.cameras[idx] if opt.data.preload else self.get_camera(opt, idx,
                                                                                              obj_scene_id=obj_scene_id)
        z_near, z_far = self.get_range(opt, idx, obj_scene_id=obj_scene_id)
        obj_mask = self.get_obj_mask(opt, idx, obj_scene_id=obj_scene_id)

        frame_index = torch.tensor(int(self.list[idx].split()[2]))
        sample.update(
            image=image,
            intr=intr,
            pose=pose_gt,
            pose_init=pose_init,
            z_near=z_near,
            z_far=z_far,
            obj_mask=obj_mask,
            frame_index=frame_index)
        return sample

    def get_image(self, opt, idx, ext='.png', obj_scene_id=0):
        # Parse the split text
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}{}".format(frame_index, ext)
        image_fname = os.path.join(self.data_path, folder, 'rgb', file)
        # directly using PIL.Image.open() leads to weird corruption....
        image = PIL.Image.fromarray(imageio.imread(image_fname))
        image = image.resize((opt.W, opt.H))
        image = torchvision_F.to_tensor(image)

        return image

    def get_depth(self, opt, idx, ext='.png', obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}{}".format(frame_index, ext)
        depth_fname = os.path.join(self.data_path, folder, 'depth', file)
        # depth = PIL.Image.fromarray(imageio.imread(depth_fname) / 1000.)  # stored in mm, scale in m
        depth = cv2.imread(depth_fname, -1) / 1000.
        depth = cv2.resize(depth, (opt.W, opt.H))
        depth = torch.from_numpy(depth).float().squeeze(-1)

        # depth = self.preprocess_image(opt, depth, aug=None)
        mask = self.get_obj_mask(opt, idx)
        depth *= opt.nerf.depth.scale
        depth *= mask.float()

        return depth

    def get_obj_mask(self, opt, idx, ext='.png', return_visib=True, return_erode=False, obj_scene_id=0):
        # TODO: replace such object mask acquisition module with predicted mask
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}_{:06d}{}".format(frame_index, obj_scene_id, ext)



        #####################
        if opt.data.scene == 'scene_vis':
            depth_file = "{:06d}{}".format(frame_index, '.png')
            depth_fname = os.path.join(self.data_path, folder, 'depth', depth_file)
            # depth = PIL.Image.fromarray(imageio.imread(depth_fname) / 1000.)  # stored in mm, scale in m
            depth = cv2.imread(depth_fname, -1) / 1000.
            depth = cv2.resize(depth, (opt.W, opt.H))
            depth = torch.from_numpy(depth).float().squeeze(-1)
            mask_full = (depth > 0).float()
        else:
            # Full mask through rasterization-based rendering
            mask_fname_full = os.path.join(self.data_path, folder, 'mask_visib', file)
            mask_full = cv2.imread(mask_fname_full, -1)
            mask_full = cv2.resize(mask_full, (opt.W, opt.H))
            mask_full = torch.from_numpy(mask_full.astype(np.float32)).squeeze(-1)
        #####################


        # Visible region
        if self.split == 'train':
            # visib_source = 'mask_pred_orig' if opt.model == 'nerf_adapt_st' else 'mask_visib'
            visib_source = 'mask_visib'
            mask_fname_visib = os.path.join(self.data_path, folder, visib_source, file)
            mask_visib = cv2.imread(mask_fname_visib, -1)
            mask_visib = cv2.resize(mask_visib, (opt.W, opt.H))
            mask_visib = torch.from_numpy(mask_visib.astype(np.float32)).squeeze(-1)
        else:
            if opt.data.scene == 'scene_vis':
                mask_visib = mask_full
            else:
                # mask_visib = torch.ones_like(mask_full)
                visib_source = 'mask_visib'
                mask_fname_visib = os.path.join(self.data_path, folder, visib_source, file)
                mask_visib = cv2.imread(mask_fname_visib, -1)
                mask_visib = cv2.resize(mask_visib, (opt.W, opt.H))
                mask_visib = torch.from_numpy(mask_visib.astype(np.float32)).squeeze(-1)

        # Rendered mask using predicted pose
        # if opt.data.pose_source == 'predicted':
        #     mask_fname_render = os.path.join(self.data_path, folder, 'mask_render_init_orig', file)
        #     mask_r = cv2.imread(mask_fname_render, -1)
        #     mask_r = torch.from_numpy(mask_r).squeeze(-1)
        # else:
        mask_r = torch.ones_like(mask_full)

        if return_visib:
            if self.split == 'train':
                # Used for loss calculation and blending --> with imperfect pose, so only consider intersection regions
                obj_mask = torch.logical_and(mask_visib > 0, mask_r > 0)
            elif self.split == 'test':
                # Deprecated!
                # Used for blending --> during eval/test phase, blend the background with the object
                # If GT annotation is perfect we actually don't need AND operation for mask_vsisib and mask_full
                # obj_mask = torch.logical_and(mask_visib > 0, mask_full > 0)
                obj_mask = mask_full > 0
            else:
                obj_mask = mask_full > 0
        else:
            # Use for depth range masking, will not be used during val/test, only use it during data preparation phase
            if self.split == 'train':
                obj_mask = mask_visib > 0
            else:
                obj_mask = mask_full > 0
        return obj_mask.float()

    def get_range(self, opt, idx, obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])

        depth_min_bg, depth_max_bg = opt.nerf.depth.range
        depth_min_bg *= opt.nerf.depth.scale
        depth_max_bg *= opt.nerf.depth.scale
        depth_min_bg = torch.Tensor([depth_min_bg]).float().expand(opt.H * opt.W)
        depth_max_bg = torch.Tensor([depth_max_bg]).float().expand(opt.H * opt.W)
        mask = self.get_obj_mask(opt, idx).float()

        if opt.nerf.depth.range_source == 'box':
            if opt.data.pose_source == 'predicted' and self.split == 'train':
                box_source = opt.nerf.depth.box_source
            else:
                box_source = 'gt_box'

            # Get meta-information, pose and intrinsics
            if self.multi_obj:
                file = "{:06d}_{:06d}{}".format(frame_index, obj_scene_id, '.npz')
            else:
                file = "{:06d}{}".format(frame_index, '.npz')
            box_fname = os.path.join(self.data_path, folder, box_source, file)
            box_range = np.load(box_fname)["data"]
            box_range = cv2.resize(box_range.transpose((1, 2, 0)), (opt.W, opt.H))
            box_range = box_range.astype(np.float32)
            box_range = torch.from_numpy(box_range)

            if opt.nerf.depth.box_mask:
                box_range = box_range * mask[..., None]
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

        cam = self.scene_cam_all[str(frame_index)]["cam_K"]
        intr = torch.from_numpy(np.array(cam).reshape(3, 3).astype(np.float32))
        cam_K = self.scene_cam_all[str(frame_index)]["cam_K"]
        cam_K = torch.from_numpy(np.array(cam_K, dtype=np.float32).reshape(3, 3))
        cam_intr = cam_K.clone()
        resize = opt.H / self.raw_H  # only assume
        intr = self.preprocess_intrinsics(cam_intr, resize=resize)

        # Load GT pose
        rot = self.scene_gt_all[str(frame_index)][obj_scene_id]['cam_R_m2c']
        tra = self.scene_gt_all[str(frame_index)][obj_scene_id]['cam_t_m2c']
        pose_gt_raw = torch.eye(4)
        pose_gt_raw[:3, :3] = torch.from_numpy(np.array(rot).reshape(3, 3).astype(np.float32))
        pose_gt_raw[:3, 3] = torch.from_numpy(np.array(tra).astype(np.float32)) / 1000  # scale in meter
        pose_gt = self.parse_raw_camera(opt, pose_gt_raw)

        # Load predicted pose
        if opt.model == 'nerf_adapt_st' and self.split == 'train':
            # TODO: save predicted pose before training
            rot = self.scene_pred_all[str(frame_index)][obj_scene_id]['cam_R_m2c']
            tra = self.scene_pred_all[str(frame_index)][obj_scene_id]['cam_t_m2c']
            pose_init_raw = torch.eye(4)
            pose_init_raw[:3, :3] = torch.from_numpy(np.array(rot).reshape(3, 3).astype(np.float32))
            pose_init_raw[:3, 3] = torch.from_numpy(np.array(tra).astype(np.float32)) / 1000  # scale in meter
            pose_init = self.parse_raw_camera(opt, pose_init_raw) if idx != 0 else pose_gt
        else:
            pose_init = pose_gt

        return intr, pose_gt, pose_init

    @staticmethod
    def preprocess_intrinsics(cam_K, resize):
        # This implementation is tested faithfully. Results in PnP with 0.02% drop.
        K = cam_K

        # First resize from original size to the target size
        K[0, 0] = K[0, 0] * resize
        K[1, 1] = K[1, 1] * resize
        K[0, 2] = (K[0, 2] + 0.5) * resize - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * resize - 0.5
        return K

    @staticmethod
    def parse_raw_camera(opt, pose_raw):
        # Acquire the pose transform point from world to camera
        pose_eye = camera.pose(R=torch.diag(torch.tensor([1, 1, 1])))
        pose = camera.pose.compose([pose_eye, pose_raw[:3]])
        # Scaling
        pose[:, 3] *= opt.nerf.depth.scale
        return pose
