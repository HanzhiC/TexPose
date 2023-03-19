import numpy as np
import os, sys, time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import cv2

import lpips
from external.pohsun_ssim import pytorch_ssim

import util, util_vis
from util import log, debug
from . import base
import camera
import random
# from renderer import MVRenderer
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes
from kornia.geometry.linalg import inverse_transformation as inv_pose
from data.cad_model import CAD_Model
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.loss.chamfer import _validate_chamfer_reduction_inputs, _handle_pointcloud_input
from layers.nerf import NeRF
import importlib

# ============================ main engine for training and evaluation ============================

class Model(base.Model):

    def __init__(self, opt):
        super().__init__(opt)
        self.lpips_loss = lpips.LPIPS(net="alex").to(opt.device)

    def load_dataset(self, opt, eval_split="val"):
        if opt.data.image_size != [128, 128]:
            dataset_name = opt.data.dataset + 'f'
        else:
            dataset_name = opt.data.dataset
        data = importlib.import_module("data.{}".format(dataset_name))
        log.info("loading training data...")
        # TODO: Modify it for LM
        train_multiobj = True if opt.data.scene in ['scene_hbbdp_finetune', 'scene_hbbdp'] else False
        # train_multiobj = True if opt.data.dataset == 'hb' else False
        self.train_data = data.Dataset(opt, split="train", subset=opt.data.train_sub, multi_obj=train_multiobj)
        # self.train_data = data.Dataset(opt, split="train", subset=opt.data.train_sub, multi_obj=opt.data.multi_obj)
        self.train_loader = self.train_data.setup_loader(opt, shuffle=True, drop_last=True)
        log.info("loading test data...")
        if opt.data.val_on_test:
            eval_split = "test"
        self.test_data = data.Dataset(opt, split=eval_split, subset=opt.data.val_sub, multi_obj=opt.data.multi_obj)
        self.test_loader = self.test_data.setup_loader(opt, shuffle=False)  # prefetch all training data
        self.train_data.prefetch_all_data(opt)
        self.train_data.all = edict(util.move_to_device(self.train_data.all, opt.device))

    def setup_optimizer(self, opt):
        log.info("setting up optimizers...")
        optimizer = getattr(torch.optim, opt.optim.algo)
        self.optim = optimizer([dict(params=self.graph.nerf.parameters(), lr=opt.optim.lr)])
        if opt.nerf.fine_sampling:
            self.optim.add_param_group(dict(params=self.graph.nerf_fine.parameters(), lr=opt.optim.lr))
        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched.type)
            if opt.optim.lr_end:
                assert (opt.optim.sched.type == "ExponentialLR")
                # opt.optim.sched.gamma = (opt.optim.lr_end / opt.optim.lr) ** (1. / opt.max_iter)
            kwargs = {k: v for k, v in opt.optim.sched.items() if k != "type"}
            self.sched = scheduler(self.optim, **kwargs)

    def train_iteration(self, opt, var, loader):
        loss = super().train_iteration(opt, var, loader)
        if opt.c2f is not None:
            self.graph.nerf.progress.data.fill_(self.it / opt.max_iter)
        return loss

    def train(self, opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(), it_mean=None)
        self.graph.train()
        self.ep = 0  # dummy for timer
        # training
        if self.iter_start == 0: self.validate(opt, 0)
        loader = tqdm.trange(opt.max_iter, desc="training", leave=False)
        for self.it in loader:
            if self.it < self.iter_start: continue
            # set var to all available images
            var = self.train_data.all
            self.train_iteration(opt, var, loader)
            if opt.optim.sched: self.sched.step()
            if self.it % opt.freq.val == 0: self.validate(opt, self.it)
            if self.it % opt.freq.ckpt == 0: self.save_checkpoint(opt, ep=None, it=self.it)
        # after training
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom: self.vis.close()
        log.title("TRAINING DONE")

    @torch.no_grad()
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        super().log_scalars(opt, var, loss, metric=metric, step=step, split=split)
        # log learning rate
        if split == "train":
            lr = self.optim.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split, "lr"), lr, step)
            if opt.nerf.fine_sampling:
                lr = self.optim.param_groups[1]["lr"]
                self.tb.add_scalar("{0}/{1}".format(split, "lr_fine"), lr, step)

        # compute PSNR
        psnr = -10 * loss.render.log10()
        self.tb.add_scalar("{0}/{1}".format(split, "PSNR"), psnr, step)

        if opt.nerf.fine_sampling:
            psnr = -10 * loss.render_fine.log10()
            self.tb.add_scalar("{0}/{1}".format(split, "PSNR_fine"), psnr, step)

    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train", eps=1e-10):
        if opt.tb:
            util_vis.tb_image(opt, self.tb, step, split, "image", var.image)
            gt_mask = var.obj_mask.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2).float()  # [B,1,H,W]
            if opt.data.erode_mask_loss:
                erode_mask = var.erode_mask.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2).float()  # [B,1,H,W]
                util_vis.tb_image(opt, self.tb, step, split, "image_masked", var.image * erode_mask + (1 - erode_mask))
            else:
                util_vis.tb_image(opt, self.tb, step, split, "image_masked", var.image * gt_mask + (1 - gt_mask))
            util_vis.tb_image(opt, self.tb, step, split, "gt_mask", gt_mask)
            util_vis.tb_image(opt, self.tb, step, split, "z_near",
                              var.z_near.float().view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2),
                              from_range=(0.9 * opt.nerf.depth.scale,  var.z_near.max()), cmap='plasma')
            if split == 'train':
                # samples_obj = util_vis.draw_samples(opt, var, 'object')
                # util_vis.tb_image(opt, self.tb, step, split, "samples_object", samples_obj, cmap='viridis')
                # samples_bg = util_vis.draw_samples(opt, var, 'background')
                # util_vis.tb_image(opt, self.tb, step, split, "samples_background", samples_bg, cmap='viridis')
                util_vis.tb_image(opt, self.tb, step, split, "depth_gt", var.depth_gt[:, None],
                                  from_range=(0.7 * opt.nerf.depth.scale, var.depth_gt.max()), cmap='plasma')

            if not opt.nerf.rand_rays or split != "train":
                mask = (var.obj_mask > 0).view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2).float()

                rgb_map = var.rgb.view(-1, opt.H, opt.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
                depth_map = var.depth.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,3,H,W]
                pred_mask = var.opacity.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2).clamp(0, 1)  # [B,1,H,W]
                gt_mask = var.obj_mask.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2).float()  # [B,1,H,W]
                depth_error = (depth_map - var.depth_gt[:, None]).abs().view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)
                depth_error *= mask

                util_vis.tb_image(opt, self.tb, step, split, "rgb", rgb_map)

                util_vis.tb_image(opt, self.tb, step, split, "pred_mask", pred_mask)
                util_vis.tb_image(opt, self.tb, step, split, "gt_mask", gt_mask)
                util_vis.tb_image(opt, self.tb, step, split, "depth", depth_map * gt_mask,
                                  from_range=(0.7 * opt.nerf.depth.scale, depth_map.max()), cmap='plasma')
                util_vis.tb_image(opt, self.tb, step, split, "depth_gt", var.depth_gt[:, None],
                                  from_range=(0.7 * opt.nerf.depth.scale, depth_map.max()), cmap='plasma')
                util_vis.tb_image(opt, self.tb, step, split, "depth_error", depth_error,
                                  from_range=(0, torch.quantile(depth_error, 0.99)), cmap='turbo')

    @torch.no_grad()
    def get_all_training_poses(self, opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        return None, pose_GT

    @torch.no_grad()
    def evaluate_full(self, opt, eps=1e-10):
        self.graph.eval()
        loader = tqdm.tqdm(self.test_loader, desc="evaluating", leave=False)
        res = []
        test_path_rgb = "{}/rgb".format(opt.output_path)
        test_path_opacity = "{}/opacity".format(opt.output_path)

        os.makedirs(test_path_rgb, exist_ok=True)
        os.makedirs(test_path_opacity, exist_ok=True)

        for i, batch in enumerate(loader):
            with torch.no_grad():
                var = edict(batch)
                var = util.move_to_device(var, opt.device)
                if opt.model == "barf" and opt.optim.test_photo:
                    # run test-time optimization to factorize imperfection in optimized poses
                    # from view synthesis evaluation
                    var = self.evaluate_test_time_photometric_optim(opt, var)
                # var = self.graph.forward(opt, var, mode="eval_no_align")
                var = self.graph.forward(opt, var, mode="eval_noalign")
                pose_gt_np = var.pose.detach().cpu().squeeze().numpy()
                pose_pred_np = var.pose_init.detach().cpu().squeeze().numpy()

                # np.save("{}/pose_gt_{}.npy".format(test_path, i), pose_gt_np)
                # np.save("{}/pose_init_{}.npy".format(test_path, i), pose_pred_np)

                # evaluate view synthesis
                invdepth = (1 - var.depth) / var.opacity if opt.camera.ndc else 1 / (var.depth / var.opacity + eps)
                invdepth_map = invdepth.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]

                rgb_map = var.rgb.view(-1, opt.H, opt.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
                depth_map = var.depth.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]
                mask_map = var.obj_mask.float().view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)
                opacity = var.opacity.float().view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)
                # rgb_map *= mask_map
                var.image *= mask_map
                # depth_map *= mask_map
                image = var.image #* mask_map
                fused = torch.cat([rgb_map, image], dim=-1)

                depth_map /= opt.nerf.depth.scale  # scale the to-save depth to metric in meter
                depth_np = (depth_map * 2000).squeeze().cpu().numpy().astype(np.uint16)

                psnr = -10 * self.graph.MSE_loss(rgb_map, image).log10().item()
                ssim = pytorch_ssim.ssim(rgb_map, image).item()
                lpips = self.lpips_loss(rgb_map * 2 - 1, image * 2 - 1).item()
                res.append(edict(psnr=psnr, ssim=ssim, lpips=lpips))
                # dump novel views
                frame_idx = str(var.frame_index.cpu().item()).zfill(6)
                torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/{}.png".format(test_path_rgb, frame_idx))
                torchvision_F.to_pil_image(opacity.cpu()[0]).save("{}/{}.png".format(test_path_opacity, frame_idx))

                # torchvision_F.to_pil_image(fused.cpu()[0]).save("{}/rgb_GT_{}.png".format(test_path, frame_idx))
                # torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save("{}/inv_depth_{}.png".format(test_path, i))
                # cv2.imwrite("{}/depth_{}.png".format(test_path, frame_idx), depth_np)

        # show results in terminal
        print("--------------------------")
        print("PSNR:  {:8.2f}".format(np.mean([r.psnr for r in res])))
        print("SSIM:  {:8.2f}".format(np.mean([r.ssim for r in res])))
        print("LPIPS: {:8.2f}".format(np.mean([r.lpips for r in res])))
        print("--------------------------")
        # dump numbers to file
        quant_fname = "{}/quant.txt".format(opt.output_path)
        with open(quant_fname, "w") as file:
            for i, r in enumerate(res):
                file.write("{} {} {} {}\n".format(i, r.psnr, r.ssim, r.lpips))

    @torch.no_grad()
    def generate_videos_synthesis(self, opt, eps=1e-10):
        self.graph.eval()
        if opt.data.dataset == "blender":
            test_path = "{}/test_view".format(opt.output_path)
            # assume the test view synthesis are already generated
            print("writing videos...")
            rgb_vid_fname = "{}/test_view_rgb.mp4".format(opt.output_path)
            depth_vid_fname = "{}/test_view_depth.mp4".format(opt.output_path)
            os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_path,
                                                                                                              rgb_vid_fname))
            os.system(
                "ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_path,
                                                                                                          depth_vid_fname))
        else:
            pose_pred, pose_GT = self.get_all_training_poses(opt)
            poses = pose_pred if opt.model == "barf" else pose_GT
            # pose_pred[:, :, 3] *= opt.nerf.depth.scale
            # pose_GT[:, :, 3] *= opt.nerf.depth.scale

            scale = 1
            # if opt.model == "barf" and opt.data.dataset == "llff":
            #     _, sim3 = self.prealign_cameras(opt, pose_pred, pose_GT)
            #     scale = sim3.s1 / sim3.s0
            # else:
            #     scale = 1
            # rotate novel views around the "center" camera of all poses
            idx_center = (poses - poses.mean(dim=0, keepdim=True))[..., 3].norm(dim=-1).argmin()
            idx_center = 5
            # pose_novel = camera.get_novel_view_poses(opt, poses[idx_center],
            #                                          N=10, scale=scale, motion='gentle').to(opt.device)
            pose_novel = camera.get_novel_view_poses_obj(opt, poses[idx_center], N=10).to(opt.device)

            # pose_novel = camera.get_novel_view_poses_obj(opt, poses[idx_center], N=60).to(opt.device)
            # pose_novel[:, :, :3] /= opt.nerf.depth.scale  # scale to metric in meter and save
            #
            pose_novel = pose_pred
            to_save = pose_novel.cpu().numpy()

            # render the novel views
            novel_path = "{}/novel_view".format(opt.output_path)
            os.makedirs(novel_path, exist_ok=True)
            np.save(os.path.join(novel_path, 'novel_pose.npy'), to_save)

            pose_novel_tqdm = tqdm.tqdm(pose_novel, desc="rendering novel views", leave=False)
            intr = edict(next(iter(self.test_loader))).intr[:1].to(opt.device)  # grab one intrinsics

            renderers = {}
            vert_max = {}
            vert_min = {}

            for idx in opt.data.reference_obj_idx + opt.data.dynamic_object_idx:
                verts, faces = load_ply(
                    "../BOP_dataset/{}/{}_models/models/obj_{}.ply".format(opt.dataset, opt.dataset, str(idx).zfill(6)))
                cad_mesh = Meshes(verts=verts[None].to(opt.device), faces=faces[None].to(opt.device))
                mv_renderer = MVRenderer(cad_mesh.cuda(), 240, 320, 1, intr.squeeze(), 'complex')

                cad_model = CAD_Model()
                cad_model.load(
                    "../BOP_dataset/{}/{}_models/models/obj_{}.ply".format(opt.dataset, opt.dataset, str(idx).zfill(6)))

                renderers[idx] = mv_renderer
                vert_min[idx] = torch.Tensor(cad_model.bb[0])[None, None, :].cuda()
                vert_max[idx] = torch.Tensor(cad_model.bb[-1])[None, None, :].cuda()
                del cad_model

            for i, pose in enumerate(pose_novel_tqdm):
                depth_min_bg, depth_max_bg = opt.nerf.depth.range
                depth_min_bg *= opt.nerf.depth.scale
                depth_max_bg *= opt.nerf.depth.scale
                depth_min_bg = torch.Tensor([depth_min_bg]).float().expand(1, opt.H * opt.W).to(opt.device)
                depth_max_bg = torch.Tensor([depth_max_bg]).float().expand(1, opt.H * opt.W).to(opt.device)
                # print(mask_blend.shape, z_near.shape, depth_min_bg.shape)
                if opt.nerf.depth.range_source == 'box':
                    z_nears = {}
                    z_fars = {}
                    depth_render = {}
                    label_render = {}
                    for obj_id in opt.data.reference_obj_idx + opt.data.dynamic_object_idx:
                        # Acquire the vertices of the CAD model
                        aabb_min = (vert_min[obj_id] * opt.nerf.depth.scale) / 1000
                        aabb_max = (vert_max[obj_id] * opt.nerf.depth.scale) / 1000

                        # Compose them together and do render, variable pose is the deforming pose, remember to
                        # invert it back
                        pose_novel = pose
                        ray_o, ray_d = camera.get_center_and_ray(opt, pose_novel[None], intr=intr)
                        z_near, z_far, valid = camera.aabb_ray_intersection(aabb_min, aabb_max, ray_o, ray_d)
                        z_nears[obj_id] = torch.where(valid > 0, z_near, torch.zeros_like(z_near))
                        z_fars[obj_id] = torch.where(valid > 0, z_far, torch.zeros_like(z_far))

                        # Render them together for Z-buffer
                        renderer = renderers[obj_id]
                        pose_render = torch.cat([pose_novel.clone(),
                                                 torch.Tensor([[0, 0, 0, 1]]).cuda()], dim=0)
                        pose_render[:3, 3] = (pose_render[:3, 3] / opt.nerf.depth.scale) * 1000

                        # Compute the deformed pose, that is very tricky need to be careful!
                        _, depth = renderer.render_nocs(pose_render.cuda()[None], True)
                        depth_render[obj_id] = depth.view(-1, opt.H * opt.W)
                        label_render[obj_id] = depth.view(-1, opt.H * opt.W) > 0

                    # Blending the depth, label and range using z-buffer
                    depth_final = [depth_render[obj_id] for obj_id in
                                   opt.data.reference_obj_idx + opt.data.dynamic_object_idx]
                    label_final = [label_render[obj_id] * obj_id for obj_id in
                                   opt.data.reference_obj_idx + opt.data.dynamic_object_idx]
                    z_near_final = [z_nears[obj_id] for obj_id in
                                    opt.data.reference_obj_idx + opt.data.dynamic_object_idx]
                    z_far_final = [z_fars[obj_id] for obj_id in
                                   opt.data.reference_obj_idx + opt.data.dynamic_object_idx]

                    depth_final = torch.cat(depth_final, dim=0)
                    label_final = torch.cat(label_final, dim=0)
                    z_near_final = torch.cat(z_near_final, dim=0)
                    z_far_final = torch.cat(z_far_final, dim=0)

                    depth_final = torch.where(depth_final > 0, depth_final, 100000 * torch.ones_like(depth_final))
                    depth_blend, near_idx = torch.min(depth_final, dim=0)

                    label_blend = torch.gather(input=label_final, dim=0, index=near_idx[None])
                    z_near_blend = torch.gather(input=z_near_final, dim=0, index=near_idx[None])
                    z_far_blend = torch.gather(input=z_far_final, dim=0, index=near_idx[None])

                    # Acquire the final range
                    depth_min = torch.where(label_blend > 0, z_near_blend, depth_min_bg)
                    depth_max = torch.where(label_blend > 0, z_far_blend, depth_max_bg)

                # TODO: Implement N-render for object depth ray sampling of the novel view
                elif opt.nerf.depth.range_source == 'render':
                    depth_render = {}
                    for obj_id in [6]:
                        renderer = renderers[obj_id]
                        # pose_obj = pose_in_world[obj_id].cuda()
                        pose_obj = torch.eye(4).unsqueeze(0).cuda()
                        pose_homo = torch.eye(4).unsqueeze(0).cuda()
                        pose_homo[:, :3, :3] = pose[:3, :3]
                        pose_homo[:, :3, 3] = (pose[:3, 3] / opt.nerf.depth.scale) * 1000

                        render_pose = pose_homo @ inv_pose(pose_obj)

                        _, depth = renderer.render_nocs(render_pose.cuda(), True)
                        depth_render[obj_id] = depth

                    # depth_final = [depth_render[obj_id] for obj_id in [1, 4, 6, 5, 8, 9, 11]]
                    depth_final = [depth_render[obj_id] for obj_id in [6]]

                    depth_final = torch.cat(depth_final, dim=0)
                    mask_final = depth_final > 0
                    depth_final = torch.where(mask_final,
                                              depth_final,
                                              100000 * torch.ones_like(depth_final))
                    depth_blend, _ = torch.min(depth_final, dim=0)
                    mask_blend = torch.sum(mask_final, dim=0) > 0
                    depth_blend *= mask_blend
                    depth = (depth_blend / 1000.) * opt.nerf.depth.scale
                    depth = depth.view(opt.H * opt.W)[None]
                    mask_blend = mask_blend.view(opt.H * opt.W)[None]

                    depth_render_vis = depth.view(opt.H, opt.W).cpu().numpy().astype(np.uint16)
                    cv2.imwrite("{}/depth_r_{}.png".format(novel_path, i), depth_render_vis * 2000)

                    z_near, z_far = depth * 0.8, depth * 1.2

                    depth_min = torch.where(mask_blend > 0, z_near, depth_min_bg)
                    depth_max = torch.where(mask_blend > 0, z_far, depth_max_bg)

                else:
                    depth_min, depth_max = opt.nerf.depth.range
                    depth_min *= opt.nerf.depth.scale
                    depth_max *= opt.nerf.depth.scale
                    depth_min = torch.Tensor([depth_min]).float().expand(1, opt.H * opt.W).to(opt.device)
                    depth_max = torch.Tensor([depth_max]).float().expand(1, opt.H * opt.W).to(opt.device)

                # Do rendering!
                depth_range = (depth_min[:, :, None], depth_max[:, :, None])
                ret = self.graph.render_by_slices(opt,
                                                  pose[None],
                                                  intr=intr,
                                                  depth_range=depth_range) \
                    if opt.nerf.rand_rays else \
                    self.graph.render(opt, pose[None], intr=intr, depth_range=depth_range)

                invdepth = (1 - ret.depth) / ret.opacity if opt.camera.ndc else 1 / (ret.depth / ret.opacity + eps)
                invdepth_map = invdepth.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]

                rgb_map = ret.rgb.view(-1, opt.H, opt.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
                depth_map = ret.depth.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]
                depth_map /= opt.nerf.depth.scale  # scale the to-save depth to metric in meter
                depth_np = (depth_map * 2000).squeeze().cpu().numpy().astype(np.uint16)
                # print(depth_np.shape)

                cv2.imwrite("{}/depth_{}.png".format(novel_path, i), depth_np)
                torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(novel_path, i))
                torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save("{}/inv_depth_{}.png".format(novel_path, i))

            # write videos
            # print("writing videos...")
            # rgb_vid_fname = "{}/novel_view_rgb.mp4".format(opt.output_path)
            # depth_vid_fname = "{}/novel_view_depth.mp4".format(opt.output_path)
            # # os.system(
            # #     "ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1}".format(novel_path, rgb_vid_fname))
            # # os.system(
            # #     "ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1}".format(novel_path, depth_vid_fname))


# ============================ computation graph for forward/backprop ============================

class Graph(base.Graph):

    def __init__(self, opt):
        super().__init__(opt)
        self.nerf = NeRF(opt)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF(opt)

    @staticmethod
    def ray_batch_sample(ray_identity, ray_idx):
        assert ray_identity.shape[0] == ray_idx.shape[0]
        B, HW, _ = ray_identity.shape
        B, N_sample = ray_idx.shape
        ray_identity = ray_identity.view(B * HW, -1)
        samples_cumsum = ray_idx + HW * torch.arange(B).cuda().unsqueeze(1)  # No need to reshape
        ray_identity_sample = ray_identity[samples_cumsum].view(B, N_sample, -1)
        return ray_identity_sample

    # @staticmethod
    # def get_ray_idx(opt, var):
    #     batch_size = len(var.idx)
    #
    #     visib_mask = var.obj_mask.long().view(batch_size, opt.H * opt.W)  # B x HW
    #     visib_num_min = torch.sum(visib_mask, dim=1).min().data  # B --> 1
    #
    #     rays_per_img = opt.nerf.rand_rays // batch_size
    #     rays_on_obj = min(int(rays_per_img * opt.nerf.ray_obj_ratio),
    #                       visib_num_min)
    #     rays_on_bg = rays_per_img - rays_on_obj
    #     ray_indices_obj = []
    #     assert rays_on_obj <= visib_num_min
    #
    #     for bi in range(batch_size):
    #         roi_region = var.obj_mask[bi]
    #         # Acquire the index of the pixels on the objects
    #         roi_idx = (roi_region.view(opt.H * opt.W) > 0).nonzero(as_tuple=True)[0]
    #         roi_sample = torch.randperm(len(roi_idx))[:rays_on_obj]  # 1 / 5 number of the sampled data
    #         ray_idx_obj = roi_idx[roi_sample]  # N_roi
    #         ray_indices_obj.append(ray_idx_obj)
    #     var.ray_idx_obj = torch.stack(ray_indices_obj, 0)  # B x N_obj
    #     # Sample rays for photometric loss
    #     var.ray_idx_bg = torch.randperm(opt.H * opt.W, device=opt.device)[:rays_on_bg].expand(batch_size, rays_on_bg)
    #     # Put all rays together, render all these rays in one-go
    #     var.ray_idx = torch.cat([var.ray_idx_obj, var.ray_idx_bg], dim=1)  # B x N_ray_per_img
    #     return var

    @staticmethod
    def get_ray_idx(opt, var):
        batch_size = len(var.idx)
        rays_on_bg = opt.nerf.rand_rays // batch_size
        var.ray_idx = torch.randperm(opt.H * opt.W, device=opt.device)[:rays_on_bg].repeat(batch_size, 1)
        # Put all rays together, render all these rays in one-go
        return var

    @staticmethod
    def get_pose(opt, var, mode=None):
        pose_source = dict(gt=var.pose, predicted=var.pose_init)
        if mode == 'train':
            return pose_source[opt.data.pose_source]
        else:
            return pose_source['gt']

    def forward(self, opt, var, mode=None):
        pose = self.get_pose(opt, var, mode=mode)
        depth_min, depth_max = var.z_near, var.z_far
        depth_range = (depth_min[:, :, None], depth_max[:, :, None])  # (B, HW, 1)

        # Do rendering
        if opt.nerf.rand_rays and mode in ["train", "test-optim"]:
            var = self.get_ray_idx(opt, var)
            ret = self.render(opt,
                              pose,
                              intr=var.intr,
                              ray_idx=var.ray_idx,
                              depth_range=depth_range,
                              mode=mode)  # [B,HW,3],[B,HW,1]

        else:
            object_mask = var.obj_mask
            ret = self.render_by_slices(opt,
                                        pose,
                                        intr=var.intr,
                                        depth_range=depth_range,
                                        object_mask=object_mask,
                                        mode=mode)
        var.update(ret)
        return var

    def compute_loss(self, opt, var, mode=None):
        loss = edict()
        batch_size = len(var.idx)
        HW = opt.H * opt.W
        image = var.image.view(batch_size, 3, opt.H * opt.W).permute(0, 2, 1)

        if opt.nerf.rand_rays and mode in ["train", "test-optim"]:
            image = image.contiguous().view(batch_size, HW, 3)  # B x HW x 3
            image = self.ray_batch_sample(image, var.ray_idx)  # B x HW x 3

        # Compute mask loss
        if opt.loss_weight.mask is not None:
            gt_mask = var.obj_mask.view(batch_size, opt.H * opt.W, 1).float()  # [B, HW, 1]
            if opt.nerf.rand_rays and mode in ["train", "test-optim"]:
                gt_mask = self.ray_batch_sample(gt_mask, var.ray_idx)
            loss.mask = self.MSE_loss(gt_mask, var.opacity)

        if opt.loss_weight.depth is not None:
            if opt.data.erode_mask_loss:
                mask_obj = var.erode_mask.view(batch_size, opt.H * opt.W, 1)
            else:
                mask_obj = var.obj_mask.view(batch_size, opt.H * opt.W, 1)

            if mode == 'train':
                depth_gt = self.ray_batch_sample(var.depth_gt.view(batch_size, opt.H * opt.W)[..., None],
                                                 var.ray_idx)
                mask_obj = self.ray_batch_sample(mask_obj, var.ray_idx)
            else:
                depth_gt = var.depth_gt.view(batch_size, opt.H * opt.W)[..., None]
                mask_obj = mask_obj

            depth_pred = var.depth
            loss.depth = self.scale_invariant_depth_loss(depth_pred, depth_gt, mask_obj)

        if opt.loss_weight.render is not None:
            if opt.nerf.mask_obj:
                if opt.data.erode_mask_loss:
                    mask_obj = var.erode_mask.view(batch_size, opt.H * opt.W, 1)
                else:
                    mask_obj = var.obj_mask.view(batch_size, opt.H * opt.W, 1)
                if mode == 'train':
                    ray_mask_sample = self.ray_batch_sample(mask_obj, var.ray_idx)  # B x HW x 1
                else:
                    ray_mask_sample = mask_obj
                loss.render = (ray_mask_sample * (image - var.rgb) ** 2).sum() / \
                              (ray_mask_sample.sum() + 1e-5)
            else:
                loss.render = self.MSE_loss(var.rgb, image)

        return loss

    def render(self,
               opt,
               pose,
               intr=None,
               ray_idx=None,
               depth_range=None,
               mode=None):
        batch_size = len(pose)
        center, ray = camera.get_center_and_ray(opt,
                                                pose,
                                                intr=intr)  # [B,HW,3], center and ray measured in world frame
        while ray.isnan().any():
            center, ray = camera.get_center_and_ray(opt, pose, intr=intr)  # [B,HW,3]

        # ray center and direction indexing
        center = self.ray_batch_sample(center, ray_idx)
        ray = self.ray_batch_sample(ray, ray_idx)

        # Depth range indexing
        depth_min, depth_max = depth_range[0], depth_range[1]
        depth_min_sample = self.ray_batch_sample(depth_min, ray_idx).squeeze(-1)
        depth_max_sample = self.ray_batch_sample(depth_max, ray_idx).squeeze(-1)
        depth_range = (depth_min_sample, depth_max_sample)

        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center, ray = camera.convert_NDC(opt, center, ray, intr=intr)

        # render with main MLP
        depth_samples = self.sample_depth(opt,
                                          batch_size,
                                          depth_range,
                                          num_rays=ray.shape[1])  # [B,HW,N,1]

        rgb_samples, density_samples = self.nerf.forward_samples(opt, center, ray, depth_samples, mode=mode)

        rgb, depth, opacity, prob = self.nerf.composite(opt, ray, rgb_samples, density_samples, depth_samples)
        ret = edict(rgb=rgb, depth=depth, opacity=opacity)  # [B,HW,K]

        return ret

    def render_by_slices(self,
                         opt,
                         pose,
                         intr=None,
                         depth_range=None,
                         object_mask=None,
                         mode=None):
        ret_all = edict(rgb=[], depth=[], opacity=[])
        if opt.nerf.fine_sampling:
            ret_all.update(rgb_fine=[], depth_fine=[], opacity_fine=[])

        # Acquire the index of the pixels on the objects
        ray_idx_obj = (object_mask.view(opt.H * opt.W) > 0).nonzero(as_tuple=True)[0]
        for c in range(0, opt.H * opt.W, opt.nerf.rand_rays):
            # print(ray_idx_obj[0], ray_idx_obj[-1])
            ray_idx = torch.arange(c, min(c + opt.nerf.rand_rays, opt.H * opt.W), device=opt.device)
            # if ray_idx[-1] <= ray_idx_obj[0] or ray_idx[0] > ray_idx_obj[-1]:
            #     # print(c)
            #     ret = edict(rgb=torch.zeros(1, len(ray_idx), 3).cuda(),
            #                 depth=torch.zeros(1, len(ray_idx), 1).cuda(),
            #                 opacity=torch.zeros(1, len(ray_idx), 1).cuda())  # [B,HW,K]
            # else:
            #     # print(c)
            ret = self.render(opt,
                              pose,
                              intr=intr,
                              ray_idx=ray_idx[None],
                              depth_range=depth_range,
                              mode=mode)  # [B,R,3],[B,R,1]
            for k in ret:
                ret_all[k].append(ret[k])
        # group all slices of images
        for k in ret_all:
            ret_all[k] = torch.cat(ret_all[k], dim=1)
        return ret_all

    # def render_by_slices(self, opt, pose, intr=None, depth_range=None, object_mask=None, mode=None):
    #     ret_all = edict(rgb=[],
    #                     opacity=[],
    #                     depth=[], )
    #
    #     if mode == 'val':
    #         for c in range(0, opt.H * opt.W, opt.nerf.rand_rays):
    #             ray_idx = torch.arange(c, min(c + opt.nerf.rand_rays, opt.H * opt.W), device=opt.device)[None]
    #             ret = self.render(opt,
    #                               pose,
    #                               intr=intr,
    #                               ray_idx=ray_idx,
    #                               depth_range=depth_range,
    #                               mode=mode)  # [B,R,3],[B,R,1]
    #             for k in ret: ret_all[k].append(ret[k])
    #         # group all slices of images
    #         for k in ret_all: ret_all[k] = torch.cat(ret_all[k], dim=1)
    #
    #     else:
    #         # accelerate rendering with mask prior
    #         # Acquire the index of the pixels on the objects
    #         ray_idx_obj = (object_mask.view(opt.H * opt.W) > 0).nonzero(as_tuple=True)[0]
    #         N_rays_obj = len(ray_idx_obj)
    #         for k in ret_all:
    #             if 'rgb' in k:
    #                 ret_all[k] = torch.zeros(1, opt.H * opt.W, 3).cuda()
    #             else:
    #                 ret_all[k] = torch.zeros(1, opt.H * opt.W, 1).cuda()
    #
    #         for c in range(0, N_rays_obj, opt.nerf.rand_rays):
    #             ray_idx = ray_idx_obj[c:min(c + opt.nerf.rand_rays, len(ray_idx_obj))][None]
    #             ret = self.render(opt,
    #                               pose,
    #                               intr=intr,
    #                               ray_idx=ray_idx,
    #                               depth_range=depth_range,
    #                               mode=mode)  # [B,R,3],[B,R,1]
    #             for k in ret:
    #                 ret_all[k][:, ray_idx] = ret[k][0]
    #     return ret_all
    #

    @staticmethod
    def sample_depth(opt, batch_size, depth_range, num_rays=None):
        depth_min, depth_max = depth_range
        depth_min = depth_min[:, :, None, None]
        depth_max = depth_max[:, :, None, None]
        num_rays = num_rays or opt.H * opt.W

        # [B,HW,N,1]
        rand_samples = torch.rand(batch_size, num_rays, opt.nerf.sample_intvs, 1,
                                  device=opt.device) if opt.nerf.sample_stratified else 0.5

        # [B,HW,N,1], Stratified sampling
        rand_samples += torch.arange(opt.nerf.sample_intvs, device=opt.device)[None, None, :, None].float()

        # Linearly interpolation
        depth_samples = rand_samples / opt.nerf.sample_intvs * (depth_max - depth_min) + depth_min  # [B,HW,N,1]

        depth_samples = dict(
            metric=depth_samples,
            inverse=1 / (depth_samples + 1e-8),
        )[opt.nerf.depth.param]
        return depth_samples
