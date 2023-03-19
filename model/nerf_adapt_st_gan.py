import numpy as np
import os, sys, time
import torch.nn.functional as torch_F
import torchvision.transforms.functional as torchvision_F
import tqdm
import torch
import lpips
import util, util_vis
import camera

from . import base
from util import log, debug
from layers.nerf_static_transient_light import NeRF
import importlib
from tools.ray_sampler import RaySampler
from tools.patch_sampler import FlexPatchSampler
from layers.perceptual_loss import PerceptualLoss
from layers.lab_loss import LabLoss
from layers.discriminator import Discriminator
from torch import autograd
from external.pohsun_ssim import pytorch_ssim
from easydict import EasyDict as edict


# ============================ main engine for training and evaluation ============================

class Model(base.Model):

    def __init__(self, opt):
        super().__init__(opt)
        self.lpips_loss = lpips.LPIPS(net="alex").to(opt.device)

    def load_dataset(self, opt, eval_split="val"):
        if opt.syn2real:
            dataset_name = opt.data.dataset + 'syn2real'
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

    def build_networks(self, opt):
        super().build_networks(opt)  # include parameters from graph (nerf ...)
        self.graph.latent_vars_trans = torch.nn.Embedding(len(self.train_data), opt.nerf.N_latent_trans).to(opt.device)
        torch.nn.init.normal_(self.graph.latent_vars_trans.weight)
        self.graph.latent_vars_light = torch.nn.Embedding(len(self.train_data), opt.nerf.N_latent_light).to(opt.device)
        torch.nn.init.normal_(self.graph.latent_vars_light.weight)
        # TODO: add anchor poses as the parameter within the graph!

    def setup_optimizer(self, opt):
        log.info("setting up optimizers...")
        optimizer = getattr(torch.optim, opt.optim.algo)

        # set up optimizer for nerf model
        self.optim_nerf = optimizer([dict(params=self.graph.nerf.parameters(), lr=opt.optim.lr)])
        self.optim_nerf.add_param_group(dict(params=self.graph.latent_vars_light.parameters(), lr=opt.optim.lr))
        self.optim_nerf.add_param_group(dict(params=self.graph.latent_vars_trans.parameters(), lr=opt.optim.lr))

        # set up scheduler for nerf model
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched.type)
            if opt.optim.lr_end:
                assert (opt.optim.sched.type == "ExponentialLR")
                if opt.optim.sched.gamma is None:
                    opt.optim.sched.gamma = (opt.optim.lr_end / opt.optim.lr) ** (1. / opt.max_epoch)
                else:
                    opt.optim.sched.gamma = 0.1 ** (1. / 6000)
            print(opt.optim.sched.gamma)
            kwargs = {k: v for k, v in opt.optim.sched.items() if k != "type"}
            self.sched_nerf = scheduler(self.optim_nerf, **kwargs)

        # set up optimizer for discriminator
        if opt.optim_disc.algo is not None and opt.gan is not None:
            optimizer_disc = getattr(torch.optim, opt.optim_disc.algo)
            self.optim_disc = optimizer_disc([dict(params=self.graph.discriminator.parameters(), lr=opt.optim_disc.lr)])

    def restore_pretrained_checkpoint(self, opt):
        epoch_start, iter_start = None, None
        if opt.resume_pretrain:
            log.info("resuming from previous checkpoint...")
            epoch_start, iter_start = util.restore_pretrain_partial_checkpoint(opt, self, resume=opt.resume_pretrain)
        elif opt.resume_real:
            log.info("resuming from previous checkpoint...")
            epoch_start, iter_start = util.restore_pretrain_nerf(opt, self, resume=opt.resume_real)

        else:
            log.info("initializing weights from scratch...")
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0

    @staticmethod
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    def nerf_trainstep(self, opt, var):
        # toggle graidents
        self.toggle_grad(self.graph.nerf, True)
        self.toggle_grad(self.graph.latent_vars_trans, True)
        self.toggle_grad(self.graph.latent_vars_light, True)
        if opt.gan is not None:
            self.toggle_grad(self.graph.discriminator, False)

        # zero out gradients
        self.optim_nerf.zero_grad()

        # forward pass of the nerf (generator)
        var = self.graph.nerf_forward(opt, var, mode='train')
        gloss = self.graph.compute_loss(opt, var, mode="train", train_step='nerf')
        gloss = self.summarize_loss(opt, var, gloss)
        gloss.all.backward()

        # update parameters of nerf
        self.optim_nerf.step()
        return var, gloss

    def disc_trainstep(self, opt, var):
        # toggle graidents
        self.toggle_grad(self.graph.nerf, False)
        self.toggle_grad(self.graph.latent_vars_trans, False)
        self.toggle_grad(self.graph.latent_vars_light, False)
        self.toggle_grad(self.graph.discriminator, True)

        # zero out gradients
        self.optim_disc.zero_grad()

        # forward pass of the discriminator on real data
        var = self.graph.disc_forward(opt, var, mode='train')
        dloss = self.graph.compute_loss(opt, var, mode="train", train_step='disc')
        dloss = self.summarize_loss(opt, var, dloss)

        # backward for each component
        # multiply the weighting parameter
        dloss.gan_disc_real *= (10 ** float(opt.loss_weight['gan_disc_real']))
        dloss.gan_disc_fake *= (10 ** float(opt.loss_weight['gan_disc_fake']))

        # backward for real patch, determine if gradients regularization is needed
        if opt.loss_weight.gan_reg_real is not None:
            dloss.gan_disc_real.backward(retain_graph=True)
            regloss_real = self.graph.compute_grad2(opt, var.d_real_disc, var.patch_real).mean()
            dloss.gan_reg_real = regloss_real
            regloss_real *= (10 ** float(opt.loss_weight['gan_reg_real']))
            regloss_real.backward()
        else:
            dloss.gan_disc_real.backward()

        # backward for fake patch, determine if gradients regularization is needed
        if opt.loss_weight.gan_reg_fake is not None:
            dloss.gan_disc_fake.backward(retain_graph=True)
            regloss_fake = self.graph.compute_grad2(opt, var.d_fake_disc, var.patch_fake).mean()
            dloss.gan_reg_fake = regloss_fake
            regloss_fake *= (10 ** float(opt.loss_weight['gan_reg_fake']))
            regloss_fake.backward()
        else:
            dloss.gan_disc_fake.backward()

        # update params of discriminator
        self.optim_disc.step()
        return var, dloss

    def train_iteration(self, opt, var, loader):
        # before train iteration
        self.timer.it_start = time.time()

        # train iteration
        var = self.graph.get_ray_idx(opt, var)
        var, gloss = self.nerf_trainstep(opt, var)
        if opt.gan is not None:
            var, dloss = self.disc_trainstep(opt, var)
            self.graph.discriminator.progress.data.fill_(self.it / opt.max_iter)
        else:
            dloss = None
        self.graph.patch_sampler.iterations = self.it

        # after train iteration
        if (self.it + 1) % opt.freq.scalar == 0:
            if opt.gan is not None:
                for train_step, loss in zip(['nerf', 'disc'], [gloss, dloss]):
                    self.log_scalars(opt, var, loss, step=self.it + 1, split="train", train_step=train_step)
            else:
                self.log_scalars(opt, var, gloss, step=self.it + 1, split="train", train_step='nerf')

        if (self.it + 1) % opt.freq.vis == 0: self.visualize(opt, var, step=self.it + 1, split="train")
        if (self.it + 1) % opt.freq.val == 0: self.validate(opt, self.it + 1)
        if (self.it + 1) % opt.freq.ckpt == 0: self.save_checkpoint(opt, ep=self.ep, it=self.it + 1)
        self.it += 1
        # loader.set_postfix(it=self.it, loss="{:.3f}".format(loss.all))
        self.timer.it_end = time.time()
        util.update_timer(opt, self.timer, self.ep, len(loader))
        return gloss, dloss

    def train_epoch(self, opt, counter):
        # before train epoch
        self.graph.train()

        loader = self.train_loader
        for batch in loader:
            # train iteration
            var = edict(batch)
            var = util.move_to_device(var, opt.device)
            gloss, dloss = self.train_iteration(opt, var, loader)

        if opt.optim.sched: self.sched_nerf.step()
        counter.set_postfix(ep=self.ep, it=self.it, nerf_loss="{:.3f}".format(gloss.all))

    def train(self, opt):
        if opt.max_epoch is not None:
            opt.max_iter = int(opt.max_epoch * len(self.train_data) // opt.batch_size)
        super().train(opt)

    @torch.no_grad()
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train", train_step='nerf'):
        super().log_scalars(opt, var, loss, metric=metric, step=step, split=split)
        if train_step == 'nerf':
            # log learning rate
            if split == "train":
                lr = self.optim_nerf.param_groups[0]["lr"]
                patch_scale_min, patch_scale_max = self.graph.patch_sampler.scales_curr
                # log scale
                self.tb.add_scalar("{0}/{1}".format(split, "lr_nerf"), lr, step)
                self.tb.add_scalar("{0}/{1}".format(split, "patch_scale_min"), patch_scale_min, step)
                self.tb.add_scalar("{0}/{1}".format(split, "patch_scale_max"), patch_scale_max, step)

            # compute PSNR
            mask = var.obj_mask.view(-1, opt.H * opt.W, 1)
            image = var.image.view(-1, 3, opt.H * opt.W).permute(0, 2, 1)
            if split == 'train':
                psnr = -10 * loss.render.log10()
            else:
                psnr = -10 * self.graph.MSE_loss(var.rgb, image * mask).log10()
            self.tb.add_scalar("{0}/{1}".format(split, "PSNR"), psnr, step)

        elif train_step == 'disc':
            assert split == 'train'
            lr = self.optim_disc.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split, "lr_disc"), lr, step)

    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train", eps=1e-10):
        if opt.tb:
            mask = (var.obj_mask > 0).view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2).float()
            util_vis.tb_image(opt, self.tb, step, split, "image_masked", var.image * mask)
            util_vis.tb_image(opt, self.tb, step, split, "image", var.image)
            util_vis.tb_image(opt, self.tb, step, split, "z_near",
                              var.z_near.float().view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2),
                              from_range=(0.6 * opt.nerf.depth.scale,  var.z_near.max()), cmap='plasma')
            # print(var.depth_gt.min(), var.depth_gt.max())
            if split == 'train':
                rgb_sample = var.rgb.view(-1, opt.patch_size, opt.patch_size, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
                rgb_sample_synmasked = rgb_sample * var.mask_syn_sample
                util_vis.tb_image(opt, self.tb, step, split, "image_sample", var.image_sample)
                util_vis.tb_image(opt, self.tb, step, split, "rgb_sample", rgb_sample)

                if 'image_syn_sample' in var.keys():
                    util_vis.tb_image(opt, self.tb, step, split, "syn_image_sample", var.image_syn_sample)
                    util_vis.tb_image(opt, self.tb, step, split, "rgb_sample_synmasked", rgb_sample_synmasked)
                if 'img_syn_lab' in var.keys() and 'rgb_lab' in var.keys():
                    util_vis.tb_image(opt, self.tb, step, split, "rgb_lab", var.rgb_lab)
                    util_vis.tb_image(opt, self.tb, step, split, "img_syn_lab", var.img_syn_lab)
                if 'patch_fake' in var.keys() and 'patch_real' in var.keys():
                    util_vis.tb_image(opt, self.tb, step, split, "patch_fake_masked", var.patch_fake[:, :3])
                    util_vis.tb_image(opt, self.tb, step, split, "patch_real_masked", var.patch_real[:, :3])
                if opt.gan is not None and opt.gan.geo_conditional:
                    assert 'nocs_sample' in var.keys() and 'normal_sample' in var.keys()
                    _, nocs, normal = var.patch_fake.split(3, dim=1)
                    normal_vis = normal * 0.5 + 0.5
                    util_vis.tb_image(opt, self.tb, step, split, "normal_predicted", normal_vis)
                    util_vis.tb_image(opt, self.tb, step, split, "nocs_predicted", nocs)

            if not opt.nerf.rand_rays or split != "train":
                rgb_map = var.rgb.view(-1, opt.H, opt.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
                depth_map = var.depth.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,3,H,W]
                pred_mask = var.opacity_static.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2).clamp(0, 1)  # [B,1,H,W]
                gt_mask = var.obj_mask.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2).float()  # [B,1,H,W]
                depth_error = (depth_map - var.depth_gt[:, None]).abs().view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)
                rgb_static_map = var.rgb_static.view(-1, opt.H, opt.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
                rgb_transient_map = var.rgb_transient.view(-1, opt.H, opt.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
                uncert_map = var.uncert.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)
                color_error = ((rgb_map - var.image * mask) ** 2).mean(dim=1, keepdim=True).float()
                # if opt.nerf.mask_obj:
                mask = (var.obj_mask > 0).view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2).float()
                depth_map *= mask
                depth_error *= mask
                util_vis.tb_image(opt, self.tb, step, split, "rgb", rgb_map)
                util_vis.tb_image(opt, self.tb, step, split, "rgb_static", rgb_static_map)
                util_vis.tb_image(opt, self.tb, step, split, "rgb_transient", rgb_transient_map)

                util_vis.tb_image(opt, self.tb, step, split, "pred_mask", pred_mask)
                util_vis.tb_image(opt, self.tb, step, split, "gt_mask", gt_mask)
                util_vis.tb_image(opt, self.tb, step, split, "depth", depth_map,
                                  from_range=(0.8 * opt.nerf.depth.scale, 1.1 * opt.nerf.depth.scale), cmap='plasma')
                util_vis.tb_image(opt, self.tb, step, split, "depth_gt", var.depth_gt[:, None],
                                  from_range=(0.8 * opt.nerf.depth.scale, 1.1 * opt.nerf.depth.scale), cmap='plasma')
                util_vis.tb_image(opt, self.tb, step, split, "depth_error", depth_error,
                                  from_range=(0, torch.quantile(depth_error, 0.99)), cmap='turbo')
                util_vis.tb_image(opt, self.tb, step, split, "color_error", color_error,
                                  from_range=(0, torch.quantile(color_error, 0.95)), cmap='turbo')
                util_vis.tb_image(opt, self.tb, step, split, "uncert", uncert_map,
                                  from_range=(uncert_map.min(), torch.quantile(uncert_map, q=0.99)), cmap='viridis')

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
        ckpt_num = 'last' if opt.resume is True else opt.resume
        if opt.render.save_path is not None:
            test_path = opt.render.save_path
        else:
            test_path = "{0}/test_view_{1}".format(opt.output_path, ckpt_num)

        os.makedirs(test_path, exist_ok=True)
        pose_anchor = self.train_data.get_all_camera_poses(opt, source='gt').to(opt.device)
        for i, batch in enumerate(loader):
            with torch.no_grad():
                var = edict(batch)
                var = util.move_to_device(var, opt.device)
                var.pose_anchor = pose_anchor
                var = self.graph.nerf_forward(opt, var, mode="eval_noalign")

                # evaluate view synthesis
                image = var.image
                rgb_map = var.rgb_static.view(-1, opt.H, opt.W, 3).permute(0, 3, 1, 2)  # * var.obj_mask  # [B,3,H,W]
                depth_map = var.depth.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]
                mask_map = var.obj_mask.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]

                if opt.data.image_size != [128, 128]:
                    image = torch_F.interpolate(image, size=[480, 640], mode='bilinear', align_corners=False)
                    rgb_map = torch_F.interpolate(rgb_map, size=[480, 640], mode='bilinear', align_corners=False)
                    depth_map = torch_F.interpolate(depth_map, size=[480, 640], mode='bilinear', align_corners=False)
                    mask_map = torch_F.interpolate(mask_map, size=[480, 640], mode='nearest')
                if opt.data.scene == 'scene_vis':
                    rgb_map = torchvision_F.center_crop(rgb_map, (256, 256))
                    image = torchvision_F.center_crop(image, (256, 256))
                    depth_map = torchvision_F.center_crop(depth_map, (256, 256))
                    mask_map = torchvision_F.center_crop(mask_map, (256, 256))
                    rgb_map = rgb_map * mask_map + torch.ones_like(rgb_map) * (1 - mask_map)
                fused = torch.cat([rgb_map * mask_map, image * mask_map], dim=-1)
                image_masked = image * mask_map
                depth_map /= opt.nerf.depth.scale  # scale the to-save depth to metric in meter
                depth_vis = util_vis.preprocess_vis_image(opt, depth_map, from_range=(0.3, 0.5), cmap="plasma")

                psnr = -10 * self.graph.MSE_loss(rgb_map, image_masked).log10().item()
                ssim = pytorch_ssim.ssim(rgb_map, image_masked).item()
                lpips = self.lpips_loss(rgb_map * 2 - 1, image_masked * 2 - 1).item()
                res.append(edict(psnr=psnr, ssim=ssim, lpips=lpips))
                # dump novel views
                frame_idx = str(var.frame_index.cpu().item()).zfill(6)
                torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/{}.png".format(test_path, frame_idx))
                if opt.data.scene == 'scene_vis':
                    torchvision_F.to_pil_image(image.cpu()[0]).save("{}/syn_{}.png".format(test_path, frame_idx))
                    torchvision_F.to_pil_image(depth_vis.cpu()[0]).save(
                        "{}/depth_vis_{}.png".format(test_path, frame_idx))

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
    def validate(self, opt, ep=None):
        self.graph.eval()
        loss_val = edict()
        loader = tqdm.tqdm(self.test_loader, desc="validating", leave=False)
        for it, batch in enumerate(loader):
            var = edict(batch)
            var = util.move_to_device(var, opt.device)
            var = self.graph.nerf_forward(opt, var, mode='val')
            loss = self.graph.compute_loss(opt, var, mode="val", train_step='nerf')
            loss = self.summarize_loss(opt, var, loss)
            for key in loss:
                loss_val.setdefault(key, 0.)
                loss_val[key] += loss[key] * len(var.idx)
            loader.set_postfix(loss="{:.3f}".format(loss.all))
            if it == 0: self.visualize(opt, var, step=ep, split="val")
        for key in loss_val: loss_val[key] /= len(self.test_data)
        self.log_scalars(opt, var, loss_val, step=ep, split="val")
        log.loss_val(opt, loss_val.all)

    @torch.no_grad()
    def generate_videos_synthesis(self, opt, eps=1e-10):
        raise NotImplementedError


# ============================ computation graph for forward/backprop ============================

class Graph(base.Graph):

    def __init__(self, opt):
        super().__init__(opt)

        # model to be trained
        self.nerf = NeRF(opt)
        if opt.gan is not None:
            self.discriminator = Discriminator(opt)

        # sampling tool
        self.ray_sampler = RaySampler(opt)
        self.patch_sampler = FlexPatchSampler(opt, scale_anneal=0.0002)

        # loss tool
        self.perceptual_loss = PerceptualLoss()
        self.lab_loss = LabLoss()

    def get_ray_idx(self, opt, var):
        coords, scales = self.patch_sampler(nbatch=opt.batch_size, patch_size=opt.patch_size, device=opt.device)
        var.ray_idx = coords
        var.ray_scales = scales
        return var

    @staticmethod
    def get_pose(opt, var, mode=None):
        pose_source = dict(gt=var.pose, predicted=var.pose_init)
        if mode == 'train':
            return pose_source[opt.data.pose_source]
        else:
            return pose_source['gt']

    @staticmethod
    def sample_geometry(opt, var, mode=None):
        batch_size = len(var.idx)
        nocs_pred = var.nocs_pred.contiguous()
        normal_pred = var.normal_pred.contiguous()
        obj_mask = (var.obj_mask > 0).float().contiguous().view(batch_size, 1, opt.H, opt.W)
        mask_syn = (var.mask_syn > 0).float().contiguous().view(batch_size, 1, opt.H, opt.W)

        if mode == 'train':
            batch_size, h, w, _ = var.ray_idx.shape  # [B, h, w, 2]
            obj_mask = torch_F.grid_sample(obj_mask, var.ray_idx, mode='nearest')  # [B, 1, h, w]
            mask_syn = torch_F.grid_sample(mask_syn, var.ray_idx, mode='nearest')  # [B, 1, h, w]
            nocs_pred = torch_F.grid_sample(nocs_pred, var.ray_idx, mode='bilinear', align_corners=True)
            normal_pred = torch_F.grid_sample(normal_pred, var.ray_idx, mode='bilinear', align_corners=True)

        var.nocs_sample = nocs_pred * mask_syn  # [B, 3, h, w]
        var.normal_sample = normal_pred * mask_syn  # [B, 3, h, w]
        return var

    # Forward pass of the network
    def nerf_forward(self, opt, var, mode=None):
        pose = self.get_pose(opt, var, mode=mode)
        depth_min, depth_max = var.z_near, var.z_far
        depth_range = (depth_min[:, :, None], depth_max[:, :, None])  # (B, HW, 1)

        # Do rendering
        if opt.nerf.rand_rays and mode == 'train':
            ret = self.render(opt,
                              pose,
                              intr=var.intr,
                              ray_idx=var.ray_idx,
                              depth_range=depth_range,
                              sample_idx=var.idx,
                              mode=mode)  # [B,HW,3],[B,HW,1]
        elif mode == 'val':
            object_mask = var.obj_mask
            ret = self.render_by_slices(opt,
                                        pose,
                                        intr=var.intr,
                                        depth_range=depth_range,
                                        object_mask=object_mask,
                                        sample_idx=None,
                                        mode=mode)
        else:
            object_mask = var.obj_mask
            R_dist = camera.rotation_distance(var.pose[..., :3, :3], var.pose_anchor[..., :3, :3]).unsqueeze(-1)
            # k can be at most 3!
            # Not sure about the situation for the other object, but for cat k = 1 is better!!
            k = int(opt.render.N_candidate)
            latent_idx_candidate = torch.topk(R_dist, k=k, dim=0, largest=False, sorted=True)[1]
            latent_light_idx = latent_idx_candidate[torch.randperm(len(latent_idx_candidate))[0]][0]
            # print(latent_idx_candidate, var.frame_index.cpu().item())
            ret = self.render_by_slices(opt,
                                        pose,
                                        intr=var.intr,
                                        depth_range=depth_range,
                                        object_mask=object_mask,
                                        sample_idx=latent_light_idx,
                                        mode=mode)
        var.update(ret)

        # forward pass of to discriminator for GAN loss in training mode
        if mode == 'train' and opt.gan is not None:
            if opt.gan.geo_conditional:
                var = self.sample_geometry(opt, var, mode)
            batch_size, h, w, _ = var.ray_idx.shape  # [B, h, w, 2]
            patch_fake = var.rgb.view(batch_size, h, w, 3).permute(0, 3, 1, 2)  # [B, 3, h, w]
            if opt.gan.geo_conditional:
                patch_fake = torch.cat([patch_fake, var.nocs_sample, var.normal_sample], dim=1)
            var.d_fake_nerf = self.discriminator(opt, patch_fake, var.ray_scales)
        return var

    def disc_forward(self, opt, var, mode):
        if mode == 'train':
            batch_size, h, w, _ = var.ray_idx.shape  # [B, h, w, 2]

            image = var.image_sample
            mask = var.mask_sample
            rgb = var.rgb.view(batch_size, h, w, 3).permute(0, 3, 1, 2).contiguous()  # [B, 3, h, w]
            mask_syn = var.mask_syn_sample
            image_syn = var.image_syn_sample
            mask_pad = torch.logical_and(mask_syn == 1, mask == 0).float()

            patch_real = (image * mask + rgb * mask_pad).detach()
            var.patch_real = patch_real
            if opt.gan.geo_conditional:
                var = self.sample_geometry(opt, var, mode)
                var.patch_real = torch.cat([var.patch_real, var.nocs_sample, var.normal_sample], dim=1)
            var.patch_real.requires_grad_()
            var.d_real_disc = self.discriminator(opt, var.patch_real, var.ray_scales)

            # forward pass of fake patch
            # patch_fake = rgb * mask
            patch_fake = rgb
            var.patch_fake = patch_fake.detach()
            if opt.gan.geo_conditional:
                var.patch_fake = torch.cat([var.patch_fake, var.nocs_sample, var.normal_sample], dim=1)
            var.patch_fake.requires_grad_()
            var.d_fake_disc = self.discriminator(opt, var.patch_fake, var.ray_scales)
        else:
            raise Exception('No use of discriminator in val/testing phase of NeRF!')
        return var

    def render(self, opt, pose, intr=None, ray_idx=None, depth_range=None, sample_idx=None, mode=None):
        # Depth range indexing
        # [B,HW,3], center and ray measured in world frame
        if mode == 'train':
            batch_size, h, w, _ = ray_idx.shape  # [B, h, w, 2]
            depth_min, depth_max = depth_range[0], depth_range[1]
            center, ray = self.ray_sampler.get_rays(opt, intrinsics=intr, coords=ray_idx, pose=pose)
            while ray.isnan().any():
                center, ray = self.ray_sampler.get_rays(opt, intrinsics=intr, coords=ray_idx, pose=pose)
            depth_min_sample, depth_max_sample = self.ray_sampler.get_bounds(opt, coords=ray_idx,
                                                                             z_near=depth_min, z_far=depth_max)

            # Reshape the output results
            center = center.view(batch_size, h * w, -1)  # [B, hw, 3]
            ray = ray.view(batch_size, h * w, -1)  # [B, hw, 3]
            depth_min_sample = depth_min_sample.view(batch_size, h * w)  # [B, hw]
            depth_max_sample = depth_max_sample.view(batch_size, h * w)  # [B, hw]

        else:
            batch_size = len(pose)
            # [B,HW,3], center and ray measured in world frame
            center, ray = camera.get_center_and_ray(opt, pose, intr=intr)
            while ray.isnan().any():
                center, ray = camera.get_center_and_ray(opt, pose, intr=intr)  # [B,HW,3]

            # ray center and direction indexing
            center = self.ray_batch_sample(center, ray_idx)
            ray = self.ray_batch_sample(ray, ray_idx)

            # Depth range indexing
            depth_min, depth_max = depth_range[0], depth_range[1]
            depth_min_sample = self.ray_batch_sample(depth_min, ray_idx).squeeze(-1)
            depth_max_sample = self.ray_batch_sample(depth_max, ray_idx).squeeze(-1)

        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center, ray = camera.convert_NDC(opt, center, ray, intr=intr)

        # Acquire depth range
        depth_range = (depth_min_sample, depth_max_sample)
        depth_samples = self.sample_depth(opt, batch_size, depth_range, num_rays=ray.shape[1])  # [B, HW, N, 1]

        # Prepare latent variables
        if mode == 'train':
            latent_variable_trans = self.latent_vars_trans.weight[sample_idx]
            latent_variable_light = self.latent_vars_light.weight[sample_idx]
        elif mode == 'val':
            latent_variable_trans = self.latent_vars_trans.weight[0][None]
            latent_variable_light = self.latent_vars_light.weight[0][None]
        else:
            # Not sure about the situation for the other object, but for cat,
            # this zero latent variable is important!!
            if opt.render.transient == 'zero':
                latent_variable_trans = torch.zeros(batch_size, opt.nerf.N_latent_trans).cuda()
            elif opt.render.transient == 'sample':
                latent_variable_trans = self.latent_vars_trans.weight[sample_idx][None]
            else:
                raise NotImplementedError
            latent_variable_light = self.latent_vars_light.weight[sample_idx][None]
        # Forward samples to get rgb values and density values
        rgb_samples, density_samples, uncert_samples = self.nerf.forward_samples(opt,
                                                                                 center=center,
                                                                                 ray=ray,
                                                                                 depth_samples=depth_samples,
                                                                                 latent_variable_trans=latent_variable_trans,
                                                                                 latent_variable_light=latent_variable_light,
                                                                                 mode=mode)

        # composition
        (rgb, rgb_static, rgb_transient, depth, opacity, opacity_static, opacity_transient,
         prob, uncert, alpha_static, alpha_transient) = self.nerf.composite(
            opt,
            ray,
            rgb_samples,
            density_samples,
            depth_samples,
            uncert_samples)

        # track the results
        ret = edict(rgb=rgb, rgb_static=rgb_static, rgb_transient=rgb_transient,
                    opacity=opacity, opacity_static=opacity_static, opacity_transient=opacity_transient,
                    uncert=uncert, depth=depth, alpha_static=alpha_static, alpha_transient=alpha_transient,
                    density=density_samples)  # [B,HW,K]

        return ret

    def render_by_slices(self, opt, pose, intr=None, depth_range=None, object_mask=None, sample_idx=None, mode=None):
        ret_all = edict(rgb=[], rgb_static=[], rgb_transient=[],
                        opacity=[], opacity_static=[], opacity_transient=[],
                        depth=[], uncert=[], alpha_static=[], alpha_transient=[], density=[])

        if mode == 'val':
            for c in range(0, opt.H * opt.W, opt.nerf.rand_rays):
                ray_idx = torch.arange(c, min(c + opt.nerf.rand_rays, opt.H * opt.W), device=opt.device)[None]
                ret = self.render(opt,
                                  pose,
                                  intr=intr,
                                  ray_idx=ray_idx,
                                  depth_range=depth_range,
                                  sample_idx=sample_idx,
                                  mode=mode)  # [B,R,3],[B,R,1]
                for k in ret: ret_all[k].append(ret[k])
            # group all slices of images
            for k in ret_all: ret_all[k] = torch.cat(ret_all[k], dim=1)

        else:
            # accelerate rendering with mask prior
            # Acquire the index of the pixels on the objects
            ray_idx_obj = (object_mask.view(opt.H * opt.W) > 0).nonzero(as_tuple=True)[0]
            N_rays_obj = len(ray_idx_obj)
            for k in ret_all:
                if k == 'uncert':
                    ret_all[k] = torch.ones(1, opt.H * opt.W, 1).cuda() * opt.nerf.min_uncert
                elif k == 'density':
                    ret_all[k] = torch.ones(1, opt.H * opt.W, opt.nerf.sample_intvs, 2).cuda()
                elif 'rgb' in k:
                    ret_all[k] = torch.zeros(1, opt.H * opt.W, 3).cuda()
                elif 'alpha' in k:
                    ret_all[k] = torch.ones(1, opt.H * opt.W, opt.nerf.sample_intvs).cuda()
                else:
                    ret_all[k] = torch.zeros(1, opt.H * opt.W, 1).cuda()

            for c in range(0, N_rays_obj, opt.nerf.rand_rays):
                ray_idx = ray_idx_obj[c:min(c + opt.nerf.rand_rays, len(ray_idx_obj))][None]
                ret = self.render(opt,
                                  pose,
                                  intr=intr,
                                  ray_idx=ray_idx,
                                  depth_range=depth_range,
                                  sample_idx=sample_idx,
                                  mode=mode)  # [B,R,3],[B,R,1]
                for k in ret:
                    ret_all[k][:, ray_idx] = ret[k][0]
        return ret_all

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

        depth_samples = dict(metric=depth_samples, inverse=1 / (depth_samples + 1e-8), )[opt.nerf.depth.param]
        return depth_samples

    @staticmethod
    def ray_batch_sample(ray_identity, ray_idx):
        assert ray_identity.shape[0] == ray_idx.shape[0]
        B, HW, _ = ray_identity.shape
        B, N_sample = ray_idx.shape
        ray_identity = ray_identity.view(B * HW, -1)
        samples_cumsum = ray_idx + HW * torch.arange(B).cuda().unsqueeze(1)  # No need to reshape
        ray_identity_sample = ray_identity[samples_cumsum].view(B, N_sample, -1)
        return ray_identity_sample

    def compute_loss(self, opt, var, mode=None, train_step='nerf'):
        loss = edict()
        batch_size = len(var.idx)
        rgb = var.rgb
        uncert = var.uncert
        image = var.image.contiguous()
        obj_mask = (var.obj_mask > 0).float().contiguous().view(batch_size, 1, opt.H, opt.W)
        if 'image_syn' in var.keys() and 'mask_syn' in var.keys():
            image_syn = var.image_syn.contiguous()
            mask_syn = (var.mask_syn > 0).float().contiguous().view(batch_size, 1, opt.H, opt.W)
        else:
            image_syn = image
            mask_syn = obj_mask

        if opt.nerf.rand_rays and mode in ["train", "test-optim"]:
            batch_size, h, w, _ = var.ray_idx.shape  # [B, h, w, 2]
            image = torch_F.grid_sample(image, var.ray_idx, mode='bilinear', align_corners=True)  # [B, 3, h, w]
            obj_mask = torch_F.grid_sample(obj_mask, var.ray_idx, mode='nearest')  # [B, 1, h, w]
            mask_syn = torch_F.grid_sample(mask_syn, var.ray_idx, mode='nearest')  # [B, 1, h, w]
            image_syn = torch_F.grid_sample(image_syn, var.ray_idx, mode='bilinear', align_corners=True)
            rgb = rgb.view(batch_size, h, w, 3).permute(0, 3, 1, 2)  # [B, 3, h, w]
            uncert = uncert.view(batch_size, h, w, 1).permute(0, 3, 1, 2)  # [B, 1, h, w]
        else:
            image = image.view(batch_size, 3, opt.H, opt.W)
            obj_mask = obj_mask.view(batch_size, 1, opt.H, opt.W)
            mask_syn = mask_syn.view(batch_size, 1, opt.H, opt.W)
            rgb = rgb.view(batch_size, opt.H, opt.W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
            uncert = uncert.view(batch_size, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B, 1, H, W]
            image_syn = image_syn.view(batch_size, 3, opt.H, opt.W)

        var.image_syn_sample = image_syn
        var.image_sample = image
        var.mask_sample = obj_mask
        var.mask_syn_sample = mask_syn

        if train_step == 'nerf':
            if opt.loss_weight.render is not None:
                if opt.nerf.mask_obj:
                    loss.render = (obj_mask * ((image - rgb) ** 2 / uncert ** 2)).sum() / (obj_mask.sum() + 1e-5)
                    # loss.render = (obj_mask * ((image - rgb) ** 2)).sum() / (obj_mask.sum() + 1e-5)
                else:
                    loss.render = self.MSE_loss(rgb, image)

            # Compute mask loss
            if opt.loss_weight.mask is not None:
                loss.mask = self.MSE_loss(obj_mask, var.opacity[..., None])

            if opt.loss_weight.uncert is not None:
                loss.uncert = 5 + torch.log(var.uncert ** 2).mean() / 2

            if opt.loss_weight.trans_reg is not None:
                loss.trans_reg = var.density[..., -1].mean()

            # experiment with masking for feature loss and lab loss
            if opt.loss_weight.feat is not None:

                mask_pad = torch.logical_and(mask_syn == 1, obj_mask == 0).float()
                loss.feat = self.perceptual_loss(rgb, image * obj_mask + image_syn * mask_pad) + \
                            5 * self.perceptual_loss(rgb * obj_mask + image * (1 - obj_mask), image)

            if opt.loss_weight.lab is not None:
                loss.lab, var.rgb_lab, var.img_syn_lab = self.lab_loss(rgb, image_syn, mask=mask_syn)

            if opt.gan is not None and opt.loss_weight.gan_nerf is not None and mode == 'train':
                loss.gan_nerf = self.compute_gan_loss(opt, d_outs=var.d_fake_nerf, target=1)

        elif train_step == 'disc':
            # regularization for real patch
            dloss_real = self.compute_gan_loss(opt, d_outs=var.d_real_disc, target=1)
            dloss_fake = self.compute_gan_loss(opt, d_outs=var.d_fake_disc, target=0)

            # compute_loss
            if opt.loss_weight.gan_disc_real is not None:
                loss.gan_disc_real = dloss_real

            if opt.loss_weight.gan_disc_fake is not None:
                loss.gan_disc_fake = dloss_fake

        else:
            raise NotImplementedError
        return loss

    @staticmethod
    def compute_grad2(opt, d_outs, x_in):
        d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
        reg = 0
        for d_out in d_outs:
            batch_size = x_in.size(0)
            grad_dout = autograd.grad(
                outputs=d_out.sum(), inputs=x_in,
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad_dout2 = grad_dout.pow(2)
            assert (grad_dout2.size() == x_in.size())
            reg += grad_dout2.view(batch_size, -1).sum(1)
        return reg / len(d_outs)

    @staticmethod
    def compute_gan_loss(opt, d_outs, target):

        d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
        loss = torch.tensor(0.0, device=d_outs[0].device)

        for d_out in d_outs:

            targets = d_out.new_full(size=d_out.size(), fill_value=target)

            if opt.gan.type == 'standard':
                loss += torch_F.binary_cross_entropy_with_logits(d_out, targets)
            elif opt.gan.type == 'wgan':
                loss += (2 * target - 1) * d_out.mean()
            else:
                raise NotImplementedError

        return loss / len(d_outs)

    def wgan_gp_reg(self, opt, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach().requires_grad_()
        d_out = self.discriminator(x_interp, y)
        reg = (self.compute_grad2(opt, d_out, x_interp).sqrt() - center).pow(2).mean()
        return reg
