from __future__ import absolute_import, division, print_function
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch

# datastructures
from pytorch3d.structures import Meshes
from pytorch3d.io import load_ply, load_obj
from pytorch3d.renderer import TexturesVertex, Textures

import data
# import tools.mvrenderer
from tools.mvrenderer import Pose
from tools.mvrenderer import MVRenderer
import open3d as o3d
import torch.nn.functional as F
from tqdm import tqdm
from compute_box import get_center_and_ray
import importlib
from easydict import EasyDict as edict
import util
import options
import struct

LM_ID2NAME = {
    1: "ape", 2: "benchvise", 3: "bowl", 4: "camera", 5: "can", 6: "cat",
    7: "cup", 8: "driller", 9: "duck", 10: "eggbox", 11: "glue", 12: "holepuncher",
    13: "iron", 14: "lamp", 15: "phone"}

LM_ID2NAME = {v: k for k, v in LM_ID2NAME.items()}


def normal_from_depth(pose, depth, intr, h, w, vis=False):
    batch_size = len(depth)
    center3D, ray = get_center_and_ray(pose, intr, h, w)
    depth_flatten = depth.view(batch_size, 1, h * w).permute(0, 2, 1)
    points3D = center3D + ray * depth_flatten  # [B,HW,3]/[B,HW,3]/[N,3]

    points3D = points3D.permute(0, 2, 1).view(batch_size, 3, h, w)
    tu = points3D[:, :, 1:-1, 2:] - points3D[:, :, 1:-1, :-2]
    tv = points3D[:, :, 2:, 1:-1] - points3D[:, :, :-2, 1:-1]

    normal = tu.cross(tv, dim=1)
    normal = torch.cat([torch.zeros(batch_size, 3, 1, w - 2), normal, torch.zeros(batch_size, 3, 1, w - 2)], dim=-2)
    normal = torch.cat([torch.zeros(batch_size, 3, h, 1), normal, torch.zeros(batch_size, 3, h, 1)], dim=-1)
    normal = F.normalize(normal, dim=1)
    normal[:, -1] *= -1
    if vis:
        normal = normal * 0.5 + 0.5
    normal *= (depth[:, None] > 0).cpu().float()
    return normal


# TODO: make sure we choose the right 2d Box (HW format, and 2D box source)

def compute_surfelinfo(opt):
    assert opt.batch_size == 1 and opt.data.pose_source == 'predicted'

    # Set the cuda device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Acquire name and ID
    if opt.data.dataset == 'lm':
        object_id = LM_ID2NAME[opt.data.object]
    else:
        object_id = opt.data.object

    # Initialize CAD Model
    cad_model = data.cad_model.CAD_Model()
    cad_model_eval = data.cad_model.CAD_Model()
    cad_model_dir = os.path.join(opt.data.root, opt.data.dataset, opt.data.dataset + '_models')
    cad_model.load(os.path.join(cad_model_dir, 'models', 'obj_{:06d}.ply'.format(object_id)))
    cad_model_eval.load(os.path.join(cad_model_dir, 'models_eval', 'obj_{:06d}.ply'.format(object_id)))

    # Initialize renderer
    ply_fn = os.path.join(cad_model_dir, 'models', 'obj_{}.ply'.format(str(object_id).zfill(6)))
    verts, faces = load_ply(ply_fn)
    textured_mesh = o3d.io.read_triangle_mesh(ply_fn)
    color = torch.from_numpy(np.asarray(textured_mesh.vertex_colors).astype(np.float32))[None]
    textures = TexturesVertex(verts_features=color)
    cad_mesh = Meshes(verts=verts[None].to(device), faces=faces[None].to(device), textures=textures.to(device))
    mv_renderer = MVRenderer(cad_mesh, opt.H, opt.W, 1, None, mode='complex')

    # Initialize dataset and dataloader
    model = importlib.import_module("model.{}".format(opt.model))

    m = model.Model(opt)
    m.load_dataset(opt, eval_split="train")
    # Iterate over all samples
    print("Rendering Surfel Geometric Information...")
    with torch.no_grad():
        loader = tqdm(m.train_loader, desc="Generating Surfel Information...", leave=False)
        for it, batch in enumerate(loader):
            var = edict(batch)
            var = util.move_to_device(var, opt.device)
            frame_idx = var.frame_index.cpu().item()

            pose, cam_K = var.pose_init, var.intr
            pose[:, :3, 3] = pose[:, :3, 3] * 1000 / opt.nerf.depth.scale  # calib scale to be mm
            h, w = opt.data.image_size
            render_pose = Pose.from_Rt(pose[:, :3, :3], pose[:, :3, 3])
            if frame_idx == 29:
                print(pose, cam_K)
            # print((var.pose == var.pose_init).all())
            # render nocs, synthetic rgb and compute normal
            rgb_syn, depth = mv_renderer(render_pose, cam_K.cuda(), mode='color', return_depth=True)
            nocs_pred, _ = mv_renderer(render_pose, cam_K.cuda(), mode='nocs', return_depth=True)
            normal_pred = normal_from_depth(pose.cpu(), depth.cpu(), cam_K.cpu(), h=opt.H, w=opt.W)

            # save nocs, synthetic rgb and normal
            alpha_syn = (depth[0] > 0).cpu().numpy()[..., None]
            rgb_syn = rgb_syn[0].cpu().permute(1, 2, 0).numpy()[..., [2, 1, 0]]
            nocs_pred = nocs_pred[0].cpu().permute(1, 2, 0).numpy()[..., [2, 1, 0]]
            normal_pred = normal_pred[0].cpu().permute(1, 2, 0).numpy()
            rgba_syn = np.concatenate([rgb_syn, alpha_syn], axis=-1)

            # save rgb
            pred_loop = opt.data.pose_loop
            os.makedirs(os.path.join(opt.render.geo_save_dir, 'rgbsyn_{}'.format(pred_loop)), exist_ok=True)
            save_path = os.path.join(opt.render.geo_save_dir, 'rgbsyn_{}/{:06d}.png'.format(pred_loop, frame_idx))
            cv2.imwrite(save_path, (rgba_syn * 255).astype(np.uint8))

            # save nocs
            os.makedirs(os.path.join(opt.render.geo_save_dir, 'nocs_{}'.format(pred_loop)), exist_ok=True)
            save_path = os.path.join(opt.render.geo_save_dir, 'nocs_{}/{:06d}.png'.format(pred_loop, frame_idx))
            cv2.imwrite(save_path, (nocs_pred * 255).astype(np.uint8))

            # save normal
            os.makedirs(os.path.join(opt.render.geo_save_dir, 'normal_{}'.format(pred_loop)), exist_ok=True)
            save_path = os.path.join(opt.render.geo_save_dir, 'normal_{}/{:06d}.npz'.format(pred_loop, frame_idx))
            # cv2.imwrite(save_path, ((normal_pred * 0.5 + 0.5) * 255).astype(np.uint8))
            np.savez_compressed(save_path, data=normal_pred.astype(np.float32))


if __name__ == "__main__":
    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    compute_surfelinfo(opt)
