from __future__ import absolute_import, division, print_function
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
import torch
import camera
import json
import cv2

# datastructures
from pytorch3d.structures import Meshes
from pytorch3d.io import load_ply
from pytorch3d.renderer import TexturesVertex

import data.cad_model
# import tools.mvrenderer
from tools.mvrenderer import Pose
from tools.mvrenderer import MVRenderer
import open3d as o3d
import torch.nn.functional as F
from util import readlines
from tqdm import tqdm

LM_ID2NAME = {
    1: "ape", 2: "benchvise", 3: "bowl", 4: "camera", 5: "can", 6: "cat",
    7: "cup", 8: "driller", 9: "duck", 10: "eggbox", 11: "glue", 12: "holepuncher",
    13: "iron", 14: "lamp", 15: "phone"}


def compose_Rt(rot_gt, tra_gt):
    gt_pose = np.eye(4)
    gt_pose[:3, :3] = rot_gt
    gt_pose[:3, 3] = tra_gt
    gt_pose = torch.from_numpy(gt_pose)[None].float()

    return gt_pose


def get_center_and_ray(pose, intr=None, H=480, W=640):  # [HW,2]
    # given the intrinsic/extrinsic matrices, get the camera center and ray directions]
    with torch.no_grad():
        # compute image coordinate grid
        y_range = torch.arange(H, dtype=torch.float32).add_(0.5)
        x_range = torch.arange(W, dtype=torch.float32).add_(0.5)
        Y, X = torch.meshgrid(y_range, x_range)  # [H,W]
        xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]

    # compute center and ray
    batch_size = len(pose)
    xy_grid = xy_grid.repeat(batch_size, 1, 1)  # [B,HW,2]
    grid_3D = camera.img2cam(camera.to_hom(xy_grid), intr)  # [B,HW,3]
    center_3D = torch.zeros_like(grid_3D)  # [B,HW,3]

    # transform from camera to world coordinates
    grid_3D = camera.cam2world(grid_3D, pose)  # [B,HW,3]
    center_3D = camera.cam2world(center_3D, pose)  # [B,HW,3]
    ray = grid_3D - center_3D  # [B,HW,3]
    return center_3D, ray

def enlarge_diagonal(v_min, v_max, alpha=0.25):
    direction = v_max - v_min
    v_max_n = v_max + direction * alpha / 2
    v_min_n = v_min - direction * alpha / 2
    return v_min_n, v_max_n


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


def parse_options():
    parser = argparse.ArgumentParser(description='LM ADD Evaluation.')

    parser.add_argument("--data_path",
                        type=str,
                        help="path to the training data",
                        default=os.path.join("dataset"))
    parser.add_argument("--height",
                        type=int,
                        help="input image height",
                        default=480)
    parser.add_argument("--width",
                        type=int,
                        help="input image width",
                        default=640)
    parser.add_argument("--res",
                        type=int,
                        help="network input size",
                        default=128)
    parser.add_argument("--object_id",
                        type=int,
                        help="which object to use",
                        default=1)
    parser.add_argument("--dataset",
                        type=str,
                        help="which training split to use",
                        default="lm")
    parser.add_argument("--target_folder",
                        type=str,
                        default='temp')
    parser.add_argument("--pred_loop",
                        type=str,
                        default='dummy')
    parser.add_argument("--generate_pred",
                        action="store_true")
    parser.add_argument("--save_box",
                        action="store_true")
    parser.add_argument("--save_predbox",
                        action="store_true")
    parser.add_argument("--split_name",
                        default='scene_all/train')
    parser.add_argument("--multi_obj",
                        action="store_true")
    parser.add_argument("--verbose",
                        action="store_true")
    return parser.parse_args()


def evaluate(opt):
    # Set the cuda device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Initialize CAD Model
    if opt.dataset == 'lm':
        object_name = LM_ID2NAME[opt.object_id]
    else:
        object_name = str(opt.object_id)

    model = data.cad_model.CAD_Model()
    model_eval = data.cad_model.CAD_Model()
    model_dir = os.path.join(opt.data_path, opt.dataset, opt.dataset + '_models')
    model.load(os.path.join(model_dir, 'models', 'obj_{}.ply'.format(str(opt.object_id).zfill(6))))
    model_eval.load(os.path.join(model_dir, 'models_eval', 'obj_{}.ply'.format(str(opt.object_id).zfill(6))))

    # Initialize renderer
    ply_fn = os.path.join(model_dir, 'models_eval', 'obj_{}.ply'.format(str(opt.object_id).zfill(6)))
    verts, faces = load_ply(ply_fn)
    textured_mesh = o3d.io.read_triangle_mesh(ply_fn)
    color = torch.from_numpy(np.asarray(textured_mesh.vertex_colors).astype(np.float32))[None]
    textures = TexturesVertex(verts_features=color)

    cad_mesh = Meshes(verts=verts[None], faces=faces[None])
    cam = torch.Tensor([572.4114, 0.0, 325.2611,
                        0.0, 573.57043, 242.04899,
                        0.0, 0.0, 1.0]).reshape(3, 3).to(torch.float32).cuda()
    mv_renderer = MVRenderer(cad_mesh, opt.res, opt.res, 1, cam, mode='complex')

    # Acquire the CAD diameter
    with open(os.path.join(model_dir, 'models_eval', 'models_info.json')) as f_info:
        model_info = json.load(f_info)
        f_info.close()
    model_diameter = float(model_info[str(opt.object_id)]['diameter'])

    # Initialize dataset and dataloader
    data_path = os.path.join(opt.data_path, opt.dataset)
    if 'train_syn2real' in opt.split_name:
        data_path = '../self6dpp/datasets/BOP_DATASETS'

    split_path = os.path.join("splits", opt.dataset, object_name, "{}.txt".format(opt.split_name))
    samples = readlines(split_path)

    # Load GT pose, camera intrinsics and meta information
    line = samples[0].split(' ')
    model_name, folder = line[0], line[1]
    scene_obj_path = os.path.join(data_path, folder, 'scene_object.json')
    scene_gt_path = os.path.join(data_path, folder, 'scene_gt.json')
    scene_pred_path = os.path.join(data_path, folder, 'scene_pred_{}.json'.format(opt.pred_loop))
    scene_cam_path = os.path.join(data_path, folder, 'scene_camera.json')
    scene_info_path = os.path.join(data_path, folder, 'scene_gt_info.json')

    if opt.save_predbox:
        with open(scene_pred_path) as f:
            scene_pred_all = json.load(f)
            f.close()
    else:
        scene_pred_all = None
    print("Loading predicted pose from:", scene_pred_path)
    if opt.multi_obj:
        with open(scene_obj_path) as f:
            scene_obj_all = json.load(f)
            f.close()
    with open(scene_gt_path) as f:
        scene_gt_all = json.load(f)
        f.close()
    with open(scene_cam_path) as f:
        scene_cam_all = json.load(f)
        f.close()
    with open(scene_info_path) as f:
        scene_info_all = json.load(f)
        f.close()
    del f

    # Iterate over all samples
    print("Saving bounding boxes ...")
    for i, sample in enumerate(tqdm(samples)):
        line = sample.split(' ')
        model_name = line[0]
        frame_index = int(line[2])
        scene_pose_source = dict(pred=scene_pred_all, gt=scene_gt_all)
        if opt.multi_obj:
            obj_scene_id = int(scene_obj_all[str(frame_index)][model_name])
        else:
            obj_scene_id = 0

        if opt.save_predbox:
            box_source = 'pred'
        else:
            box_source = 'gt'
        box_enlarge_ratio = 0.25   # fixed
        # Acquire cad model
        aabb_min_init = torch.Tensor(model.bb[0]).unsqueeze(0).unsqueeze(0)  # 1 x 1 x 3
        aabb_max_init = torch.Tensor(model.bb[-1]).unsqueeze(0).unsqueeze(0)  # 1 x 1 x 3

        # bbox to square for some thin objects
        _aabb_min = torch.Tensor(model.bb[4]).unsqueeze(0).unsqueeze(0)  # 1 x 1 x 3
        _aabb_max = torch.Tensor(model.bb[3]).unsqueeze(0).unsqueeze(0)  # 1 x 1 x 3
        _aabb_min2 = torch.Tensor(model.bb[2]).unsqueeze(0).unsqueeze(0)  # 1 x 1 x 3
        _aabb_max2 = torch.Tensor(model.bb[5]).unsqueeze(0).unsqueeze(0)  # 1 x 1 x 3
        _aabb_min3 = torch.Tensor(model.bb[1]).unsqueeze(0).unsqueeze(0)  # 1 x 1 x 3
        _aabb_max3 = torch.Tensor(model.bb[6]).unsqueeze(0).unsqueeze(0)  # 1 x 1 x 3

        scale_factor = 6  # fix ratio 6 for ycb
        aabb_min = aabb_min_init + F.normalize(aabb_min_init - _aabb_min) * model.scale / scale_factor
        aabb_max = aabb_max_init + F.normalize(aabb_max_init - _aabb_max) * model.scale / scale_factor
        aabb_min += F.normalize(aabb_min_init - _aabb_min2) * model.scale / scale_factor
        aabb_max += F.normalize(aabb_max_init - _aabb_max2) * model.scale / scale_factor
        aabb_min += F.normalize(aabb_min_init - _aabb_min3) * model.scale / scale_factor
        aabb_max += F.normalize(aabb_max_init - _aabb_max3) * model.scale / scale_factor
        aabb_min, aabb_max = enlarge_diagonal(v_min=aabb_min, v_max=aabb_max, alpha=box_enlarge_ratio)

        # Acquire the emitted rays measured in current frame
        name = str(line[2]).zfill(6)

        # acquire the pose and intrinsics
        rot = scene_pose_source[box_source][str(frame_index)][obj_scene_id]['cam_R_m2c']
        tra = scene_pose_source[box_source][str(frame_index)][obj_scene_id]['cam_t_m2c']
        cam = np.array(scene_cam_all[str(frame_index)]["cam_K"], dtype=np.float32).reshape(3, 3)
        cam = torch.from_numpy(cam)[None]
        pose = torch.eye(4)[None]
        pose[..., :3, :3] = torch.from_numpy(np.array(rot).reshape(3, 3).astype(np.float32))
        pose[..., :3, 3] = torch.from_numpy(np.array(tra).astype(np.float32))
        pose = pose[:, :3]
        ray_o, ray_d = get_center_and_ray(pose, cam.cpu(), H=opt.height, W=opt.width)

        # Transform the corner points to the world coordinate system
        t_near, t_far, valid = aabb_ray_intersection(aabb_min, aabb_max, ray_o, ray_d)
        t_near = torch.where(valid > 0, t_near, torch.zeros_like(t_near)).view(opt.height, opt.width)
        t_far = torch.where(valid > 0, t_far, torch.zeros_like(t_far)).view(opt.height, opt.width)

        # Save the bound
        box_bound = torch.stack([t_near, t_far], 0).cpu().numpy()  # 2 x H x W
        if opt.multi_obj:
            obj_scene_id = str(int(scene_obj_all[str(frame_index)][model_name])).zfill(6)
            box_save_path = os.path.join(opt.target_folder, 'pred_box_{}'.format(opt.pred_loop),
                                         '{}_{}.npz'.format(name, obj_scene_id))
        else:
            box_save_path = os.path.join(opt.target_folder, 'pred_box_{}'.format(opt.pred_loop),
                                         '{}.npz'.format(name))
        os.makedirs(os.path.join(opt.target_folder, 'pred_box_{}'.format(opt.pred_loop)), exist_ok=True)
        np.savez_compressed(box_save_path, data=box_bound)
        # t_near_vis = ((t_near > 0).float().cpu().numpy() * 255).astype(np.uint8)
        # cv2.imwrite(os.path.join(opt.target_folder, '{}_box/{}.png'.format(box_source, name)), t_near_vis)

        if i < len(samples) - 1:
            del box_bound, t_near, t_far, ray_d, ray_o, valid
        else:
            if opt.verbose:
                print("Visualize generated bounding box on last sample ...")
                # Dummy visualization of the last generated box
                with torch.no_grad():
                    # Generate depth of CAD model
                    batch_size = 1
                    mv_renderer_full = MVRenderer(cad_mesh.cuda(), opt.height, opt.width, batch_size, cam[0].cuda(), mode='simplified')
                    render_pose = Pose.from_Rt(pose[:, :3, :3], pose[:, :3, 3])
                    _, depth = mv_renderer_full(render_pose.cuda(), cam.cuda(), mode='nocs', return_depth=True)

                    # compute center and ray
                    H, W = opt.height, opt.width
                    y_range = torch.arange(H, dtype=torch.float32).add_(0.5)
                    x_range = torch.arange(W, dtype=torch.float32).add_(0.5)
                    Y, X = torch.meshgrid(y_range, x_range)  # [H,W]
                    xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]

                    # Back-projection of CAD model
                    depth = depth.clamp(min=1e-3)
                    depth = depth.view(batch_size, 480 * 640).unsqueeze(-1).cpu() / 1000  # [B,HW,1]
                    xy_grid = xy_grid.repeat(batch_size, 1, 1)  # [B,HW,2]
                    grid_3D = camera.img2cam(camera.to_hom(xy_grid) * depth, cam.cpu())  # [B,HW,3]
                    cad_points = grid_3D.float().cpu().squeeze().float().numpy()

                    # Back-projection of near and far plane and transform them to world coordinate
                    ray_o, ray_d = get_center_and_ray(pose[:, :3], cam.cpu(), H=240 * 2, W=320 * 2)
                    near_points = (ray_o + t_near.view(-1)[None, :, None] * ray_d)
                    points_world_near = camera.world2cam(near_points, pose[:, :3, :])  # [B,HW,3]
                    points_near_np = points_world_near.cpu().numpy().squeeze() / 1000

                    far_points = (ray_o + t_far.view(-1)[None, :, None] * ray_d)
                    points_world_far = camera.world2cam(far_points, pose[:, :3, :])  # [B,HW,3]
                    points_far_np = points_world_far.cpu().numpy().squeeze() / 1000

                    # Painting for visualization
                    pcd_cad = o3d.geometry.PointCloud()
                    pcd_cad.points = o3d.utility.Vector3dVector(np.array(cad_points).astype(np.float32))
                    pcd_cad.paint_uniform_color((0, 0.7, 0))

                    pcd_box_near = o3d.geometry.PointCloud()
                    pcd_box_near.points = o3d.utility.Vector3dVector(points_near_np.astype(np.float32))
                    pcd_box_near.paint_uniform_color((1, 0, 0))

                    pcd_box_far = o3d.geometry.PointCloud()
                    pcd_box_far.points = o3d.utility.Vector3dVector(points_far_np.astype(np.float32))
                    pcd_box_far.paint_uniform_color((0, 0, 1))

                    o3d.visualization.draw_geometries([pcd_cad, pcd_box_far, pcd_box_near])


if __name__ == "__main__":
    options = parse_options()
    evaluate(options)

