# TexPose
Official repository for "TexPose: Neural Texture Learning for Self-Supervised 6D Object Pose Estimation", CVPR 2023.
  

## TL;DR
We re-formulate self-supervised pose estimation as two sub-optimization problems on texture learning and pose learning. 
With this re-formulation, we can do effective self-training for object pose estimator without any supervision signals like depth or deep pose refiner requried in previous works.

## Prerequisites
1. Set up the environment with cond
```bash
conda env create --file requirements.yaml python=3.8
conda activate texpose
```

2. Download PyTorch3D following the instruction [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

## Data Preparation
We use **Duck** class (ID 9) from LineMOD as an example here:

### Download dataset and predicted pose labels after synthetic pretraining
Download our pre-processed dataset from [here](https://1drv.ms/u/c/687c6713e4f289c2/QcKJ8uQTZ3wggGjvxAAAAAAANAXArr9m0B9-cQ) 

### Download pre-trained weights
Download the pre-trained weights (pre-trained geometric branch and texture learner) from [here](https://1drv.ms/u/c/687c6713e4f289c2/QcKJ8uQTZ3wggGjwxAAAAAAAxvR4Ol4YwlCnGw) 

### Compute box for ray sampling
You can skip this step if you download our pre-processed dataset.
```bash
python3 compute_box.py --pred_loop init_calib --object_id 9 --generate_pred \
--save_predbox --target_folder dataset/lm/lm_test_all/test/000009/
```

### Generate surfel information and synthetic image
You can skip this step if you download our pre-processed dataset.
```bash
python3 compute_surfelinfo.py --name='' \
--model=nerf_adapt_st_gan --yaml=nerf_lm_adapt_gan \
--data.pose_source=predicted --data.pose_loop=init_calib \
--gan= --loss_weight.feat=  \
--batch_size=1 \
--data.object=duck  \
--render.geo_save_dir=dataset/lm/lm_test_all/test/000009/
```

## Running the code
We use **Duck** class (ID 9) from LineMOD as an example here:

### Synthesize novel views
```bash
python3 evaluate.py --model=nerf_adapt_st_gan --yaml=nerf_lm_adapt_gan --batch_size=1  
--data.preload=false --data.object=duck --data.scene=scene_syn2real_layer --name=test_run3 \
--data.image_size=[480,640] --resume --syn2real --render.save_path=PATH/YOU/WANT/TO/SAVE/
```

### Train texture learner
```bash
python3 train.py --model=nerf_adapt_st_gan --yaml=nerf_lm_adapt_gan --resume_pretrain \
--data.pose_source=predicted --data.preload=true  --group Duck --data.object=duck \
--data.scene=scene_all --name=test_run3
```

### Supervise pose estimator
Now you can use the synthesized data to supervise the pose estimator. All synthesized data is organized in BOP format for ease of adapting to variant pose estimators. We use improved GDR-Net from [Self6D++](https://github.com/THU-DA-6D-Pose-Group/self6dpp).

## Citation

If you find our work useful, please consider citing us:
```bibtex
@inproceedings{chen2023texpose,
title = {TexPose: Neural Texture Learning for Self-Supervised 6D Object Pose Estimation},
author = {Hanzhi Chen and
            Fabian Manhardt and
            Nassir Navab and
            Benjamin Busam},
journal={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2023}
}
```

## Acknowledgement
Our implementation is based on [BARF](https://github.com/chenhsuanlin/bundle-adjusting-NeRF) and follows their code structure. Thanks for their great contribution!


