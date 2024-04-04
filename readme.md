# [CVPR 2024] Relightable and Animatable Neural Avatar from Sparse-View Video

[Paper](https://arxiv.org/abs/2308.07903) | [Project Page](https://zju3dv.github.io/relightable_avatar) | [Video](https://youtu.be/BQ3pL7Uwbdk)



![teaser_video](https://github.com/dendenxu/relightable_avatar/assets/43734697/874231a3-3366-4d2f-a081-05ccf05e4096)

This paper tackles the challenge of creating relightable and animatable neural avatars from sparse-view (or even monocular) videos of dynamic humans under unknown illumination.
Compared to studio environments, this setting is more practical and accessible but poses an extremely challenging ill-posed problem.


## Quick Start

### Prepare Trained Model

We provide an example trained model for the *xuzhen* sequence of the MobileStage dataset:
- The base AniSDF model can be downloaded here: [latest.zip](https://github.com/dendenxu/relightable_avatar/files/13040304/latest.zip).
- The relightable model can be downloaded here: [latest.zip](https://github.com/dendenxu/relightable_avatar/files/13040308/latest.zip).
- Furthermore, you'll need to download a skeleton dataset (very small, only with some basic information needed to run `relightable_avatar`) here: [xuzhen_skeleton.tar.gz](https://github.com/dendenxu/relightable_avatar/files/13040284/xuzhen_skeleton.tar.gz).
  - The skeleton dataset is only required if the full dataset hasn't been downloaded and placed at its corresponding location.
- For relighting, we also provide the downscaled environment map: [16x32.zip](https://github.com/dendenxu/relightable_avatar/files/13050868/16x32.zip). If you see errors about `data/lighting`, download this.

Trained model and skeleton data placement:
- The base AniSDF model should be put in `data/trained_model/deform/xuzhen_12v_geo`, after which we expect `latest.pth` to reside at `data/trained_model/deform/xuzhen_12v_geo/latest.pth`.
- The relightable model should be put in `data/trained_model/relight/xuzhen_12v_geo_fix_mat`, after which we expect `latest.pth` to reside at `data/trained_model/deform/xuzhen_12v_geo_fix_mat/latest.pth`.
- The skeleton dataset should be extracted at `data/mobile_stage/xuzhen`, leading to `data/mobile_stage/xuzhen/motion.npz...`.
- The environment map should be placed at `data/lighting`, after which a `data/lighting/16x32` folder is expected.

### Prepare Custom Pose

For the human pose, we use a compact `motion.npz` to store the pose, shape and global translation parameters.
You can find an example file at `data/mobile_stage/xuzhen/motion.npz`.
If you've downloaded the skeleton data provided above, you should also see other motion files ending with `.npz`.

We also provide a script for preparing other common motion formats into our `motion.npz` structure at `scripts/toosl/prepare_motion.py`.
You can learn more about the structure of `motion.npz` in this script.

### Run the AniSDF Model With Custom Pose

```shell
# Fixed view + novel pose
python run.py -t visualize -c configs/mobile_stage/xuzhen_12v_geo.yaml ground_attach_envmap False vis_pose_sequence True num_eval_frame 100 H 512 W 512 novel_view_ixt_ratio 0.80 vis_ext .png test_view 0, test_motion gPO_sFM_cAll_d12_mPO1_ch16.npz

# Novel rotating view + novel pose
python run.py -t visualize -c configs/mobile_stage/xuzhen_12v_geo.yaml ground_attach_envmap False vis_novel_view True perform True num_render_view 100 H 512 W 512 novel_view_ixt_ratio 0.80 vis_ext .png test_motion gPO_sFM_cAll_d12_mPO1_ch16.npz

# For faster rendering, use sphere tracing instead of volume rendering by adding the `vis_sphere_tracing True` entry
# Will speed up the rendering, but might produce artifacts
```

Try to tune these entries `H 512 W 512 novel_view_ixt_ratio 0.80` to customize your output image.
Moreover, select the source view using `test_view 0,` and the motion using `test_motion gPO_sFM_cAll_d12_mPO1_ch16.npz`.
`num_eval_frame` and `num_render_view` control the number of rendered images for the novel pose and novel view setting, respectively.

Example motions files are provided at `data/mobile_stage/xuzhen/*.npz`.
To use skeleton data, customize your dataset root using `test_dataset.data_root <CUSTOM_DATASET_PATH>`.
The recommended way of switching to another set of motions is to put the prepared motion file into `<CUSTOM_DATASET_PATH>` (wherever the `test_dataset.data_root` points to) and set `test_motion`.
You can also use `test_motion` to specify a motion file outside the dataset root by providing an absolute path to the motion file.

### Run the Relightable Model With Custom Pose

```shell
python run.py -t visualize -c configs/mobile_stage/xuzhen_12v_geo.yaml relighting True vis_novel_light True vis_pose_sequence True vis_rendering_map True vis_shading_map True vis_albedo_map True vis_normal_map True vis_envmap_map True vis_roughness_map True vis_specular_map True vis_surface_map True vis_residual_map True vis_depth_map True num_eval_frame 100 H 512 W 512 novel_view_ixt_ratio 0.80 vis_ext .png vis_ground_shading True test_light '["main", "venetian_crossroads", "pink_sunrise", "shanghai_bund", "venice_sunrise", "quattro_canti", "olat0002-0027", "olat0004-0019"]' test_view 0, extra_prefix "gPO_sFM_cAll_d12_mPO1_ch16" test_motion gPO_sFM_cAll_d12_mPO1_ch16.npz
```

## Todo

- [ ] Add documentation on training on the SyntheticHuman++ dataset
- [ ] Add documentation on training on the MobileStage dataset

## Citation

If you find this code useful for your research, please cite us using the following BibTeX entry.

```bibtex
@inproceedings{xu2024relightable,
    title={Relightable and Animatable Neural Avatar from Sparse-View Video},
    author={Xu, Zhen and Peng, Sida and Geng, Chen and Mou, Linzhan and Yan, Zihan and Sun, Jiaming and Bao, Hujun and Zhou, Xiaowei},
    booktitle={CVPR},
    year={2024}
}
```
