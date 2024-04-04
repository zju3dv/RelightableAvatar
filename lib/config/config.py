from enum import Enum, auto
import os
import sys
import argparse
import warnings
import numpy as np
from . import yacs
from .yacs import CN
from rich import pretty
from rich import traceback
from os.path import join
from pdbr import RichPdb

from easymocap.config.baseconfig import load_object, Config
from easymocap.bodymodel.smplx import SMPLHModel, SMPLModel

from lib.utils.log_utils import log
pretty.install()
traceback.install()
warnings.filterwarnings("ignore")


def set_trace(*args, **kwargs):
    from lib.utils.log_utils import update_log_stats
    if hasattr(update_log_stats, 'live'):
        update_log_stats.live.stop()
    rich_pdb = RichPdb()
    rich_pdb.set_trace(sys._getframe(1))


os.environ["PYTHONBREAKPOINT"] = "lib.config.config.set_trace"


cfg = CN()
cfg.check_bound_sdf = False
cfg.check_termination_sdf = False
cfg.bruteforce_st = False
cfg.smpl_distance = False
cfg.H = -1
cfg.W = -1
cfg.normalize_shading = False
cfg.normalize_specular = True
cfg.vis_lvis_map = False
cfg.vis_ldot_map = False
cfg.ground_shading_multiplier = 1.0
cfg.min_clip = 1.0 # should be very large to view the full spectrum
cfg.novel_view_ixt_ratio = 1.0
cfg.lambert_only = False # ???
cfg.glossy_only = False # ???
cfg.light_xyz_noise_std = 1.0
cfg.shadow_dist_th = 0.05 # faster?
cfg.use_geometry = False  # whether to use the learned geometry as base instead of SMPL

cfg.ablate_hdq = False
cfg.ablate_hdq_mode = 'hdq'  # world, can, curve, hdq
cfg.shade_max = 4.0
# this should be enabled when the training and test motions are sampled from different sources
cfg.fix_material = -1  # use material of first frame for novel pose?

# a good configuration should like: no extra configuration for unused module
# multi models use different option sets
cfg.relighting = False
cfg.no_claybook = False
cfg.no_visibility = False
cfg.light_multiplier = 1.0

# cachable dataset will use this option
cfg.dilation_bias = 0.0025
cfg.dilation_multiplier = 0.5
cfg.randperm_pass = 2
cfg.clip_grad_norm = 40.0
cfg.clip_grad_value = 40.0
cfg.no_data_cache = False

# surface guided volume sampling section
cfg.surf_sample_range = 0.005  # in-out 5mm for 3 point volume rendering

cfg.fps = 30
cfg.clip_near = 0.02
cfg.clip_far = 10.0
cfg.box_far = 5.0
cfg.lambertian = False
cfg.achro_light = False
cfg.envmap_upscale = 2
cfg.find_unused_parameters = False

cfg.geometry_mesh = ''
cfg.geometry_pretrain = ''
cfg.fresnel_f0 = 0.02  # frenel term in BRDF
cfg.xyz_noise_std = 0.02
cfg.dd_xyz_backup = 0.05  # backup along view direction for this distance
cfg.dd_xyz_noise_std = 0.10  # sample along this range
cfg.dd_view_noise_std = 0.25  # smaller ratio of invalid samples
cfg.dd_mult = 10  # four times more jittering to train directional distance module

cfg.olats = [0, 27, 91, 149, 200, 288, 333, 398, 488,
             2 * 32 + 0, 4 * 32 + 7,
             4 * 32 + 13, 4 * 32 + 15, 4 * 32 + 17, 4 * 32 + 19, 4 * 32 + 21, 4 * 32 + 23, 4 * 32 + 25, 4 * 32 + 27,
             2 * 32 + 13, 2 * 32 + 15, 2 * 32 + 17, 2 * 32 + 19, 2 * 32 + 21, 2 * 32 + 23, 2 * 32 + 25, 2 * 32 + 27]  # OLAT index (w * i + j)
cfg.olat_inten = 100.0  # 100 too much
cfg.ambient_inten = 0.25

cfg.lighting_dir = 'data/lighting'  # HDRI lighting conditions
cfg.ground_normal = [0, 0, 1]
cfg.ground_origin = [0, 0, 0]
cfg.ground_albedo = [0.05, 0.05, 0.05]
cfg.ground_roughness = 0.1  # is this realy roughness? or is it specular

cfg.env_image_h = 6144
cfg.env_image_w = 8192
cfg.env_h = 16
cfg.env_w = 32
cfg.env_r = 10

# find surface points
cfg.sphere_tracing = CN()
cfg.sphere_tracing.iter = 16
cfg.sphere_tracing.tan_i = 1000  # sharpness
cfg.sphere_tracing.relax = 0.0
cfg.sphere_tracing.offset = 0.02  # this determines the resulting density
cfg.sphere_tracing.eps = 1e-8
cfg.sphere_tracing.near_offset = 0.01  # march away (this thin structure will be ignored)
cfg.sphere_tracing.shadow_skip_iter = 1  # the first iteration of sphere tracing will not compute shadow
cfg.sphere_tracing.tan_i_multiplier = 1

# self shadow
cfg.obj_lvis = CN()  # 2 and a half minutes to render an 1024 x 1024 image...
cfg.obj_lvis.iter = 4  # smaller number of iteration -> artifacts on soft-shadow
cfg.obj_lvis.offset = 0.01
cfg.obj_lvis.relax = 0.0
cfg.obj_lvis.near_offset = 0.02  # march away (this thin structure will be ignored)
cfg.obj_lvis.dist_th = 0.05

# cast shadow onto environment
cfg.env_lvis = CN()  # 2 and a half minutes to render an 1024 x 1024 image...
cfg.env_lvis.iter = 16
cfg.env_lvis.offset = 0.01  # this value couples heavily with the iteration count
cfg.env_lvis.relax = 0.0
cfg.env_lvis.near_offset = 0.02  # march away (this thin structure will be ignored)
cfg.env_lvis.bbox_margin = 0.25
cfg.env_lvis.dist_th = 0.005

cfg.xyz_res = 10
cfg.view_res = 4
cfg.surf_reg_th = 0.02
cfg.temporal_dim = 256
cfg.interpolate_path = False

cfg.mesh = CN()
cfg.mesh.meta = ''
cfg.mesh.type = 'tpose'
cfg.mesh.lambda_smooth = 9
cfg.mesh.w = 256
cfg.mesh.d = 8
cfg.mesh.replace_tjoints = False

cfg.nt = CN()
cfg.nt.H = 4096  # 4k texture might just be too large
cfg.nt.W = 4096
cfg.nt.step = 11
cfg.nt.height = 0.02  # 0.05 in, and 0.05 out
cfg.nt.dim = 64
cfg.nt.nerf_w = 256
cfg.nt.nerf_d = 8
cfg.nt.skips = [4, ]
cfg.nt.cnn_dim = 256
cfg.use_cvae = False
cfg.use_deform = False

cfg.print_network = True
cfg.table_row_limit = 5

cfg.profiling = CN()
cfg.profiling.enabled = False
cfg.profiling.clear_previous = True
cfg.profiling.skip_first = 10
cfg.profiling.wait = 5
cfg.profiling.warmup = 5
cfg.profiling.active = 10
cfg.profiling.repeat = 5
cfg.profiling.record_dir = ""
cfg.profiling.record_shapes = True
cfg.profiling.profile_memory = True
cfg.profiling.with_stack = True
cfg.profiling.with_flops = True
cfg.profiling.with_modules = True

cfg.detect_anomaly = False
cfg.mesh_th_to_sdf = False

cfg.blend_radius = 0.075
cfg.sample_vert_cnt = 3

cfg.fixed_lbs_pose = -1
cfg.surface_blend_weight = False

# Loss Configuration
cfg.img_loss_weight = 1.0
cfg.kld_loss_weight = 1e-5
cfg.resd_loss_weight = 0.01
cfg.resd_loss_weight_gamma = 1.0
cfg.resd_loss_weight_milestone = 1
cfg.presd_loss_weight = 0.01
cfg.dist_loss_weight = 0.01
cfg.msk_loss_weight = 0.01
cfg.norm_loss_weight = 0.001
cfg.sem_loss_weight = 0.001
cfg.eikonal_loss_weight = 0.025
cfg.observed_eikonal_loss_weight = 0.050
cfg.ctof_loss_weight = 0.1
cfg.albedo_sparsity = 5.0e-4
cfg.albedo_smooth_weight = 5.0e-3
cfg.roughness_smooth_weight = 5.0e-3

cfg.eval_whole_img = True
cfg.dry_run = False
cfg.sdf_res = 6
cfg.train_chunk_size = 4096
cfg.render_chunk_size = 8192
cfg.network_chunk_size = 4096 * 64
cfg.bg_brightness = 0.0
cfg.sdf_beta_init_value = 0.1
cfg.feat_dim = 256
cfg.resd_limit = 0.05
cfg.cond_dim = -1
cfg.occ_th = 0.5
cfg.dist_th = 0.1
cfg.surf_reg_sdf_th = 0.02
cfg.sdf_finite_diff = 0

cfg.latent_dim = 128
cfg.collate = True
cfg.load_others = True
cfg.bw_sample_blend_K = 16


cfg.bkgd = 'bkgd'
cfg.mask = 'mask'  # empty mask dir, will iteration through predefined ones

cfg.pin_memory = True
cfg.prefetch_factor = 10
cfg.subpixel_sample = False
cfg.n_bones = 24
cfg.fixed_latent = -1  # whether to fix passed in latent code for feature
cfg.smoothing_term = 10.0  # how much smoothing to use on camera path, 1.0 is a lot
cfg.perform = False  # performing in visualing novel view?
cfg.crop_min_size = 180
cfg.crop_max_size = 200

cfg.perturb = 1.
cfg.n_samples = 64
cfg.n_importance = 128

cfg.mesh_simp_face = -1  # target number of faces to retain if doing mesh simplification

# experiment name
cfg.exp_name = 'default'

# network
cfg.distributed = False

# data
cfg.skip = []
cfg.human = 313
cfg.training_view = [0, 6, 12, 18]
cfg.test_view = [0, 1, 2, 3]
cfg.begin_ith_latent = 0
cfg.begin_ith_frame = 0  # the first smpl
cfg.num_train_frame = 1  # number of smpls
cfg.num_render_frame = -1  # number of frames to render
cfg.frame_interval = 1
cfg.mask_bkgd = True
cfg.body_sample_ratio = 0.5
cfg.face_sample_ratio = 0.
cfg.use_geodesic_filter = True

# mesh
cfg.mesh_th = 0.5  # threshold of alpha

# task
cfg.task = 'deform'

# gpus
cfg.gpus = list(range(8))
cfg.resume = True  # if load the pretrained network

# epoch
cfg.ep_iter = -1
cfg.save_ep = 200
cfg.eval_ep = 100
cfg.save_latest_ep = 1

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 10000
cfg.train.load_epoch = -1
cfg.train.num_workers = 8
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.train.shuffle = True

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.eps = 1e-8
cfg.train.weight_decay = 0.

cfg.train.lr_table = CN()  # will query the parameter, if found match, use lr in table
cfg.train.eps_table = CN()
cfg.train.weight_decap_table = CN()

cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})
cfg.train.batch_size = 4

# -----------------------------------------------------------------------------
# test
# -----------------------------------------------------------------------------
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.sampler = 'default'
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.test.frame_sampler_interval = 30
cfg.test.view_sampler_interval = 3

# trained model
cfg.trained_model_dir = 'data/trained_model'

# recorder
cfg.record_dir = 'data/record'
cfg.log_interval = 1
cfg.record_interval = 5

# result
cfg.result_dir = 'data/result'

# training
cfg.tpose_geometry = 'bigpose'
cfg.erode_dilate_edge = True

# evaluation
cfg.replace_light = ''
cfg.test_light = ['gym_entrance']  # assume no test light by default
cfg.rotate_ratio = 4  # will upscale then roll then downscale
cfg.vis_ground_shading = False
cfg.sdf_add_specular = False
cfg.ground_attach_envmap = True  # try to attach the environment for rendering
cfg.probe_size_ratio = 0.2
cfg.fix_random = False
cfg.skip_eval = False
cfg.test_novel_pose = False

# visualization
cfg.novel_view_center = []
cfg.novel_view_z_off = -1


class Output(Enum):
    # visualization keys and configurations
    Semantic = auto()
    Feature = auto()
    Surface = auto()
    Residual = auto()
    Depth = auto()
    Alpha = auto()
    Normal = auto()
    Specular = auto()
    Albedo = auto()
    Roughness = auto()
    Shading = auto()
    Rendering = auto()
    Envmap = auto()


for type in Output:
    cfg[f'vis_{type.name.lower()}_map'] = False  # initialize configuration for output visualization

cfg.vis_median_depth = False
cfg.vis_rotate_light = False
cfg.vis_sphere_tracing = False  # will force to used a sphere tracing renderer for neural sdf
cfg.vis_novel_light = False  # will load some custom lighting condition or just use olat? TODO: add to visualizer / dataset implementation
cfg.vis_pose_sequence = False
cfg.vis_novel_view = False
cfg.vis_tpose_mesh = False
cfg.vis_posed_mesh = False
cfg.vis_can_mesh = False
cfg.track_tpose_mesh = False
cfg.shading_albedo = 0.8  # for visualing the shade map
cfg.vis_ext = '.jpg'  # store 16 bit png by default

# visualization image store config
cfg.store_alpha_channel = True  # will try to store output images with alpha channel as pngs if possible
cfg.store_ground_truth = False
cfg.store_image_error = False
cfg.print_render_progress = False
cfg.geometry_normal = False
cfg.geometry_visibility = False
cfg.local_visibility = False  # dot product between normal and light direction
cfg.always_fix_material = True
cfg.no_dfss = False
cfg.albedo_slope = 1.0
cfg.albedo_bias = 0.0
cfg.roughness_slope = 0.90  # force larger roughness values
cfg.roughness_bias = 0.09
cfg.relight_network_width = 128
cfg.relight_network_depth = 2
cfg.relight_xyz_res = 10
cfg.relight_view_res = 4
cfg.envmap_init_intensity = 0.2
cfg.tonemapping_albedo = True  # better visualization
cfg.tonemapping_rendering = True
cfg.rgb_as_albedo = False
cfg.zero_roughness = False
cfg.ray_samples = 64
cfg.vis_samples = 64  # this might be too slow for rendering, training would be fine
cfg.extra_prefix = ''
cfg.store_video_output = True
cfg.only_visibility = False
cfg.albedo_multiplier = 1.0


def default_cfg():
    return cfg


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # Load bodymodel related configurations
    cfg_model = Config.load(join(cfg.train_dataset.data_root, cfg.body_model))
    cfg_model.module = cfg_model.module.replace('SMPLHModelEmbedding', 'SMPLHModel')
    cfg_model.args.device = 'cpu'
    bodymodel: SMPLModel = load_object(cfg_model.module, cfg_model.args)
    cfg.n_bones = bodymodel.NUM_POSES_FULL // 3

    # Load visualization relatedc configurations
    types = [k for k in Output if cfg[f'vis_{k.name.lower()}_map']]
    if not types: cfg[f'vis_{Output.Rendering.name.lower()}_map'] = True
    if cfg.vis_ext == '.exr' or cfg.vis_ext == '.hdr':
        cfg.tonemapping_rendering = False
        cfg.tonemapping_albedo = False

    # Load light visibility related configurations
    if cfg.vis_ground_shading:
        cfg.store_alpha_channel = False
        # cfg.no_claybook = False # will always use claybook when using ground_shading

    # Claybook banding removal technique
    if not cfg.no_claybook:
        if cfg.obj_lvis.offset >= 0.05:
            log(f'cfg.obj_lvis.offset is {cfg.obj_lvis.offset}, claybook might produce artifacts', 'red')
        if cfg.env_lvis.offset >= 0.05:
            log(f'cfg.env_lvis.offset is {cfg.env_lvis.offset}, claybook might produce artifacts', 'red')

    if cfg.fixed_latent == -1:
        cfg.fixed_latent = 0 if cfg.test_novel_pose else -1

    if cfg.cond_dim < 0:
        cfg.cond_dim = cfg.n_bones * 3  # maybe hand rotation should not be considered?

    # NOTE: assign the gpus, this will ignore cfg.gpus if you've assigned CUDA_VISIBLE_DEVICES already
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    # Get rid of ugly TF logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)

    cfg.local_rank = args.local_rank
    cfg.distributed = cfg.distributed or args.launcher not in ['none']

    if cfg.profiling.enabled:
        cfg.train.epoch = 1
        cfg.ep_iter = cfg.profiling.skip_first + cfg.profiling.repeat * (cfg.profiling.wait + cfg.profiling.warmup + cfg.profiling.active)
        cfg.profiling.record_dir = cfg.record_dir


def update_cfg(cfg: CN, args):
    cfg_file = yacs.load_cfg(open(args.cfg_file, 'r'))
    cfg.merge_strain(cfg_file)
    cfg.merge_from_list(args.opts)  # load commandline config before merging

    # if cfg.vis_novel_light:
    #     if not cfg.vis_novel_view:
    #         cfg.vis_pose_sequence = True # use the test dataset by default
    # This will mess up test.frame_sampler_interval

    # These two should come first
    if cfg.relighting:
        cfg.merge_from_other_cfg(cfg.relighting_cfg)

    if cfg.vis_pose_sequence:
        cfg.merge_from_other_cfg(cfg.pose_seq_cfg)

    if cfg.vis_novel_view:
        cfg.merge_from_other_cfg(cfg.novel_view_cfg)

    if cfg.vis_tpose_mesh or cfg.vis_posed_mesh or cfg.vis_can_mesh:
        cfg.merge_from_other_cfg(cfg.mesh_cfg)

    if cfg.vis_sphere_tracing:
        cfg.merge_from_other_cfg(cfg.sphere_tracing_cfg)

    # This one should come last
    if cfg.vis_novel_light:
        cfg.merge_from_other_cfg(cfg.novel_light_cfg)

    cfg.merge_from_list(args.opts)  # load commandline config after merging
    parse_cfg(cfg, args)  # load some environment variables and defaults
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument('-c', "--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('-t', "--type", type=str, default="")
parser.add_argument('-r', '--local_rank', type=int, default=0)
parser.add_argument('-l', '--launcher', type=str, default='none', choices=['none', 'pytorch'])
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
parser.add_argument('--test', action='store_true', dest='test', default=False)

args = None

if sys.argv[0].endswith('run.py') or sys.argv[0].endswith('train.py'):
    args = parser.parse_args()
    cfg = default_cfg()
    if len(args.type) > 0:
        cfg.task = "run"

    cfg = update_cfg(cfg, args)
    log(cfg.exp_name, 'magenta')

"""
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 train.py -c configs/phdeform/monosdf_my_313.yaml distributed True exp_name monosdf_ddp load_normal False load_semantics False

torchrun --nproc_per_node=2 train.py -c configs/phdeform/phdeform_xuzhen36.yaml distributed True
"""
