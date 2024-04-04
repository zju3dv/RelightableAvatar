# Parent: sphere_tracing_renderer
import time
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from termcolor import colored

from lib.config import cfg
from lib.utils.base_utils import dotdict
from lib.utils.log_utils import log, run
from lib.utils.data_utils import to_cpu
from lib.utils.net_utils import normalize
from lib.utils.relight_utils import expand_envmap_probe, sample_envmap_image, rotate_envmap, add_light_probe, expand_envmap_xyz

from . import sphere_tracing_renderer
from .sphere_tracing_renderer import evaluate_shade, linear2srgb, blend_output_, pix_chunkify


@pix_chunkify
def render_human(ray_o: torch.Tensor, surf: torch.Tensor, norm: torch.Tensor, albedo: torch.Tensor, roughness: torch.Tensor, lvis_map: torch.Tensor, ldot_map: torch.Tensor, inputs: dotdict):
    xyz = inputs.xyz
    area = inputs.area
    envmap = inputs.envmap
    microfacet = inputs.microfacet

    # render the human part
    surf2light = normalize(xyz[:, :, None] - surf[:, None, None])
    surf2cam = normalize(ray_o - surf)
    light = sample_envmap_image(envmap.probe, surf2light)  # B, eH, eW, P, 3
    B, eH, eW, P, _ = light.shape

    # prepare names
    lvis = lvis_map.permute(0, 2, 1).view(B, eH, eW, P)
    ldot = ldot_map.permute(0, 2, 1).view(B, eH, eW, P)

    brdf = microfacet(surf2light, surf2cam, norm, albedo, roughness)
    if microfacet.cancel_cosine:
        ori_ldot = ldot
        ldot = torch.ones_like(ldot)
    shade = evaluate_shade(lvis,
                           ldot,
                           area,
                           light)
    rgb = brdf * shade
    rgb = rgb.sum(dim=1).sum(dim=1)
    rgb = linear2srgb(rgb)

    spec_brdf = microfacet(surf2light, surf2cam, norm, 0.0, roughness)
    if microfacet.cancel_cosine:
        # ignore these to make it more visible
        ldot = 1 / (torch.abs(ldot) + 1e-8)
    else:
        ldot = torch.ones_like(ldot)
    spec_shade = evaluate_shade(torch.ones_like(lvis), ldot, area, light)  # memory?
    spec = spec_brdf * spec_shade
    spec = spec.sum(dim=1).sum(dim=1)

    # http://www.joshbarczak.com/blog/?p=272
    ldot = ldot if 'ori_ldot' not in locals() else ori_ldot
    shade = evaluate_shade(lvis, ldot, area, light)  # memory?
    shade = shade.sum(1).sum(1) * cfg.shading_albedo / np.pi  # brdf should sum to 1

    # Now, the rendering has finished
    return rgb, shade, spec


@pix_chunkify
def render_ground(ray_d: torch.Tensor, surf_map: torch.Tensor, albedo_map: torch.Tensor, lvis_map: torch.Tensor, ldot_map: torch.Tensor, inputs: dotdict):
    xyz = inputs.xyz
    area = inputs.area
    envmap = inputs.envmap

    # render the ground part
    surf2light = normalize(xyz[:, :, None] - torch.zeros_like(surf_map[:, None, None]))
    light = sample_envmap_image(envmap.probe, surf2light)  # B, eH, eW, P, 3
    if cfg.ground_attach_envmap:
        if 'image' in envmap:
            albedo = sample_envmap_image(envmap.image, ray_d)  # B, F, 3
        else:
            albedo = sample_envmap_image(envmap.probe, ray_d)  # B, F, 3
    else:
        albedo = albedo_map
    brdf = albedo[:, None, None] / np.pi

    B, eH, eW, P, _ = light.shape
    shade = evaluate_shade(lvis_map.permute(0, 2, 1).view(B, eH, eW, P),
                           ldot_map.permute(0, 2, 1).view(B, eH, eW, P),
                           area,
                           light)
    rgb = brdf * shade
    rgb = rgb.sum(dim=1).sum(dim=1)
    rgb = linear2srgb(rgb)
    shade = shade.sum(1).sum(1) / np.pi
    # Now, the rendering has finished

    return rgb, albedo, shade, shade / 20


class Renderer(sphere_tracing_renderer.Renderer):

    def render(self, batch):
        # render the master image
        log(f'rendering for trained environment map (compute intersection and visibility)')
        
        torch.cuda.synchronize()
        tick = time.perf_counter()

        main = super(self.__class__, self).render(batch)  # the real heavy lifting happens here

        torch.cuda.synchronize()
        tock = time.perf_counter()
        diff = tock - tick
        log(f'Net work rendering time: {diff}', 'green')

        # remove unwanted stuff to save memory
        keys = ['surf_map',
                'depth_map',
                'acc_map',
                'rgb_map',
                'albedo_map',
                'roughness_map',
                'norm_map',
                'shade_map',
                'ray_o',
                'ray_d',
                'lvis_map',  # MARK: MEM
                'ldot_map',  # MARK: MEM
                'ground',
                'inds',
                'cpts_map',
                'bpts_map',
                'spec_map',
                'envmap',
                ]
        main = dotdict({k: main[k] for k in keys if k in main})  # will remove the reference, so get deallocated
        if 'ground' in main:
            main.ground = dotdict({k: main.ground[k] for k in keys if k in main.ground})

        # less strain on memory
        visual = ['rgb_map',
                  'acc_map',
                  'norm_map',
                  'surf_map',
                  'bpts_map',
                  'cpts_map',
                  'spec_map',
                  'shade_map',
                  'depth_map',
                  'albedo_map',
                  'roughness_map',
                  'envmap',
                  ]

        # render relighted image one by one from batch.novel_lights by calling get_pixel_value
        relight = dotdict()
        if 'main' in cfg.test_light:  # store the rendering results with estimated light if possible
            relight.main = dotdict({k: main[k] for k in visual if k in main})
            if 'ground' in main:
                relight.main = blend_output_(main.ground.acc_map, main.ground.inds, main.ground, relight.main)

        rotate_ratio = cfg.rotate_ratio if cfg.vis_rotate_light else -1
        rotation = rotate_ratio * cfg.env_w if cfg.vis_rotate_light else 1
        n_total_light = len(batch.novel_lights) * rotation  # total number of light
        pbar = tqdm(total=n_total_light)

        log('Rendering for all lights, this may take a while.')
        log('Note: only will images be saved to disk when all rendering in this light pass finishes.')
        for i in range(n_total_light):
            name, envmap = rotate_envmap(batch.novel_lights, i, rotate_ratio, cfg.env_w, cfg.env_image_w)
            # log(f'rendering for envmap {colored(f"#{i:02d}/{n_total_light:02d}", "magenta")}: {name}')
            pbar.desc = name

            # prepare names
            inputs = dotdict()
            inputs.xyz = self.net.light_xyz
            inputs.area = self.net.light_area
            inputs.microfacet = self.net.microfacet
            inputs.envmap = envmap

            rgb, shade, spec = render_human(main.ray_o, main.surf_map, main.norm_map, main.albedo_map, main.roughness_map, main.lvis_map, main.ldot_map, inputs)

            # temporary solution to memory issue -> much slower
            human = dotdict(rgb_map=rgb,
                            shade_map=shade,
                            spec_map=spec,
                            )
            human = dotdict({**main, **human})  # will reference instead of copying tensor

            if 'ground' in main:
                inputs = dotdict()
                inputs.xyz = self.net.light_xyz
                inputs.area = self.net.light_area
                inputs.microfacet = self.net.microfacet
                inputs.envmap = envmap

                rgb, albedo, shade, spec = render_ground(main.ground.ray_d, main.ground.surf_map, main.ground.albedo_map, main.ground.lvis_map, main.ground.ldot_map, inputs)

                # temporary solution to memory issue -> might in turn lead to excessive main memory usage?
                ground = dotdict(rgb_map=rgb,
                                 albedo_map=albedo,
                                 shade_map=shade,
                                 spec_map=spec,
                                 )
                ground = dotdict({**main.ground, **ground})  # will reference instead of copying tensor

                # merge rendering results
                inds = ground.inds
                acc = ground.acc_map
                human = dotdict({k: human[k] for k in visual if k in human})
                ground = dotdict({k: ground[k] for k in visual if k in ground})
                human = blend_output_(acc, inds, ground, human)  # !: REPEATED BLENDING OPERATION

            human.envmap = dotdict(probe=envmap.probe)  # do not store the full resolution envmap
            relight[name] = to_cpu(human)  # save memory

            pbar.update(1)

        relight.diff = diff
        return relight
