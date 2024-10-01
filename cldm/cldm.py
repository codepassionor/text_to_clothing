import os
from os.path import join as opj
import omegaconf

import cv2
import einops
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

from utils import tensor2img, resize_mask
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

class PMJewelryControlLDM(LatentDiffusion):
    def __init__(
            self, 
            control_stage_config, 
            validation_config, 
            control_key, 
            only_mid_control, 
            use_VAEdownsample=False,
            all_unlocked=False,
            config_name="",
            control_scales=None,
            use_pbe_weight=False,
            u_cond_percent=0.0,
            img_H=512,
            img_W=384,
            imageclip_trainable=True,
            pbe_train_mode=False,
            use_attn_mask=False,
            always_learnable_param=False,
            mask1_key="neck_mask",
            mask2_key="ear_mask",
            *args, 
            **kwargs
        ):
        self.control_stage_config = control_stage_config
        self.use_pbe_weight = use_pbe_weight
        self.u_cond_percent = u_cond_percent
        self.img_H = img_H
        self.img_W = img_W
        self.config_name = config_name
        self.imageclip_trainable = imageclip_trainable
        self.pbe_train_mode = pbe_train_mode
        self.use_attn_mask = use_attn_mask
        self.mask1_key = mask1_key
        self.mask2_key = mask2_key
        self.always_learnable_param = always_learnable_param
        super().__init__(*args, **kwargs)
        control_stage_config.params["use_VAEdownsample"] = use_VAEdownsample
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        if control_scales is None:
            self.control_scales = [1.0] * 13
        else:
            self.control_scales = control_scales
        self.first_stage_key_cond = kwargs.get("first_stage_key_cond", None)
        self.valid_config = validation_config
        self.use_VAEDownsample = use_VAEdownsample
        self.all_unlocked = all_unlocked
        self.gmm = None
        self.clothflow = None

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        if isinstance(self.control_key, omegaconf.listconfig.ListConfig):
            control_lst = []
            for key in self.control_key:
                control = batch[key]
                if bs is not None:
                    control = control[:bs]
                control = control.to(self.device)
                control = einops.rearrange(control, 'b h w c -> b c h w')
                control = control.to(memory_format=torch.contiguous_format).float()
                control_lst.append(control)
            control = control_lst
        else:
            control = batch[self.control_key]
            if bs is not None:
                control = control[:bs]
            control = control.to(self.device)
            control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float()
            control = [control]
        cond_dict = dict(c_crossattn=[c], c_concat=control)
        if self.first_stage_key_cond is not None:
            first_stage_cond = []
            for key in self.first_stage_key_cond:
                if not "mask" in key:
                    cond, _ = super().get_input(batch, key, *args, **kwargs)
                else:
                    cond, _ = super().get_input(batch, key, no_latent=True, *args, **kwargs)      
                first_stage_cond.append(cond)
            first_stage_cond = torch.cat(first_stage_cond, dim=1)
            cond_dict["first_stage_cond"] = first_stage_cond
        return x, cond_dict

    def apply_model(self, x_noisy, t, cond, user_preferences, *args, **kwargs):
        assert isinstance(cond, dict)
        
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond["c_crossattn"], 1)
        
        # Adjust jewelry style based on user preferences
        personalized_jewelry = self.adjust_jewelry_style(cond_txt, user_preferences)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=personalized_jewelry, control=None, only_mid_control=self.only_mid_control)
        else:
            if "first_stage_cond" in cond:
                x_noisy = torch.cat([x_noisy, cond["first_stage_cond"]], dim=1)
            hint = torch.cat(cond["c_concat"], dim=1)
            control, cond_output = self.control_model(x=x_noisy, hint=hint, timesteps=t, context=personalized_jewelry, only_mid_control=self.only_mid_control)
            if len(control) == len(self.control_scales):
                control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=personalized_jewelry, control=control, only_mid_control=self.only_mid_control)
        return eps, None

    def adjust_jewelry_style(self, jewelry_features, user_preferences):
        # Modify the jewelry features based on user preferences (e.g., length, material)
        preference_weights = self.get_user_preference_weights(user_preferences)
        adjusted_jewelry = jewelry_features * preference_weights
        return adjusted_jewelry

    def get_user_preference_weights(self, user_preferences):
        # Dummy implementation, should be replaced with actual logic to compute weights based on user preferences
        return torch.ones_like(user_preferences)

    def depth_conditioning(self, jewelry_mask, depth_map):
        # Apply depth-aware transformations on jewelry mask
        jewelry_with_depth = apply_depth_map(jewelry_mask, depth_map)
        return jewelry_with_depth

    def apply_jewelry_total_variation_loss(self, attention_map, jewelry_mask):
        # Calculate total variation loss specific for jewelry
        F = self.weighted_center(attention_map)
        loss_atv = torch.sum(torch.abs(F - jewelry_mask))
        return loss_atv

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        if self.first_stage_key_cond:
            first_stage_cond = c["first_stage_cond"][:N]
            log["first_stage_cond"] = first_stage_cond
        c_cat = [i[:N] for i in c["c_concat"]]
        c = c["c_crossattn"][0][:N]
        if c.ndim == 4:
            c = self.get_learned_conditioning(c)
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        
        x = batch[self.first_stage_key]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        log["input"] = x
        log["reconstruction"] = self.decode_first_stage(z)
        log_c_cat = torch.cat(c_cat, dim=1)
        log["control"] = log_c_cat * 2.0 - 1.0

        if sample:
            samples, z_denoise_row = self.sample_log(cond={"c_concat": c_cat, "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

        if unconditional_guidance_scale >= 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat
            cond = {"c_concat": c_cat, "c_crossattn": [c]}
            uc_full = {"c_concat": uc_cat, "c_crossattn": [uc_cross]}
            samples_cfg, _, cond_output_dict = self.sample_log(cond=cond,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full)
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps=5, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates, cond_output_dict = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates, cond_output_dict
