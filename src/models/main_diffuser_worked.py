# Adapted from https://github.com/MooreThreads/Moore-AnimateAnyone/blob/master/src/pipelines/pipeline_pose2vid_long.py
import inspect
import math
from enum import Enum
from typing import Callable, List, Optional, Union

import torch
from diffusers import DiffusionPipeline
from diffusers.loaders import LoraLoaderMixin
from diffusers.schedulers import (
    DDIMScheduler,
    LCMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils.torch_utils import randn_tensor

from .mutual_self_attention import ReferenceAttentionControl
from ..utils.util import get_tensor_interpolation_method, get_context_scheduler

class CFGType(Enum):
    rcfg_self_negative = 1
    rcfg_one_time_negative = 2
    full_double_pass = 3
    
class AADiffusion(DiffusionPipeline, LoraLoaderMixin):
    def __init__(
        self, 
        reference_unet,
        denoising_unet,
        scheduler: Union[
            DDIMScheduler,
            LCMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()
        self.register_modules(
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        )
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.scheduler = scheduler
        
        self.noise_pred_uncond_prev = None
        
    def load_lora(self, pretrained_model_name_or_path_or_dict, adapter_name=None, **kwargs):
        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict, network_alphas = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", True)

        self.load_lora_into_unet(
            state_dict,
            network_alphas=network_alphas,
            unet=getattr(self, self.denoising_unet_name) if not hasattr(self, "unet") else self.denoising_unet,
            low_cpu_mem_usage=low_cpu_mem_usage,
            adapter_name=adapter_name,
            _pipeline=self,
        )
    
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        video_length,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height,
            width,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def interpolate_latents(self, latents: torch.Tensor, interpolation_factor: int, device):
        if interpolation_factor < 2:
            return latents

        new_latents = torch.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                ((latents.shape[2] - 1) * interpolation_factor) + 1,
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )

        org_video_length = latents.shape[2]
        rate = [i / interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
            v0 = latents[:, :, i0, :, :]
            v1 = latents[:, :, i1, :, :]

            new_latents[:, :, new_index, :, :] = v0
            new_index += 1

            for f in rate:
                v = get_tensor_interpolation_method()(
                    v0.to(device=device), v1.to(device=device), f
                )
                new_latents[:, :, new_index, :, :] = v.to(latents.device)
                new_index += 1

        new_latents[:, :, new_index, :, :] = v1
        new_index += 1

        return new_latents
    
    def perform_guidance(
        self, 
        noise_pred,
        counter,
        cfg,
        delta,
        do_classifier_free_guidance,
        do_double_pass,
        noise_pred_uncond_prev=None,
        ):
        
        # perform CFG/RCFG guidance
        if do_classifier_free_guidance:
            if do_double_pass:
                noise_pred_uncond, noise_pred_cond = (noise_pred / counter).chunk(2)
            elif noise_pred_uncond_prev is not None:
                noise_pred_cond = noise_pred / counter
                noise_pred_uncond = noise_pred_uncond_prev
            else:
                noise_pred_cond = noise_pred_uncond = noise_pred / counter
                cfg = 0.0
            # RCFG of formula (7) from https://arxiv.org/pdf/2312.12491.pdf
            noise_pred = delta * noise_pred_uncond + cfg * (noise_pred_cond - delta * noise_pred_uncond)

        return noise_pred
    
    def denoise_all_one_step(
        self, 
        t, 
        latents, 
        encoder_hidden_states,
        pose_latent, 
        global_context, 
        do_double_pass,
        device,
    ):
        noise_pred = torch.zeros(
            (
                latents.shape[0] * (2 if do_double_pass else 1),
                *latents.shape[1:],
            ),
            device=device,
            dtype=latents.dtype,
        )
        counter = torch.zeros(
            (1, 1, latents.shape[2], 1, 1),
            device=device,
            dtype=latents.dtype,
        )

        for context in global_context:
            # 3.1 expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents[:, :, c] for c in context]) # Concatenates the context in the same batch
                .to(device)
                .repeat(2 if do_double_pass else 1, 1, 1, 1, 1)
            )
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )
            b, c, f, h, w = latent_model_input.shape
            latent_pose_input = torch.cat(
                [pose_latent[:, :, c] for c in context]
            ).repeat(2 if do_double_pass else 1, 1, 1, 1, 1)
            pred = self.denoising_unet(
                latent_model_input,     # torch.Size([2, 4, 24, 96, 64])
                t,
                encoder_hidden_states=encoder_hidden_states[:b],
                pose_cond_fea=latent_pose_input,    # torch.Size([2, 320, 24, 96, 64])
                return_dict=False,
            )[0]

            for j, c in enumerate(context):
                noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                counter[:, :, c] = counter[:, :, c] + 1
                
        return noise_pred, counter
        
    def solve_one_step(
        self,
        t, 
        latents, 
        encoder_hidden_states,
        pose_latent, 
        global_context, 
        cfg,
        delta,
        do_classifier_free_guidance,
        do_double_pass,
        do_rcfg,
        noise_pred_uncond_prev,
        device,
        extra_step_kwargs,
    ):
        
        noise_pred, counter = self.denoise_all_one_step(
            t, 
            latents, 
            encoder_hidden_states,
            pose_latent, 
            global_context, 
            do_double_pass,
            device,
        )
        noise_pred = self.perform_guidance(
            noise_pred,
            counter,
            cfg,
            delta,
            do_classifier_free_guidance,
            do_double_pass,
            noise_pred_uncond_prev,
        )
  
        scheduler_output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
        
        # compute the residual empty/negative condition noise sample
        if do_rcfg:
            pred_original_sample = scheduler_output.pred_original_sample
            
            alpha_prod_t = self.scheduler.alphas_cumprod[t].to(device)
            beta_prod_t = 1 - alpha_prod_t
            
            if self.scheduler.config.prediction_type == "epsilon":
                # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                # rearranged from formular: pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                noise_pred_uncond_prev = (latents - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
                
            elif self.scheduler.config.prediction_type == "v_prediction":
                # v_prediction: (see section 2.4 of [Imagen Video](https://imagen.research.google/video/paper.pdf)
                # rearranged from formular: pred_original_sample = (alpha_prod_t ** (0.5) * latents - (beta_prod_t ** (0.5) * noise_pred
                noise_pred_uncond_prev = (alpha_prod_t ** (0.5) * latents - pred_original_sample) / beta_prod_t ** (0.5)
            
        return scheduler_output.prev_sample, noise_pred_uncond_prev

    @torch.no_grad()
    def __call__(
        self,
        steps,
        cfg,
        delta,
        cfg_type,
        steps_rcfg,
        ref_image_latent,
        pose_latent,
        encoder_hidden_states,
        seed = 999999999,
        eta: float = 0.0,
        context_schedule="uniform",
        context_frames=24,
        context_stride=1,
        context_overlap=4,
        context_batch_size=1,
        interpolation_factor=1,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):

        do_classifier_free_guidance = cfg > 0
        do_double_pass = do_classifier_free_guidance and cfg_type == CFGType.full_double_pass
        do_rcfg = do_classifier_free_guidance and (cfg_type == CFGType.rcfg_self_negative or cfg_type == CFGType.rcfg_one_time_negative)

        device = pose_latent.device
        generator=torch.manual_seed(seed)
        video_length = pose_latent.shape[2] # number of frames
        
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        if hasattr(self.scheduler, "final_alpha_cumprod"):
            self.scheduler.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        
        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # Prepare Input Batches
        context_scheduler = get_context_scheduler(context_schedule)
        
        context_queue = list(
            context_scheduler(
                0,
                steps,
                video_length,
                context_frames,
                context_stride,
                context_overlap,
            )
        )

        num_context_batches = math.ceil(len(context_queue) / context_batch_size)
        global_context = []
        for i in range(num_context_batches):
            global_context.append(
                context_queue[
                    i * context_batch_size : (i + 1) * context_batch_size
                ]
            )
            
        """
        Example: num_frames == 72, context_frames == 24, context_stride == 1, context_overlap == 4, context_batch_size == 1
        context_queue: [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], 
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], 
            [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        ]
        global_context: [   # shape: (num_context_batches, context_batch_size, context_frames)
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]], 
            [[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]], 
            [[40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]], 
            [[60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        ]
        """
        
        # Prepare initial latents
        latents = self.prepare_latents(
            batch_size=pose_latent.shape[0],
            num_channels_latents=self.denoising_unet.config.in_channels,
            video_length=video_length,
            height=pose_latent.shape[-2],
            width=pose_latent.shape[-1],
            dtype=encoder_hidden_states.dtype,
            device=device,
            generator=generator,
        )
        
        # Prepare confitional embeds
        if do_classifier_free_guidance:
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
            if cfg_type == CFGType.full_double_pass:
                encoder_hidden_states = torch.cat(
                    [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
                )
                
        # Prepare reference attention control
        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_double_pass,
            mode="write",
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_double_pass,
            mode="read",
            fusion_blocks="full",
        )
        
        if do_rcfg:
            # predict initial uncondition noise for RCFG
            self.reference_unet(
                ref_image_latent,
                torch.zeros(size=(1,), dtype=int, device=device),
                encoder_hidden_states=uncond_encoder_hidden_states,
                return_dict=False,
            )
            reference_control_reader.update(reference_control_writer)
            
            self.scheduler.set_timesteps(steps_rcfg, device=device)
            timesteps = self.scheduler.timesteps
            
            with self.progress_bar(total=steps_rcfg) as progress_bar:
                for i, t in enumerate(timesteps):
                    _, self.noise_pred_uncond_prev = self.solve_one_step(
                        t, 
                        latents, 
                        encoder_hidden_states,
                        pose_latent, 
                        global_context, 
                        cfg,
                        delta,
                        do_classifier_free_guidance,
                        do_double_pass,
                        i+1 == steps_rcfg,
                        self.noise_pred_uncond_prev,
                        device,
                        extra_step_kwargs,
                    )
                    
                    progress_bar.update()
            
            reference_control_writer.clear()
            reference_control_reader.clear()
            
            #if hasattr(self.scheduler, "_step_index"):
            #    self.scheduler._step_index = 0
            
        self.reference_unet(
            ref_image_latent.repeat(2 if do_double_pass else 1, 1, 1, 1),
            torch.zeros(size=(1,), dtype=int, device=device),
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        reference_control_reader.update(reference_control_writer)
        
        # Prepare timesteps
        self.scheduler.set_timesteps(steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # denoising loop
        num_warmup_steps = len(timesteps) - steps * self.scheduler.order
        with self.progress_bar(total=steps) as progress_bar:
            for i, t in enumerate(timesteps):

                latents, self.noise_pred_uncond_prev = self.solve_one_step(
                    t, 
                    latents, 
                    encoder_hidden_states,
                    pose_latent, 
                    global_context, 
                    cfg,
                    delta,
                    do_classifier_free_guidance,
                    do_double_pass,
                    do_rcfg,
                    self.noise_pred_uncond_prev,
                    device,
                    extra_step_kwargs,
                )

                # update progress bar and callback
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if interpolation_factor > 0:
            latents = self.interpolate_latents(latents, interpolation_factor, device)
            
        reference_control_reader.clear()
        reference_control_writer.clear()
        
        
        
        # double pass fine-tune
        

        return latents