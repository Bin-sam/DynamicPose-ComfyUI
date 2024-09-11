import os
import folder_paths as comfy_paths
import torch
import inspect
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import OrderedDict
from mmpose.apis import MMPoseInferencer

from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LCMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

from comfy import latent_formats
from .src.pose.dwpose import DWposeDetector
from .src.pose.draw_pose import get_info, draw_pose
from .src.models.pose_guider import PoseGuider
from .src.models.unet_2d_condition import UNet2DConditionModel
from .src.models.unet_3d import UNet3DConditionModel
from .src.models.main_diffuser import DPDiffusion
from .src.utils.align import align_process

ROOT_PATH = os.path.join(comfy_paths.get_folder_paths("custom_nodes")[0], "dynamicpose")
DEVICE = 'cuda'
WEIGHT_DETYPE = torch.float16
DEFAULT_CONFIG1_PATH = os.path.join(ROOT_PATH, "./configs/inference/inference_v1.yaml")
DEFAULT_CONFIG2_PATH = os.path.join(ROOT_PATH, "./configs/inference/inference_v2.yaml")
CONFIG1 = OmegaConf.load(DEFAULT_CONFIG1_PATH)
CONFIG2 = OmegaConf.load(DEFAULT_CONFIG2_PATH)

SCHEDULER_DICT = OrderedDict([
    ("DDIM", DDIMScheduler),
    ("DPM++ 2M Karras", DPMSolverMultistepScheduler),
    ("LCM", LCMScheduler),
    ("Euler", EulerDiscreteScheduler),
    ("Euler Ancestral", EulerAncestralDiscreteScheduler),
    ("LMS", LMSDiscreteScheduler),
    ("PNDM", PNDMScheduler),
])

class align:
    @classmethod
    def INPUT_TYPES(s):
        return{
            "required": {
                "ref_image": ('IMAGE',),
                "ref_info": ('POSEINFOS',),
                "pose_images" :('IMAGE',),
                "pose_info": ('POSEINFOS',),
            },
            "optional":{
                "height": ("INT",{"default": 784, "min": 100, "max": 4000}),
                "width": ("INT",{"default": 512, "min": 100, "max": 4000}),
            }
        }
    
    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES = ('aligend_ref_image',"aligned_pose_images",)
    FUNCTION = "align_pose"
    CATEGORY = "DynamicPose"

    def align_pose(self,ref_image,ref_info,pose_images,pose_info,height,width):
        ref_img_align, pose_imaegs_align = align_process(ref_image,ref_info,pose_images,pose_info,(width,height))
        return (torch.from_numpy(ref_img_align).unsqueeze(0), torch.from_numpy(pose_imaegs_align),)
        

class load_pose_model:
    @classmethod
    def INPUT_TYPES(s):
        return{
            "optional": {
                "body_model": (["rtmpose-x", "dwpose"],),
                "hand_model": (["rtmw-x", "dwpose"],),
                "foot_model": (["dwpose","rtmw-x"],),
            }
        }
    RETURN_TYPES = ("POSEMODELS",)
    RETURN_NAMES = ("pose_models",)
    FUNCTION = "load_pose"
    CATEGORY = "DynamicPose/loaders"

    def load_pose(self,body_model,hand_model,foot_model):
        model_dir = ROOT_PATH
        if foot_model == 'dwpose':
            model_path =  model_dir + '/pretrained_weights/dwpose'
            inferencer_foot = DWposeDetector(foot = True, model_path = model_path).to("cuda")
        else:
            print('not support yet')

        if body_model == "rtmpose-x":
            inferencer_body = MMPoseInferencer(
                pose2d = model_dir + '/pretrained_weights/rtmpose/rtmpose-x_8xb256-700e_coco-384x288.py',
                pose2d_weights= model_dir + '/pretrained_weights/rtmpose/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth',
                det_model= model_dir + '/pretrained_weights/rtmpose/rtmdet_m_640-8xb32_coco-person.py',
                det_weights = model_dir + '/pretrained_weights/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
                det_cat_ids=[0],
            )
        else:
            print('not support yet')

        if hand_model == "rtmw-x":
            inferencer_hand = MMPoseInferencer(
                pose2d = model_dir + '/pretrained_weights/rtmpose/rtmw-x_8xb320-270e_cocktail14-384x288.py',
                pose2d_weights = model_dir + '/pretrained_weights/rtmpose/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth',
                det_model = model_dir + '/pretrained_weights/rtmpose/rtmdet_m_640-8xb32_coco-person.py',
                det_weights = model_dir + '/pretrained_weights/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
                det_cat_ids=[0],
            )
        else:
            print('not support yet')
        
        return ([inferencer_body,inferencer_hand,inferencer_foot],)


class pose_extraction:
    @classmethod
    def INPUT_TYPES(s):
        return{
            "required":{
                "images": ("IMAGE",),
                "models": ("POSEMODELS",)
            },
            "optional": {
                "show_org" : ("BOOLEAN",{"default":False}),
            }
        }
        
    RETURN_TYPES = ("IMAGE","POSEINFOS" )
    RETURN_NAMES = ("pose_images","pose_infos")
    FUNCTION = "process"
    CATEGORY = "DynamicPose"
    
    def process(self,images,models,show_org):
        inferencer_body = models[0]
        inferencer_hand = models[1]
        inferencer_foot = models[2]
        pose_frames = []
        pose_infos = []
        for image in tqdm(images):
            frame = image.numpy() * 255
            H,W = frame.shape[0:2]
            body = inferencer_body(frame)
            body = next(body)['predictions'][0][0]
            body_w = inferencer_hand(frame)
            hand = next(body_w)['predictions'][0][0]
            feet, feet_scores = inferencer_foot(frame)
            feet = feet.tolist()[0]
            for points in feet:
                points[0] = points[0] * W
                points[1] = points[1] * H
            foot = {
                'keypoints':feet,
                'keypoint_scores':feet_scores.tolist()[0],
            }
            info = get_info(body,hand,foot, W, H)
            pose_infos.append(info)

            if not show_org:
                detected_map = draw_pose(info)
            else:
                detected_map = draw_pose(info, frame)
            
            pose_frames.append(detected_map)
            

        pose_images = torch.cat(pose_frames, dim=0)

        return (pose_images,pose_infos,)

class Load_reference_unet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "pretrained_base_unet_folder_path": ("STRING", {"default": ".../models/diffusers/SD1.5/unet/", "multiline": False}),
                "unet2d_model_path": ("STRING", {"default": "./pretrained_weights/reference_unet.pth", "multiline": False}),     
            },
        }

    RETURN_TYPES = (
        "UNET2D",
    )
    RETURN_NAMES = (
        "unet2d",
    )
    FUNCTION = "load_unet2d"

    CATEGORY = "DynamicPose/loaders"

    def load_unet2d(self, pretrained_base_unet_folder_path, unet2d_model_path):
        if not os.path.isabs(pretrained_base_unet_folder_path):
            pretrained_base_unet_folder_path = os.path.join(ROOT_PATH, pretrained_base_unet_folder_path)
        if not os.path.isabs(unet2d_model_path):
            unet2d_model_path = os.path.join(ROOT_PATH, unet2d_model_path)        
        
            
        unet2d = UNet2DConditionModel.from_pretrained(
            pretrained_base_unet_folder_path,
        ).to(dtype=WEIGHT_DETYPE, device=DEVICE)
        
        unet2d.load_state_dict(
            torch.load(unet2d_model_path, map_location="cpu"),
        )
        
        return (unet2d,)
    
class Load_denoising_unet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "type": (['image', 'video'],),
                "pretrained_base_unet_folder_path": ("STRING", {"default": ".../models/diffusers/SD1.5/unet/", "multiline": False}),
                "unet3d_model_path": ("STRING", {"default": "./pretrained_weights/denoising_unet.pth", "multiline": False}),
                "motion_module_path": ("STRING", {"default": "./pretrained_weights/motion_module.pth", "multiline": False}),
            },
        }

    RETURN_TYPES = (
        "UNET3D",
    )
    RETURN_NAMES = (
        "unet3d",
    )
    FUNCTION = "load_unet3d"

    CATEGORY = "DynamicPose/loaders"

    def load_unet3d(self, type, pretrained_base_unet_folder_path, unet3d_model_path, motion_module_path):

        if not os.path.isabs(pretrained_base_unet_folder_path):
            pretrained_base_unet_folder_path = os.path.join(ROOT_PATH, pretrained_base_unet_folder_path)     
        if not os.path.isabs(unet3d_model_path):
            unet3d_model_path = os.path.join(ROOT_PATH, unet3d_model_path)
        if not os.path.isabs(motion_module_path):
            motion_module_path = os.path.join(ROOT_PATH, motion_module_path)
        
        if type == 'video':
            unet3d = UNet3DConditionModel.from_pretrained_2d(
                pretrained_base_unet_folder_path,
                motion_module_path,
                unet_additional_kwargs=CONFIG2.unet_additional_kwargs,
            ).to(dtype=WEIGHT_DETYPE, device=DEVICE)
        else:
            unet3d = UNet3DConditionModel.from_pretrained_2d(
                pretrained_base_unet_folder_path,
                "",
                unet_additional_kwargs=CONFIG1.unet_additional_kwargs,
            ).to(dtype=WEIGHT_DETYPE, device=DEVICE)           
        
        unet3d.load_state_dict(
            torch.load(unet3d_model_path, map_location="cpu"),
            strict=False,
        )
        
        return (unet3d,)
    
class Load_Pose_Guider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_guider_model_path": ("STRING", {"default": "./pretrained_weights/pose_guider.pth", "multiline": False}),
            },
        }

    RETURN_TYPES = (
        "POSE_GUIDER",
    )
    RETURN_NAMES = (
        "pose_guider",
    )
    FUNCTION = "load_pose_guider"

    CATEGORY = "DynamicPose/loaders"
    
    def load_pose_guider(self, pose_guider_model_path):
        if not os.path.isabs(pose_guider_model_path):
            pose_guider_model_path = os.path.join(ROOT_PATH, pose_guider_model_path)
        
        pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
            dtype=WEIGHT_DETYPE, device=DEVICE
        )
        pose_guider.load_state_dict(
            torch.load(pose_guider_model_path, map_location="cpu"),
        )
        
        return (pose_guider,)
    
class Pose_Guider_Encode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_guider": ("POSE_GUIDER",),
                "pose_images": ("IMAGE",), 
            },
            "optional":{
                "height": ("INT",{"default": 784, "min": 100, "max": 4000}),
                "width": ("INT",{"default": 512, "min": 100, "max": 4000}),
            }
        }

    RETURN_TYPES = (
        "POSE_LATENT",
    )
    RETURN_NAMES = (
        "pose_latent",
    )
    FUNCTION = "pose_guider_encode"

    CATEGORY = "DynamicPose/processors"
    
    def pose_guider_encode(self, pose_guider, pose_images,height, width):
        
        cond_image_processor = VaeImageProcessor(
            do_convert_rgb=True,
            do_normalize=False,
        )
        
        # (b, h, w, c) -> (b, c, h, w)
        pose_images = pose_images.permute(0, 3, 1, 2).to(DEVICE, dtype=WEIGHT_DETYPE)
        
        # Prepare a list of pose condition images
        pose_cond_tensor_list = []
        for pose_image in pose_images:
            pose_cond_tensor = cond_image_processor.preprocess(
                pose_image, height=height, width=width
            )
            pose_cond_tensor = pose_cond_tensor.unsqueeze(2)  # (b, c, 1, h, w)
            pose_cond_tensor_list.append(pose_cond_tensor)

        pose_cond_tensor = torch.cat(pose_cond_tensor_list, dim=2)  # (b, c, f, h, w)
        pose_cond_tensor = pose_cond_tensor.to(
            device=DEVICE, dtype=pose_guider.dtype
        )
        pose_latent = pose_guider(pose_cond_tensor)
        #print(f"pose_cond_tensor.shape: {pose_cond_tensor.shape}\pose_latent.shape: {pose_latent.shape}")
        #pose_cond_tensor.shape: torch.Size([1, 3, 24, 768, 512]) pose_latent.shape: torch.Size([1, 320, 24, 96, 64])
        
        return (pose_latent,)

class DynamicPose_Sampler:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_unet": ("UNET2D",),
                "denoising_unet": ("UNET3D",),
                "ref_image_latent": ("LATENT",),
                "clip_image_embeds": ("CLIP_VISION_OUTPUT",),
                "pose_latent": ("POSE_LATENT",),
                "seed": ("INT", {"default": 999999999, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "delta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "context_frames": ("INT", {"default": 24, "min": 1}),
                "context_stride": ("INT", {"default": 1, "min": 1}),
                "context_overlap": ("INT", {"default": 4, "min": 0}),
                "context_batch_size": ("INT", {"default": 1, "min": 1}),
                "interpolation_factor": ("INT", {"default": 1, "min": 0}),
                "sampler_scheduler_pairs": (list(SCHEDULER_DICT.keys()),),
                "beta_start": ("FLOAT", {"default": 0.00085, "min": 0.0, "step":0.00001}),
                "beta_end": ("FLOAT", {"default": 0.012, "min": 0.0, "step":0.00001}),
                "beta_schedule": (["linear", "scaled_linear", "squaredcos_cap_v2"],),
                "prediction_type": (["v_prediction", "epsilon", "sample"],),
                "timestep_spacing": (["trailing", "linspace", "leading"],),
                "steps_offset": ("INT", {"default": 1, "min": 0, "max": 10000}),
            },
            "optional": {
                "clip_sample": ("BOOLEAN", {"default": False},),
                "rescale_betas_zero_snr": ("BOOLEAN", {"default": True},),
                "use_lora": ("BOOLEAN", {"default": False},),
                "lora_name": (comfy_paths.get_filename_list("loras"),),
            }
        }
        
    RETURN_TYPES = (
        "LATENT",
    )
    RETURN_NAMES = (
        "latent",
    )
    FUNCTION = "inferrence"

    CATEGORY = "DynamicPose"
    
    def inferrence(
        self, 
        reference_unet, 
        denoising_unet, 
        ref_image_latent, 
        clip_image_embeds, 
        pose_latent, 
        seed, 
        steps, 
        cfg,
        delta,
        context_frames,
        context_stride,
        context_overlap,
        context_batch_size,
        interpolation_factor,
        sampler_scheduler_pairs, 
        beta_start,
        beta_end,
        beta_schedule,
        prediction_type,
        timestep_spacing,
        steps_offset,
        clip_sample=False,
        rescale_betas_zero_snr=True,
        use_lora=False,
        lora_name=None
    ):
        
        latent_format = latent_formats.SD15()
        
        # encoder_hidden_states.shape: torch.Size([1, 1, 768]) clip_image_embeds.shape: torch.Size([1, 768])
        encoder_hidden_states = clip_image_embeds["image_embeds"].unsqueeze(1).to(DEVICE, dtype=WEIGHT_DETYPE)

        # forward reference image latent with shape (1, 4, 96, 64) to reference net
        ref_image_latent = latent_format.process_in(ref_image_latent["samples"]).to(DEVICE, dtype=WEIGHT_DETYPE)
        
        # setup scheduler from user inputs
        scheduler_class = SCHEDULER_DICT[sampler_scheduler_pairs]
        
        sched_kwargs = {
            "beta_start": beta_start,
            "beta_end": beta_end,
            "beta_schedule": beta_schedule,
            "steps_offset": steps_offset,
            "prediction_type": prediction_type,
            "timestep_spacing": timestep_spacing,
        }
        
        if "clip_sample" in inspect.signature(scheduler_class.__init__).parameters:
            sched_kwargs["clip_sample"] = clip_sample
        if "rescale_betas_zero_snr" in inspect.signature(scheduler_class.__init__).parameters:
            sched_kwargs["rescale_betas_zero_snr"] = rescale_betas_zero_snr
            
        scheduler = scheduler_class(**sched_kwargs)
        
        # setup diffuser and then denoise
        diffuser = DPDiffusion(reference_unet, denoising_unet, scheduler)
        
        if use_lora:
            lora_path = comfy_paths.get_full_path("loras", lora_name)
            diffuser.load_lora(lora_path)
        
        samples = diffuser(
            steps, 
            cfg,
            delta,
            ref_image_latent,
            pose_latent, 
            encoder_hidden_states,
            seed,
            context_frames=context_frames,
            context_stride=context_stride,
            context_overlap=context_overlap,
            context_batch_size=context_batch_size,
            interpolation_factor=interpolation_factor,
        )
        samples = latent_format.process_out(samples)
        
        # (1, 4, f, h, w) -> (f, 4, h, w)
        samples = torch.squeeze(samples, 0).permute(1, 0, 2, 3)
        
        return ({"samples":samples}, )

NODE_CLASS_MAPPINGS = {
    "pose_extraction" : pose_extraction,
    "Load_reference_unet" : Load_reference_unet,
    "Load_denoising_unet": Load_denoising_unet,
    "Load_Pose_Guider": Load_Pose_Guider,
    "Pose_Guider_Encode": Pose_Guider_Encode,
    "DynamicPose_Sampler": DynamicPose_Sampler,
    "load_pose_model": load_pose_model,
    "align": align,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "pose_extraction" : "pose_extraction",
    "Load_reference_unet" : 'Load_reference_unet',
    "Load_denoising_unet": 'Load_denoising_unet',
    "Load_Pose_Guider": 'Load_Pose_Guider',
    "Pose_Guider_Encode": 'Pose_Guider_Encode',
    "DynamicPose_Sampler": 'DynamicPose_Sampler',
    "load_pose_model": 'load_pose_model',
    "align":"align",
}
