# import os
# import numpy as np
# import json
# import math
# from PIL import Image
# import cv2
# from mmpose.apis import MMPoseInferencer
# from moviepy.editor import VideoFileClip
# import sys
# pwd=os.getcwd()
# sys.path.append(pwd)

# from .src.dwpose import DWposeDetector
# from tqdm import tqdm


class pose_extraction:
    @classmethod
    def INPUT_TYPES(s):
        return{
            "required":{
                "Type": (["Image","Video"]),
                "Body_model": ("STRING", {"defualt": "rtmpose"}),
            }
        }
    RETURN_TYPES = (
        "Image", "Video"
    )
    RETURN_NAMES = (
        "Image","Video"
    )

    FUNCTION = "main"

    CATEGORY = "DynamicPose"
    
    def main(Type,Body_model):
        return 1, 1