# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import torch
from typing import Optional
from PIL import Image # ImageSequence is not used

from cog import BasePredictor, Input, Path

# Diffusers imports
from diffusers import (
    LTXPipeline,
    LTXImageToVideoPipeline,
    LTXVideoTransformer3DModel,
    AutoencoderKLLTXVideo
)
from diffusers.utils import export_to_video
# Transformers imports for T5 components are not strictly needed here
# if relying on from_pretrained to load them as part of the pipeline.
# from transformers import T5EncoderModel, T5Tokenizer

MODEL_CACHE = "model_cache"
BASE_URL = "https://weights.replicate.delivery/default/lightricks/ltx-video"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")





class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
            print(f"Created cache directory: {MODEL_CACHE}")

        dtype = torch.bfloat16  # LTX-Video recommended dtype

        # Load base pipelines to get compatible schedulers, tokenizers, and text encoders
        print("Loading base TTV pipeline from Lightricks/LTX-Video...")
        self.ttv_pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video", torch_dtype=dtype, cache_dir=MODEL_CACHE
        )
        print("Loading base I2V pipeline from Lightricks/LTX-Video...")
        self.i2v_pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video", torch_dtype=dtype, cache_dir=MODEL_CACHE
        )

        # Define the specific 0.9.7 model file URL on Hugging Face Hub
        model_file_hf_url = "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.7-dev.safetensors"

        # Load the specific 0.9.7 VAE and Transformer components directly from HF Hub URL
        # diffusers will handle downloading and caching to MODEL_CACHE
        print(f"Loading VAE v0.9.7 from HF Hub: {model_file_hf_url}...")
        vae_097 = AutoencoderKLLTXVideo.from_single_file(
            model_file_hf_url, torch_dtype=dtype, cache_dir=MODEL_CACHE
        )
        # Replace the VAE in the loaded pipelines
        self.ttv_pipe.vae = vae_097
        self.i2v_pipe.vae = vae_097
        print("VAE v0.9.7 loaded and replaced in pipelines.")

        print(f"Loading Transformer v0.9.7 from HF Hub: {model_file_hf_url}...")
        transformer_097 = LTXVideoTransformer3DModel.from_single_file(
            model_file_hf_url, torch_dtype=dtype, cache_dir=MODEL_CACHE
        )
        # Replace the Transformer in the loaded pipelines
        self.ttv_pipe.transformer = transformer_097
        self.i2v_pipe.transformer = transformer_097
        print("Transformer v0.9.7 loaded and replaced in pipelines.")
        
        # The text_encoder and tokenizer are inherited from the base pipelines loaded above.
        # The scheduler is also part of the pipeline object.

        self.ttv_pipe.to("cuda")
        self.i2v_pipe.to("cuda")

        self.export_to_video = export_to_video
        print("Setup complete. LTX-Video 0.9.7 (13B) components configured from Hugging Face Hub.")

    def predict(
        self,
        prompt: str = Input(description="Text prompt for video generation"),
        input_image: Optional[Path] = Input(
            description="Input image for image-to-video generation. If not provided, text-to-video generation will be used.",
            default=None
        ),
        negative_prompt: str = Input(
            description="Negative prompt for video generation.",
            default="worst quality, inconsistent motion, blurry, jittery, distorted"
        ),
        width: int = Input(
            description="Width of the output video. Actual width will be a multiple of 32.",
            default=704
        ),
        height: int = Input(
            description="Height of the output video. Actual height will be a multiple of 32.",
            default=480
        ),
        num_frames: int = Input(
            description="Number of frames to generate. Actual frame count will be 8N+1 (e.g., 9, 17, 25, 161).",
            default=161
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps.",
            default=50
        ),
        guidance_scale: float = Input(
            description="Guidance scale. Recommended range: 3.0-3.5.",
            default=3.0,
            ge=1.0,
            le=10.0
        ),
        fps: int = Input(
            description="Frames per second for the output video.",
            default=24
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible results. Leave blank for a random seed.",
            default=None
        ),
    ) -> Path:
        """Generate a video from a text prompt or an image+text prompt"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Adjust dimensions according to LTX-Video requirements
        # Width and height must be divisible by 32
        processed_width = (width // 32) * 32
        processed_height = (height // 32) * 32
        if processed_width == 0: processed_width = 32
        if processed_height == 0: processed_height = 32
        
        # Number of frames must be 8N+1
        # ((N-1)//F)*F + 1 ensures result >= 1 and is of the form K*F+1
        processed_num_frames = ((num_frames - 1) // 8) * 8 + 1
        if processed_num_frames <= 0: # Ensure at least 1 frame, typically 8*0+1 = 1
            processed_num_frames = 1


        print(f"Original inputs: width={width}, height={height}, num_frames={num_frames}")
        print(f"Processed inputs: width={processed_width}, height={processed_height}, num_frames={processed_num_frames}")

        video_frames_tensor = None

        if input_image:
            print(f"[~] Mode: Image-to-Video")
            print(f"[~] Loading input image from: {input_image}")
            try:
                pil_image = Image.open(str(input_image)).convert("RGB")
                pil_image = pil_image.resize((processed_width, processed_height), Image.Resampling.LANCZOS)
                print(f"[~] Resized input image to: {processed_width}x{processed_height}")
            except Exception as e:
                raise ValueError(f"Failed to load or resize input image: {e}")

            print(f"[~] Generating video with prompt: '{prompt}'")
            video_frames_tensor = self.i2v_pipe(
                prompt=prompt,
                image=pil_image,
                negative_prompt=negative_prompt,
                width=processed_width,
                height=processed_height,
                num_frames=processed_num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
        else:
            print(f"[~] Mode: Text-to-Video")
            print(f"[~] Generating video with prompt: '{prompt}'")
            video_frames_tensor = self.ttv_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=processed_width,
                height=processed_height,
                num_frames=processed_num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]

        # Define output path
        output_video_path_str = "output.mp4"

        print(f"[~] Exporting video to {output_video_path_str} with {fps} FPS...")
        # export_to_video expects video_frames_tensor to be (num_frames, height, width, channels)
        # or a list of PIL Images. LTXPipeline output is typically compatible.
        self.export_to_video(video_frames_tensor, output_video_path_str, fps=fps)

        print(f"[+] Video generation complete: {output_video_path_str}")
        return Path(output_video_path_str)
