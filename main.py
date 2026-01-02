import subprocess
from pathlib import Path

import modal

# Create a Volume to persist model cache
vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


def hf_download():
    """Download models from Hugging Face and symlink them to ComfyUI directories."""
    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    comfyui_dir = Path("/root/comfy/ComfyUI")
    models_dir = comfyui_dir / "models"

    # Cache both model families into the Volume once; subsequent starts should hit the cache.
    #
    # Qwen-Image-2512 in ComfyUI uses the ComfyUI-native weights repo:
    # - https://docs.comfy.org/tutorials/image/qwen/qwen-image-2512
    # - https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI
    models: list[dict[str, str]] = [
        # z-image-turbo: https://huggingface.co/Comfy-Org/z_image_turbo
        {
            "repo_id": "Comfy-Org/z_image_turbo",
            "filename": "split_files/text_encoders/qwen_3_4b.safetensors",
            "target_name": "qwen_3_4b.safetensors",
            "sub_dir": "text_encoders",
        },
        {
            "repo_id": "Comfy-Org/z_image_turbo",
            "filename": "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
            "target_name": "z_image_turbo_bf16.safetensors",
            "sub_dir": "diffusion_models",
        },
        {
            "repo_id": "Comfy-Org/z_image_turbo",
            "filename": "split_files/vae/ae.safetensors",
            "target_name": "ae.safetensors",
            "sub_dir": "vae",
        },
        # Qwen-Image-2512 (fp8) + VAE + text encoder
        {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "target_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "sub_dir": "text_encoders",
        },
        # Qwen-Image-Edit-2511 docs point the *same-named* text encoder at a repackaged repo:
        # https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511
        {
            "repo_id": "Comfy-Org/HunyuanVideo_1.5_repackaged",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "target_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "sub_dir": "text_encoders",
        },
        {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors",
            "target_name": "qwen_image_2512_fp8_e4m3fn.safetensors",
            "sub_dir": "diffusion_models",
        },
        {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/vae/qwen_image_vae.safetensors",
            "target_name": "qwen_image_vae.safetensors",
            "sub_dir": "vae",
        },
        # Optional but useful for the "4 steps" workflow (small download)
        {
            "repo_id": "lightx2v/Qwen-Image-Lightning",
            "filename": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
            "target_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
            "sub_dir": "loras",
        },
        # Qwen-Image-Edit-2511 (edit model + optional lightning LoRA)
        # https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511
        {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
            "target_name": "qwen_image_edit_2511_bf16.safetensors",
            "sub_dir": "diffusion_models",
        },
        {
            "repo_id": "lightx2v/Qwen-Image-Edit-2511-Lightning",
            "filename": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "target_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "sub_dir": "loras",
        },
    ]

    for model in models:
        # Download to cache
        cached_path = hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            cache_dir="/cache",
        )
        print(f"Downloaded {model['filename']} to {cached_path}")

        # Create target directory
        target_dir = models_dir / model["sub_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)

        # Symlink to ComfyUI directory
        target_path = target_dir / model["target_name"]
        if not target_path.exists():
            target_path.symlink_to(cached_path)


# Build the Modal image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install("fastapi[standard]==0.115.4")
    .uv_pip_install("comfy-cli==1.5.3")
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia")
    .uv_pip_install("huggingface-hub==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(
        hf_download,
        volumes={"/cache": vol},
    )
)

app = modal.App(name="comfy-image-playground", image=image)


@app.function(
    max_containers=1,  # limit interactive session to 1 container
    gpu="L40S",  # good starter GPU for inference
    volumes={"/cache": vol},  # mount cached models
)
@modal.concurrent(max_inputs=10)  # required for UI startup
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
