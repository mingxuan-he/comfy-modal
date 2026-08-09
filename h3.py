"""MiniMax-H3 (PinkCherry) ComfyUI deployment on Modal.

Separate app/Volume from `main.py` (the image playground) so the two never share
weights, GPUs, or build cache.

    modal run   h3.py::download_models   # populate the Volume, no UI
    modal serve h3.py                    # ephemeral dev URL
    modal deploy h3.py                   # persistent URL
"""

import shutil
import subprocess
from pathlib import Path

import modal

APP_NAME = "comfy-minimax-h3"
VOLUME_NAME = "minimax-h3-models"

# Dedicated Volume: ~55 GB of H3 weights, kept apart from main.py's "hf-hub-cache".
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Gated/rate-limited HF downloads and runtime need a token. Values never get printed.
hf_secret = modal.Secret.from_name("huggingface-secret")

CACHE_DIR = "/cache"
COMFY_DIR = Path("/root/comfy/ComfyUI")
WORKFLOW_NAME = "minimax_h3_pinkcherry_turbo_t2va.json"
WORKFLOW_SRC = Path("/root/workflows") / WORKFLOW_NAME
WORKFLOW_DST_DIR = COMFY_DIR / "user" / "default" / "workflows"

# Exact filenames here are what the bundled workflow references - keep them in sync.
MODELS: list[dict[str, str]] = [
    # Base model: PinkCherry v0.5-alpha, pruned INT8 (~21 GB).
    # The Turbo node auto-detects a pruned base and re-injects time conditioning.
    {
        "repo_id": "SexGod1979/PinkCherry_MiniMax-H3",
        "filename": "alpha-0.5-testing/PinkCherry_h3_fl2va_pruned_int8_v0.5-alpha.safetensors",
        "target_name": "PinkCherry_h3_fl2va_pruned_int8_v0.5-alpha.safetensors",
        "sub_dir": "diffusion_models",
    },
    # Text encoder: Qwen3-VL 32B, INT8 convrot (~27 GB).
    {
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "target_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "sub_dir": "text_encoders",
    },
    # H3 generates video and native stereo audio jointly, so both VAEs are required.
    {
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "vae/minimax_h3_video_vae_fp16.safetensors",
        "target_name": "minimax_h3_video_vae_fp16.safetensors",
        "sub_dir": "vae",
    },
    {
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "target_name": "minimax_h3_audio_vae_fp32.safetensors",
        "sub_dir": "vae",
    },
    # Turbo LoRA, current recommended checkpoint - the workflow default.
    # https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora
    {
        "repo_id": "larryvrh/MiniMax-H3-Turbo-Lora",
        "filename": "minimax_h3_turbo_v4_step600_ema.safetensors",
        "target_name": "minimax_h3_turbo_v4_step600_ema.safetensors",
        "sub_dir": "loras",
    },
    # v1 (~850) EMA: friendlier than v4 at 4 steps with heavy/fast motion, kept for comparison.
    {
        "repo_id": "larryvrh/MiniMax-H3-Turbo-Lora",
        "filename": "minimax_h3_turbo_4step_ema_ckpt850.safetensors",
        "target_name": "minimax_h3_turbo_4step_ema_ckpt850.safetensors",
        "sub_dir": "loras",
    },
]


def hf_download():
    """Download every model into the Volume and symlink it into ComfyUI's model dirs."""
    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    models_dir = COMFY_DIR / "models"

    for model in MODELS:
        cached_path = hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            cache_dir=CACHE_DIR,
        )
        print(f"Downloaded {model['repo_id']}/{model['filename']} to {cached_path}")

        target_dir = models_dir / model["sub_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / model["target_name"]
        # A dangling symlink reports False from exists(), so check the link itself.
        if target_path.is_symlink() or target_path.exists():
            target_path.unlink()
        target_path.symlink_to(cached_path)
        print(f"Linked {target_path} -> {cached_path}")


def install_workflow():
    """Copy the bundled workflow into ComfyUI's user workflow directory.

    Runs at build time and again at container start, so an updated JSON lands even
    when the rest of the image layer is cached.
    """
    WORKFLOW_DST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WORKFLOW_SRC, WORKFLOW_DST_DIR / WORKFLOW_NAME)

    # ComfyUI runs single-user here and reads workflows from user/default/, but it
    # only writes users.json under --multi-user. Writing it keeps the "default"
    # profile explicit so the sidebar always resolves that directory.
    users_file = COMFY_DIR / "user" / "users.json"
    if not users_file.exists():
        users_file.write_text('{"default": "default"}\n')

    print(f"Installed workflow at {WORKFLOW_DST_DIR / WORKFLOW_NAME}")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install("fastapi[standard]==0.115.4")
    .uv_pip_install("comfy-cli==1.15.0")
    # nightly = latest master; MiniMax-H3 support and ModelSamplingAV are recent.
    .run_commands(
        "comfy --skip-prompt install --fast-deps --skip-manager --nvidia --cuda-version 13.0 --version nightly"
    )
    # Turbo node currently has no Python dependencies. Install it directly from
    # the upstream commit. Running Comfy-Manager's installer here is actively
    # harmful: even when registry lookup fails, its dependency restore re-resolves
    # unpinned `torch` and silently replaces the requested cu130 build with cu126.
    .run_commands(
        "git clone https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo "
        "/root/comfy/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo && "
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo && "
        "git checkout 55fee864dd7b2976b1c4ce3c3d5f7968f181409f"
    )
    # Do not override ComfyUI's resolved transformers / huggingface-hub pair.
    # Current nightly resolves them together; independently pinning Hub caused an
    # import-time ABI mismatch (`is_offline_mode` missing). Fail the image build
    # immediately if the pair cannot import.
    .run_commands(
        "python -c \"import torch, transformers, huggingface_hub; "
        "from transformers import CLIPTokenizer; "
        "print('torch', torch.__version__, 'cuda', torch.version.cuda, "
        "'transformers', transformers.__version__, 'hub', huggingface_hub.__version__)\""
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(
        hf_download,
        volumes={CACHE_DIR: vol},
        secrets=[hf_secret],
        timeout=60 * 60 * 4,  # ~55 GB on a cold Volume
    )
    .add_local_file(
        Path(__file__).parent / "workflows" / WORKFLOW_NAME,
        WORKFLOW_SRC.as_posix(),
        copy=True,  # needed during the build so the next step can copy it
    )
    .run_function(install_workflow)
)

app = modal.App(name=APP_NAME, image=image)


@app.function(
    volumes={CACHE_DIR: vol},
    secrets=[hf_secret],
    timeout=60 * 60 * 4,
)
def download_models():
    """Pre-download / re-sync weights into the Volume without starting the UI.

    `modal run h3.py::download_models`
    """
    hf_download()
    vol.commit()  # persist before the container exits
    print("Volume committed.")


@app.function(
    max_containers=1,  # one interactive container
    gpu="H100",  # 80 GB - comfortable for the 33B base at 1344x768
    volumes={CACHE_DIR: vol},
    secrets=[hf_secret],
    timeout=60 * 60,  # long enough for multi-clip sessions
    scaledown_window=60 * 5,
)
@modal.concurrent(max_inputs=10)  # required for UI startup
@modal.web_server(8000, startup_timeout=300)
def ui():
    install_workflow()
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
