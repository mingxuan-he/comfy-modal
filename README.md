# comfy-modal

ComfyUI running on Modal, with cached model weights stored in a Modal Volume.

Two independent deployments, each with its own Modal app and Volume:

| entrypoint | app | volume | GPU | purpose |
|---|---|---|---|---|
| `main.py` | `comfy-image-playground` | `hf-hub-cache` | L40S | image + Hunyuan video playground |
| `h3.py` | `comfy-minimax-h3` | `minimax-h3-models` | H100 80GB | MiniMax-H3 PinkCherry audio-video |

## Run

```bash
modal serve main.py        # image playground
modal serve h3.py          # MiniMax-H3
```

`modal serve` prints an ephemeral dev URL that stays alive while the command runs
and rebuilds on file changes. `modal deploy <file>` gives a persistent URL that
survives the terminal; the container still scales to zero when idle and cold-starts
on the next request.

## MiniMax-H3 (`h3.py`)

R18 audio-video generation with [PinkCherry v0.5-alpha](https://huggingface.co/SexGod1979/PinkCherry_MiniMax-H3)
(a MiniMax-H3 finetune) plus the [larryvrh Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
for few-step sampling. H3 generates video and native stereo audio jointly in one pass.

```bash
modal run    h3.py::download_models   # populate the Volume, no UI, no GPU
modal serve  h3.py                    # ephemeral dev URL
modal deploy h3.py                    # persistent URL
```

Requires a Modal secret named `huggingface-secret` holding an HF token
(`HF_TOKEN`). It is injected into the download and runtime containers; nothing is
printed or committed.

### Models

Downloaded into the `minimax-h3-models` Volume and symlinked into ComfyUI's model
directories. **First run downloads ~55 GB (~52 GiB)**; after that the Volume is
warm and startup skips it. Weights are fetched during the image build, so the
first `modal serve`/`deploy` is the slow one — or run `download_models` first.

| file | size | source | ComfyUI dir |
|---|---|---|---|
| `PinkCherry_h3_fl2va_pruned_int8_v0.5-alpha.safetensors` | 21.0 GB | `SexGod1979/PinkCherry_MiniMax-H3` (`alpha-0.5-testing/`) | `diffusion_models` |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27.1 GB | `Comfy-Org/MiniMax-H3` | `text_encoders` |
| `minimax_h3_video_vae_fp16.safetensors` | 5.2 GB | `Comfy-Org/MiniMax-H3` | `vae` |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB | `Comfy-Org/MiniMax-H3` | `vae` |
| `minimax_h3_turbo_v4_step600_ema.safetensors` | 0.8 GB | `larryvrh/MiniMax-H3-Turbo-Lora` | `loras` |
| `minimax_h3_turbo_4step_ema_ckpt850.safetensors` | 0.8 GB | `larryvrh/MiniMax-H3-Turbo-Lora` | `loras` |

Why these:

- **Pruned INT8 base** (~21 GB) instead of bf16 (~66 GB) or full int8 (~34 GB) — it
  fits an H100 80GB alongside the 27 GB text encoder with headroom for a 1344x768
  clip. The Turbo node auto-detects a pruned base and re-injects the collapsed time
  conditioning at run time, so the LoRA works unchanged.
- **INT8 text encoder** rather than bf16 (52 GB), which would not fit next to the base.
- **Both VAEs** are required: video and audio are decoded separately, then muxed.
- **`v4_step600_ema`** is the LoRA author's current recommendation and the workflow
  default. **`4step_ema_ckpt850`** (the older v1 line) is also downloaded because it
  is the friendlier pick at 4 steps with heavy/fast motion; swap it in the
  `MiniMax-H3 Turbo LoRA` node to compare.

### Custom nodes

[`Larryvrh/ComfyUI-MiniMax-H3-Turbo`](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)
is installed via `comfy node install` from the Comfy Registry, falling back to a
git clone if the registry copy is missing the bundled `h3_silu_temb_grid.safetensors`
that pruned bases need.

### Workflow

`workflows/minimax_h3_pinkcherry_turbo_t2va.json`, derived from upstream
`minimax_h3_t2v_turbo.json` with the model filenames pointed at the files above. The image uses current ComfyUI nightly, lets ComfyUI resolve its matching `transformers`/`huggingface-hub` pair, and explicitly installs the CUDA 13.0 PyTorch stack recommended by current ComfyUI for optimized kernels.

It is baked into the image at `ComfyUI/user/default/workflows/` and copied there
again at container start, so it shows up directly in ComfyUI's **Workflows**
sidebar — no dragging JSON onto the canvas.

Defaults: **6 steps**, scheduler `simple`, LoRA strength **1.0**, `low_vram` off
(bypass), **1344x768**, **124 frames** (~5s at 24fps, via duration `5.0` snapped to
the model's 17k+5 grid).

Tuning, per the LoRA model card:

- 4-8 steps is the useful range; 6-8 looks best. Past 8 it over-sharpens.
- Keep strength at 1.0. Blurry ghosting -> ~1.05-1.2; over-sharp grain -> ~0.8-0.95.
- Keep the scheduler on `simple`.
- Turn `low_vram` on only if you OOM (it merges the LoRA — lower peak VRAM, softer
  on quantized bases).
- Frame count is validated ~124-362 (~5-15s); short edge is typically 768.

## Workflows

Workflow JSONs are in `workflows/`:

### Image Generation
- `workflows/image_z_image_turbo.json` - z-image-turbo model
- `workflows/image_qwen_Image_2512.json` - Qwen-Image-2512 model
- `workflows/image_qwen_image_edit_2511.json` - Qwen-Image-Edit-2511 model
- `workflows/image_z_image_turbo_fun_union_controlnet.json` - z-image-turbo with ControlNet

### Video Generation
- `workflows/hunyuan_video_t2v_720p.json` - Hunyuan Video Text-to-Video (720p)
- `workflows/hunyuan_video_i2v.json` - Hunyuan Video Image-to-Video
- `workflows/hunyuan_video_v2v.json` - Hunyuan Video Video-to-Video
- `workflows/minimax_h3_pinkcherry_turbo_t2va.json` - MiniMax-H3 PinkCherry + Turbo LoRA (`h3.py`)

Drag a JSON into the ComfyUI canvas to load it (the MiniMax-H3 one is already in
the sidebar).
