# MiniMax H3 PinkCherry Modal Deployment

## Objective
Add a separate Modal-hosted ComfyUI deployment for MiniMax-H3 R18 generation using PinkCherry v0.5 pruned INT8 and LarryVRH Turbo LoRA. Preserve the existing image playground deployment.

## Requirements
1. Add a dedicated Python entrypoint (prefer `h3.py`) and Modal app/Volume; do not overload or break existing `main.py`.
2. GPU: H100 80GB. One interactive container max, sensible timeout/startup timeout.
3. Models must download into and persist in a dedicated Modal Volume, then symlink into ComfyUI model directories:
   - `SexGod1979/PinkCherry_MiniMax-H3`, `alpha-0.5-testing/PinkCherry_h3_fl2va_pruned_int8_v0.5-alpha.safetensors` -> diffusion_models
   - `Comfy-Org/MiniMax-H3`, `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` -> text_encoders
   - `Comfy-Org/MiniMax-H3`, video and audio VAEs -> vae
   - `larryvrh/MiniMax-H3-Turbo-Lora`, recommended current `minimax_h3_turbo_v4_step600_ema.safetensors` -> loras
   - Also include v1-850 EMA for 4-step heavy motion comparison.
4. Use Modal secret `huggingface-secret` for downloads/runtime. Never print or commit secret values.
5. Install latest ComfyUI and `Larryvrh/ComfyUI-MiniMax-H3-Turbo`; install required dependencies. Follow Ming's preference for comfy-cli installation.
6. Bundle a valid Turbo T2VA workflow based on upstream `minimax_h3_t2v_turbo.json`, modifying model filenames to the exact PinkCherry/encoder/VAE/LoRA names above. Default to 6 steps, scheduler simple, LoRA strength 1.0, 1344x768, 124 frames unless upstream node semantics require a safer exact adjustment.
7. Put workflow in `ComfyUI/user/default/workflows/` at image build and copy it again at runtime. Create `user/users.json` as needed so the workflow appears in ComfyUI's workflow sidebar and can be loaded without dragging JSON. Also keep it under this repo's `workflows/`.
8. Add an explicit Modal function/CLI path to pre-download or re-sync model files into the Volume without launching the UI, if practical. Ensure `vol.commit()` after downloads where required by Modal semantics.
9. Update README with exact serve/deploy/download commands, app/volume names, URL behavior, expected first-run download size, model choices, and workflow location.
10. Keep code importable/syntax-valid locally without requiring GPU execution.

## Verification
- `python3 -m py_compile main.py h3.py` passes.
- Workflow parses as JSON.
- Inspect workflow nodes and prove all referenced checkpoint filenames exactly match files created by `h3.py`.
- No secrets or token values in git diff.
- Existing `main.py` remains functional and minimally changed or unchanged.
- Commit changes locally with a clear commit message. Do not deploy or push yet; Yui will review first.

## Boundaries
- Never reveal, print, or commit Modal/Hugging Face credentials.
- Do not delete existing models/workflows/deployment.
- Do not push or deploy.
