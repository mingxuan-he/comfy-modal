# comfy-modal

ComfyUI running on Modal, with cached model weights stored in a Modal Volume.

## Run

```bash
modal serve main.py
```

## Workflows

Workflow JSONs are in `workflows/`:

- `workflows/image_z_image_turbo.json`
- `workflows/image_qwen_Image_2512.json`
- `workflows/image_qwen_image_edit_2511.json`

Drag a JSON into the ComfyUI canvas to load it.

