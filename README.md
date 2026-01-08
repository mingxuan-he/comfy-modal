# comfy-modal

ComfyUI running on Modal, with cached model weights stored in a Modal Volume.

## Run

```bash
modal serve main.py
```

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

Drag a JSON into the ComfyUI canvas to load it.

