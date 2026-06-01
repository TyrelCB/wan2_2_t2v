# Wan 2.2 14B Text-to-Video

Gradio UI and MCP server for generating video from text using the [Wan 2.2 14B](https://github.com/Wan-Video/Wan2.1) model via ComfyUI.

Uses a dual-pass LightX2V sampling approach (high-noise → low-noise UNETs) to generate 848×480 video in 4 steps.

## Requirements

- ComfyUI running with the Wan 2.2 14B models loaded
- Python 3.10+

```
pip install -r requirements.txt
```

## Models

Place these in your ComfyUI model directories:

| File | Directory |
|------|-----------|
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models/clip/` |
| `wan_2.1_vae.safetensors` | `models/vae/` |
| `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | `models/unet/` |
| `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | `models/unet/` |
| `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` | `models/loras/` |
| `wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors` | `models/loras/` |

## Configuration

Set the `COMFY_URL` environment variable to point at your ComfyUI instance (default: `http://192.168.6.181:8188`):

```bash
export COMFY_URL=http://your-comfyui-host:8188
```

## Gradio UI

```bash
python app.py
```

Open `http://localhost:7860`. Enter a prompt, adjust resolution/frames/seed/FPS, and click **Generate**. The status bar shows the resolved seed and generation time.

## MCP Server

```bash
python mcp_server.py
```

Runs a FastMCP streamable-HTTP server on `http://0.0.0.0:8000/mcp`.

Add it to your MCP client with the URL `http://<host>:8000/mcp`.

### Tool: `generate_wan_video`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | — | Text description of the video |
| `width` | int | 848 | Frame width in pixels |
| `height` | int | 480 | Frame height in pixels |
| `length` | int | 81 | Number of frames (~5s at 16fps) |
| `seed` | int | -1 | Random seed (-1 = random) |
| `fps` | int | 16 | Output frames per second |

Returns `{"path": "/absolute/path/to/video.mp4", "filename": "..."}`.

Videos are saved to `./outputs/`.
