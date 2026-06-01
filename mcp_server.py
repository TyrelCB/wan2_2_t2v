from pathlib import Path

from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

import comfy_client

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

mcp = FastMCP("wan2-t2v")


@mcp.tool()
def generate_wan_video(
    prompt: str,
    width: int = 848,
    height: int = 480,
    length: int = 81,
    seed: int = -1,
    fps: int = 16,
) -> dict:
    """Generate a video from a text prompt using Wan 2.2 14B via ComfyUI.

    Args:
        prompt: Text description of the video to generate.
        width: Frame width in pixels (default 848).
        height: Frame height in pixels (default 480).
        length: Number of frames to generate (default 81, ~5s at 16fps).
        seed: Random seed; -1 for a random seed.
        fps: Output video frames per second (default 16).

    Returns:
        dict with 'path' (absolute path to saved video) and 'filename'.
    """
    resolved_seed = None if seed < 0 else seed
    video_bytes, filename = comfy_client.generate(
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        seed=resolved_seed,
        fps=fps,
    )
    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(video_bytes)
    return {"path": str(out_path.resolve()), "filename": filename}


CORS = [Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])]

@mcp.tool()
def generate_wan_i2v(
    image_path: str,
    prompt: str,
    width: int = 512,
    height: int = 512,
    length: int = 81,
    seed: int = -1,
    fps: int = 16,
) -> dict:
    """Generate a video from a start image and text prompt using Wan 2.2 14B I2V via ComfyUI.

    Args:
        image_path: Absolute path to the start frame image file.
        prompt: Text description of the motion to generate.
        width: Frame width in pixels (default 512).
        height: Frame height in pixels (default 512).
        length: Number of frames (default 81, ~5s at 16fps).
        seed: Random seed; -1 for a random seed.
        fps: Output video frames per second (default 16).

    Returns:
        dict with 'path' (absolute path to saved video) and 'filename'.
    """
    resolved_seed = None if seed < 0 else seed
    image_bytes = Path(image_path).read_bytes()
    image_filename = Path(image_path).name
    video_bytes, filename = comfy_client.generate_i2v(
        image_bytes=image_bytes, image_filename=image_filename,
        prompt=prompt, width=width, height=height,
        length=length, seed=resolved_seed, fps=fps,
    )
    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(video_bytes)
    return {"path": str(out_path.resolve()), "filename": filename}


@mcp.tool()
def generate_wan_s2v(
    image_path: str,
    audio_path: str,
    prompt: str = "",
    seed: int = -1,
    fps: int = 16,
    chunk_length: int = 43,
) -> dict:
    """Generate a talking-head video from a reference image and audio using Wan 2.2 14B S2V via ComfyUI.

    Args:
        image_path: Absolute path to the reference face image.
        audio_path: Absolute path to the audio file (WAV recommended).
        prompt: Optional text prompt (default empty).
        seed: Random seed; -1 for a random seed.
        fps: Output video frames per second (default 16).
        chunk_length: Frames per generation chunk (default 43). Lower = less VRAM.

    Returns:
        dict with 'path' (absolute path to saved video) and 'filename'.
    """
    resolved_seed = None if seed < 0 else seed
    image_bytes = Path(image_path).read_bytes()
    audio_bytes = Path(audio_path).read_bytes()
    video_bytes, filename = comfy_client.generate_s2v(
        image_bytes=image_bytes, image_filename=Path(image_path).name,
        audio_bytes=audio_bytes, audio_filename=Path(audio_path).name,
        prompt=prompt, seed=resolved_seed, fps=fps, chunk_length=chunk_length,
    )
    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(video_bytes)
    return {"path": str(out_path.resolve()), "filename": filename}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000, middleware=CORS, stateless_http=True)
