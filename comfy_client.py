import copy
import os
import random
import time

import requests

COMFY_URL = os.environ.get("COMFY_URL", "http://192.168.6.181:8188").rstrip("/")

_NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走，裸露，NSFW"
)

# ── T2V ───────────────────────────────────────────────────────────────────────

BASE_WORKFLOW = {
    "71": {"inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}, "class_type": "CLIPLoader"},
    "72": {"inputs": {"text": _NEG, "clip": ["71", 0]}, "class_type": "CLIPTextEncode"},
    "73": {"inputs": {"vae_name": "wan_2.1_vae.safetensors"}, "class_type": "VAELoader"},
    "74": {"inputs": {"width": 848, "height": 480, "length": 81, "batch_size": 1}, "class_type": "EmptyHunyuanLatentVideo"},
    "75": {"inputs": {"unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "76": {"inputs": {"unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "78": {"inputs": {"add_noise": "disable", "noise_seed": 0, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 2, "end_at_step": 4, "return_with_leftover_noise": "disable", "model": ["86", 0], "positive": ["89", 0], "negative": ["72", 0], "latent_image": ["81", 0]}, "class_type": "KSamplerAdvanced"},
    "80": {"inputs": {"filename_prefix": "video/ComfyUI", "format": "auto", "codec": "auto", "video-preview": "", "video": ["88", 0]}, "class_type": "SaveVideo"},
    "81": {"inputs": {"add_noise": "enable", "noise_seed": 687304397804133, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable", "model": ["82", 0], "positive": ["89", 0], "negative": ["72", 0], "latent_image": ["74", 0]}, "class_type": "KSamplerAdvanced"},
    "82": {"inputs": {"shift": 5.0, "model": ["83", 0]}, "class_type": "ModelSamplingSD3"},
    "83": {"inputs": {"lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", "strength_model": 1.0, "model": ["75", 0]}, "class_type": "LoraLoaderModelOnly"},
    "85": {"inputs": {"lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors", "strength_model": 1.0, "model": ["76", 0]}, "class_type": "LoraLoaderModelOnly"},
    "86": {"inputs": {"shift": 5.0, "model": ["85", 0]}, "class_type": "ModelSamplingSD3"},
    "87": {"inputs": {"samples": ["78", 0], "vae": ["73", 0]}, "class_type": "VAEDecode"},
    "88": {"inputs": {"fps": 16, "images": ["87", 0]}, "class_type": "CreateVideo"},
    "89": {"inputs": {"text": "", "clip": ["71", 0]}, "class_type": "CLIPTextEncode"},
}

# ── I2V ───────────────────────────────────────────────────────────────────────

_NEG_I2V = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

I2V_BASE_WORKFLOW = {
    "97":     {"inputs": {"image": ""}, "class_type": "LoadImage"},
    "108":    {"inputs": {"filename_prefix": "video/Wan2.2_image_to_video", "format": "auto", "codec": "auto", "video-preview": "", "video": ["116:94", 0]}, "class_type": "SaveVideo"},
    "116:84": {"inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}, "class_type": "CLIPLoader"},
    "116:90": {"inputs": {"vae_name": "wan_2.1_vae.safetensors"}, "class_type": "VAELoader"},
    "116:95": {"inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "116:96": {"inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "116:101": {"inputs": {"lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "strength_model": 1.0, "model": ["116:95", 0]}, "class_type": "LoraLoaderModelOnly"},
    "116:102": {"inputs": {"lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "strength_model": 1.0, "model": ["116:96", 0]}, "class_type": "LoraLoaderModelOnly"},
    "116:103": {"inputs": {"shift": 5.0, "model": ["116:102", 0]}, "class_type": "ModelSamplingSD3"},
    "116:104": {"inputs": {"shift": 5.0, "model": ["116:101", 0]}, "class_type": "ModelSamplingSD3"},
    "116:93": {"inputs": {"text": "", "clip": ["116:84", 0]}, "class_type": "CLIPTextEncode"},
    "116:89": {"inputs": {"text": _NEG_I2V, "clip": ["116:84", 0]}, "class_type": "CLIPTextEncode"},
    "116:98": {"inputs": {"width": 512, "height": 512, "length": 81, "batch_size": 1, "positive": ["116:93", 0], "negative": ["116:89", 0], "vae": ["116:90", 0], "start_image": ["97", 0]}, "class_type": "WanImageToVideo"},
    "116:86": {"inputs": {"add_noise": "enable", "noise_seed": 0, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable", "model": ["116:104", 0], "positive": ["116:98", 0], "negative": ["116:98", 1], "latent_image": ["116:98", 2]}, "class_type": "KSamplerAdvanced"},
    "116:85": {"inputs": {"add_noise": "disable", "noise_seed": 0, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 2, "end_at_step": 4, "return_with_leftover_noise": "disable", "model": ["116:103", 0], "positive": ["116:98", 0], "negative": ["116:98", 1], "latent_image": ["116:86", 0]}, "class_type": "KSamplerAdvanced"},
    "116:87": {"inputs": {"samples": ["116:85", 0], "vae": ["116:90", 0]}, "class_type": "VAEDecode"},
    "116:94": {"inputs": {"fps": 16, "images": ["116:87", 0]}, "class_type": "CreateVideo"},
}

# ── S2V (Talking Head) ────────────────────────────────────────────────────────

_NEG_S2V = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

S2V_BASE_WORKFLOW = {
    "3":   {"inputs": {"seed": 764053441674844, "steps": ["103", 0], "cfg": ["105", 0], "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1, "model": ["54", 0], "positive": ["93", 0], "negative": ["93", 1], "latent_image": ["93", 2]}, "class_type": "KSampler"},
    "6":   {"inputs": {"text": "", "clip": ["38", 0]}, "class_type": "CLIPTextEncode"},
    "7":   {"inputs": {"text": _NEG_S2V, "clip": ["38", 0]}, "class_type": "CLIPTextEncode"},
    "37":  {"inputs": {"unet_name": "wan2.2_s2v_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "38":  {"inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}, "class_type": "CLIPLoader"},
    "39":  {"inputs": {"vae_name": "wan_2.1_vae.safetensors"}, "class_type": "VAELoader"},
    "52":  {"inputs": {"image": ""}, "class_type": "LoadImage"},
    "54":  {"inputs": {"shift": 8, "model": ["107", 0]}, "class_type": "ModelSamplingSD3"},
    "56":  {"inputs": {"audio_encoder": ["57", 0], "audio": ["58", 0]}, "class_type": "AudioEncoderEncode"},
    "57":  {"inputs": {"audio_encoder_name": "wav2vec2_large_english_fp16.safetensors"}, "class_type": "AudioEncoderLoader"},
    "58":  {"inputs": {"audio": ""}, "class_type": "LoadAudio"},
    "80":  {"inputs": {"samples": ["95", 0], "vae": ["39", 0]}, "class_type": "VAEDecode"},
    "82":  {"inputs": {"fps": 16, "images": ["96", 0], "audio": ["58", 0]}, "class_type": "CreateVideo"},
    "93":  {"inputs": {"width": 512, "height": 512, "length": ["104", 0], "batch_size": 1, "positive": ["6", 0], "negative": ["7", 0], "vae": ["39", 0], "audio_encoder_output": ["56", 0], "ref_image": ["52", 0]}, "class_type": "WanSoundImageToVideo"},
    "94":  {"inputs": {"dim": "t", "index": 0, "amount": 1, "samples": ["85:182", 0]}, "class_type": "LatentCut"},
    "95":  {"inputs": {"dim": "t", "samples1": ["94", 0], "samples2": ["85:182", 0]}, "class_type": "LatentConcat"},
    "96":  {"inputs": {"batch_index": ["100", 0], "length": 4096, "image": ["80", 0]}, "class_type": "ImageFromBatch"},
    "100": {"inputs": {"value": 3}, "class_type": "PrimitiveInt"},
    "103": {"inputs": {"value": 4}, "class_type": "PrimitiveInt"},
    "104": {"inputs": {"value": 43}, "class_type": "PrimitiveInt"},
    "105": {"inputs": {"value": 1}, "class_type": "PrimitiveFloat"},
    "107": {"inputs": {"lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", "strength_model": 1, "model": ["37", 0]}, "class_type": "LoraLoaderModelOnly"},
    "113": {"inputs": {"filename_prefix": "video/ComfyUI", "format": "auto", "codec": "auto", "video-preview": "", "video": ["82", 0]}, "class_type": "SaveVideo"},
    "79:76": {"inputs": {"length": ["104", 0], "positive": ["6", 0], "negative": ["7", 0], "vae": ["39", 0], "video_latent": ["3", 0], "audio_encoder_output": ["56", 0], "ref_image": ["52", 0]}, "class_type": "WanSoundImageToVideoExtend"},
    "79:77": {"inputs": {"seed": 1, "steps": ["103", 0], "cfg": ["105", 0], "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1, "model": ["54", 0], "positive": ["79:76", 0], "negative": ["79:76", 1], "latent_image": ["79:76", 2]}, "class_type": "KSampler"},
    "79:78": {"inputs": {"dim": "t", "samples1": ["3", 0], "samples2": ["79:77", 0]}, "class_type": "LatentConcat"},
    "85:182": {"inputs": {"dim": "t", "samples1": ["79:78", 0], "samples2": ["85:183", 0]}, "class_type": "LatentConcat"},
    "85:183": {"inputs": {"seed": 250, "steps": ["103", 0], "cfg": ["105", 0], "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1, "model": ["54", 0], "positive": ["85:184", 0], "negative": ["85:184", 1], "latent_image": ["85:184", 2]}, "class_type": "KSampler"},
    "85:184": {"inputs": {"length": ["104", 0], "positive": ["6", 0], "negative": ["7", 0], "vae": ["39", 0], "video_latent": ["79:78", 0], "audio_encoder_output": ["56", 0], "ref_image": ["52", 0]}, "class_type": "WanSoundImageToVideoExtend"},
}

# ── Upload helpers ────────────────────────────────────────────────────────────

def upload_image(image_bytes: bytes, filename: str) -> str:
    resp = requests.post(
        f"{COMFY_URL}/upload/image",
        data={"overwrite": "true", "type": "input"},
        files={"image": (filename, image_bytes)},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["name"]


def upload_audio(audio_bytes: bytes, filename: str) -> str:
    # ComfyUI has no /upload/audio endpoint; /upload/image accepts any file type
    resp = requests.post(
        f"{COMFY_URL}/upload/image",
        data={"overwrite": "true", "type": "input"},
        files={"image": (filename, audio_bytes)},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["name"]

# ── Workflow builders ─────────────────────────────────────────────────────────

def build_workflow(
    prompt: str,
    width: int = 848,
    height: int = 480,
    length: int = 81,
    seed: int | None = None,
    fps: int = 16,
) -> dict:
    wf = copy.deepcopy(BASE_WORKFLOW)
    wf["89"]["inputs"]["text"] = prompt
    wf["74"]["inputs"]["width"] = width
    wf["74"]["inputs"]["height"] = height
    wf["74"]["inputs"]["length"] = length
    wf["81"]["inputs"]["noise_seed"] = seed if seed is not None else random.randint(0, 2**32 - 1)
    wf["88"]["inputs"]["fps"] = fps
    return wf


def build_i2v_workflow(
    image_filename: str,
    prompt: str,
    width: int = 512,
    height: int = 512,
    length: int = 81,
    seed: int | None = None,
    fps: int = 16,
) -> dict:
    wf = copy.deepcopy(I2V_BASE_WORKFLOW)
    wf["97"]["inputs"]["image"] = image_filename
    wf["116:93"]["inputs"]["text"] = prompt
    wf["116:98"]["inputs"]["width"] = width
    wf["116:98"]["inputs"]["height"] = height
    wf["116:98"]["inputs"]["length"] = length
    wf["116:86"]["inputs"]["noise_seed"] = seed if seed is not None else random.randint(0, 2**32 - 1)
    wf["116:94"]["inputs"]["fps"] = fps
    return wf


def build_s2v_workflow(
    image_filename: str,
    audio_filename: str,
    prompt: str = "",
    seed: int | None = None,
    fps: int = 16,
    chunk_length: int = 43,
) -> dict:
    wf = copy.deepcopy(S2V_BASE_WORKFLOW)
    wf["52"]["inputs"]["image"] = image_filename
    wf["58"]["inputs"]["audio"] = audio_filename
    wf["6"]["inputs"]["text"] = prompt
    wf["3"]["inputs"]["seed"] = seed if seed is not None else random.randint(0, 2**32 - 1)
    wf["104"]["inputs"]["value"] = chunk_length
    wf["82"]["inputs"]["fps"] = fps
    return wf

# ── Core API ──────────────────────────────────────────────────────────────────

def submit(workflow: dict) -> str:
    resp = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow}, timeout=30)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def poll(
    prompt_id: str,
    interval: float = 5.0,
    timeout: float = 600.0,
    on_progress=None,
) -> dict:
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(f"Generation timed out after {timeout}s")
        if on_progress:
            on_progress(elapsed)
        time.sleep(interval)
        try:
            resp = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
            resp.raise_for_status()
            history = resp.json()
        except requests.RequestException:
            continue
        if prompt_id in history:
            data = history[prompt_id]
            status = data.get("status", {}).get("status_str", "")
            if status == "error":
                raise RuntimeError(f"ComfyUI generation error: {data['status']}")
            return data


def download_video(output_info: dict) -> bytes:
    params = {
        "filename": output_info["filename"],
        "subfolder": output_info["subfolder"],
        "type": output_info["type"],
    }
    resp = requests.get(f"{COMFY_URL}/view", params=params, timeout=120)
    resp.raise_for_status()
    return resp.content

# ── High-level generate functions ─────────────────────────────────────────────

def generate(
    prompt: str,
    width: int = 848,
    height: int = 480,
    length: int = 81,
    seed: int | None = None,
    fps: int = 16,
    on_progress=None,
) -> tuple[bytes, str]:
    wf = build_workflow(prompt, width, height, length, seed, fps)
    prompt_id = submit(wf)
    data = poll(prompt_id, on_progress=on_progress)
    video_info = data["outputs"]["80"]["images"][0]
    return download_video(video_info), video_info["filename"]


def generate_i2v(
    image_bytes: bytes,
    image_filename: str,
    prompt: str,
    width: int = 512,
    height: int = 512,
    length: int = 81,
    seed: int | None = None,
    fps: int = 16,
    on_progress=None,
) -> tuple[bytes, str]:
    stored_name = upload_image(image_bytes, image_filename)
    wf = build_i2v_workflow(stored_name, prompt, width, height, length, seed, fps)
    prompt_id = submit(wf)
    data = poll(prompt_id, on_progress=on_progress)
    video_info = data["outputs"]["108"]["images"][0]
    return download_video(video_info), video_info["filename"]


def generate_s2v(
    image_bytes: bytes,
    image_filename: str,
    audio_bytes: bytes,
    audio_filename: str,
    prompt: str = "",
    seed: int | None = None,
    fps: int = 16,
    chunk_length: int = 43,
    on_progress=None,
) -> tuple[bytes, str]:
    stored_image = upload_image(image_bytes, image_filename)
    stored_audio = upload_audio(audio_bytes, audio_filename)
    wf = build_s2v_workflow(stored_image, stored_audio, prompt, seed, fps, chunk_length)
    prompt_id = submit(wf)
    data = poll(prompt_id, on_progress=on_progress)
    video_info = data["outputs"]["113"]["images"][0]
    return download_video(video_info), video_info["filename"]
