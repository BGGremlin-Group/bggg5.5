#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
####################################
# BGGG (Background Gremlin Group): Batch Video Upscaler v5.5 for ComfyUI
####################################
# End-to-end pipeline:
#  1) Extract frames with ffmpeg              -> frames/frame_%06d.png
#  2) Upscale each frame through ComfyUI      -> download outputs -> frames_upscaled/frame_%06d.png
#  3) Recombine frames (and audio if present) -> output_upscaled.mp4
####################################
# UIs:
#  • TUI (retro number-menu)      • Web UI (http://127.0.0.1:7860)      • Retro CRT GUI (pygame: scanlines, logo, amber/cyan/magenta)
####################################
# Pipelines:
#  • ESRGAN (vanilla ImageUpscaleWithModel). If extra_scale != 1.0, adds ImageScale (bicubic) after SR.
#  • SDXL + (Tiled) VAE + LatentUpscaleBy + KSampler (VAE correctly wired to Encode/Decode). Tiled nodes optional.
####################################
# Requirements:
#  Python 3.9+ • ffmpeg in PATH • ComfyUI running (default http://127.0.0.1:8188)
#  pip install requests flask colorama pygame  watchdog
####################################
# Assets:
#  assets/logo_bggg.png   (logo, drawn top-left)
#  assets/bg_crt.png      (amber tech grid background)
#
# Developed by the BG Gremlin Group 2025
####################################
"""

import os, sys, json, time, shlex, random, threading, traceback, platform, subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import quote

# ---- Lazy imports (optional deps don’t block non-GUI modes) ----
_MISSING: List[str] = []
try:
    import requests
except Exception:
    _MISSING.append("requests")
try:
    from colorama import init as colorama_init
except Exception:
    _MISSING.append("colorama")
try:
    from flask import Flask, request as flask_request, render_template_string, redirect, url_for
except Exception:
    _MISSING.append("flask")
try:
    import pygame
except Exception:
    pygame = None
    _MISSING.append("pygame")
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:
    Observer = None
    class FileSystemEventHandler: ...  # typing shim
    _MISSING.append("watchdog")

# ---- App constants & CRT style ----
APP_NAME = "BGGG Video Upscaler"
VERSION  = "2.2.0"
DEFAULT_COMFY_HOST, DEFAULT_COMFY_PORT = "127.0.0.1", 8188
DEFAULT_WEB_PORT = 7860

ASSETS_DIR = Path("assets")
ASSET_LOGO = ASSETS_DIR / "logo_bggg.png"
ASSET_BG   = ASSETS_DIR / "bg_crt.png"

CRT_AMBER = "\033[38;2;255;191;0m"; CRT_CYAN = "\033[96m"; CRT_MAG = "\033[95m"
CRT_DIM = "\033[2m"; CRT_BOLD = "\033[1m"; CRT_RESET = "\033[0m"

ASCII_BANNER = r"""
██████╗  ██████╗  ██████╗  ██████╗ 
██╔══██╗██╔════╝ ██╔════╝ ██╔════╝ 
██████╔╝██║  ███╗██║  ███╗██║  ███╗
██╔══██╗██║   ██║██║   ██║██║   ██║
██║  ██║╚██████╔╝╚██████╔╝╚██████╔╝
╚═╝  ╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ 
   B G G G   •  Batch Gremlin Group
""".strip("\n")

if "colorama" not in _MISSING:
    colorama_init()

def is_windows() -> bool: return platform.system().lower().startswith("win")
def _ts() -> str: return time.strftime("%Y-%m-%d %H:%M:%S")

# ---- Logger ----
class Log:
    def __init__(self, path: Path, debug_console: bool = False):
        self.path = path; self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock(); self.debug_console = debug_console
    def _out(self, lvl: str, msg: str):
        line = f"[{_ts()}] [{lvl.upper()}] {msg}\n"
        with self._lock: self.path.open("a", encoding="utf-8").write(line)
        color = CRT_AMBER if lvl!="error" else CRT_MAG
        if lvl=="debug": color = CRT_CYAN
        if lvl!="debug" or self.debug_console:
            print(color + line.rstrip() + CRT_RESET)
    def info(self, m): self._out("info", m)
    def warn(self, m): self._out("warn", m)
    def error(self, m): self._out("error", m)
    def debug(self, m): self._out("debug", m)

# ---- Config ----
@dataclass
class AppConfig:
    comfy_host: str = DEFAULT_COMFY_HOST
    comfy_port: int = DEFAULT_COMFY_PORT
    web_port: int = DEFAULT_WEB_PORT

    # Paths
    comfy_root: str = ""     # root of ComfyUI for autodiscovery/output direct-copy
    input_video: str = ""
    frames_dir: str = "frames"
    frames_glob: str = "frame_%06d.png"
    upscale_out_dir: str = "frames_upscaled"
    output_video: str = "output_upscaled.mp4"
    watch_folder: str = ""

    # Video
    fps_override: float = 0.0
    audio_copy: bool = True

    # ESRGAN
    esrgan_model_name: str = "4x-UltraSharp.pth"
    esrgan_extra_scale: float = 1.0
    esrgan_batch_pause: float = 0.0

    # SDXL
    sdxl_ckpt_name: str = "sd_xl_base_1.0.safetensors"
    sdxl_positive: str = "highly detailed, sharp, natural texture, faithful upscale"
    sdxl_negative: str = "blur, noise, artifacts, distortion, oversharpen"
    sdxl_steps: int = 18
    sdxl_cfg: float = 4.5
    sdxl_sampler_name: str = "dpmpp_2m"
    sdxl_scheduler: str = "karras"
    sdxl_denoise: float = 0.4
    sdxl_latent_upscale_by: float = 2.0
    sdxl_seed: int = 0
    sdxl_use_tiled_vae: bool = True
    sdxl_tile_size: int = 512

    # Misc
    debug: bool = False
    gui_scale: float = 1.0
    config_path: str = "bggg_config.json"
    web_running: bool = False

    def save(self): Path(self.config_path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
    @staticmethod
    def load(path: str = "bggg_config.json") -> "AppConfig":
        p = Path(path)
        return AppConfig(**json.loads(p.read_text(encoding="utf-8"))) if p.exists() else AppConfig()

# ---- FS & process helpers ----
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def find_ffmpeg() -> Optional[str]:
    exe = "ffmpeg.exe" if is_windows() else "ffmpeg"
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(d, exe)
        if cand.is_file(): return str(cand)
    # common guesses
    guesses = [r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"] if is_windows() \
              else ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
    for g in guesses:
        if Path(g).is_file(): return g
    return None

def run_cmd(cmd: List[str], logger: Log, cwd: Optional[Path] = None, log_on_err: bool = True) -> Tuple[int,str,str]:
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, text=True)
        out, err = p.communicate()
        if p.returncode != 0 and log_on_err:
            logger.error(f"Command failed (rc={p.returncode}): {' '.join(shlex.quote(x) for x in cmd)}")
            if err.strip(): logger.error(err.strip())
        else:
            logger.debug(f"Command ok: {' '.join(shlex.quote(x) for x in cmd)}")
        return p.returncode, out, err
    except FileNotFoundError:
        if log_on_err: logger.error(f"Executable not found: {cmd[0]}")
        return 127, "", "not found"
    except Exception as e:
        if log_on_err: logger.error(f"run_cmd exception: {e}")
        return 1, "", str(e)

def parse_fps(text: str) -> Optional[float]:
    for line in text.splitlines():
        if " fps," in line:
            try: return float(line.split(" fps,")[0].split()[-1])
            except: pass
        if " tbr," in line:
            try: return float(line.split(" tbr,")[0].split()[-1])
            except: pass
    return None

def get_video_fps(ffmpeg: str, vid: Path, logger: Log) -> Optional[float]:
    rc, out, err = run_cmd([ffmpeg, "-hide_banner", "-i", str(vid)], logger, log_on_err=False)
    fps = parse_fps(out + err)
    if not fps: logger.warn("FPS autodetect failed; will fall back.")
    return fps

def has_audio(ffmpeg: str, vid: Path, logger: Log) -> bool:
    rc, out, err = run_cmd([ffmpeg, "-hide_banner", "-i", str(vid)], logger, log_on_err=False)
    return "Audio:" in (out+err) or "audio:" in (out+err)

# ---- ComfyUI HTTP client ----
class ComfyClient:
    def __init__(self, host: str, port: int, logger: Log, timeout: float = 10.0):
        self.base = f"http://{host}:{port}"; self.logger = logger; self.timeout = timeout
    def ping(self) -> bool:
        try:
            r = requests.get(self.base, timeout=self.timeout); return r.status_code < 500
        except Exception as e: self.logger.error(f"ComfyUI ping failed: {e}"); return False
    def upload_image(self, path: Path) -> Optional[str]:
        try:
            with path.open("rb") as f:
                r = requests.post(f"{self.base}/upload/image", files={"image": (path.name, f, "image/png")}, timeout=120)
            if r.status_code == 200:
                data = r.json(); name = data.get("name") or data.get("filename") or data.get("path")
                if not name: raise RuntimeError(str(data))
                return name
            self.logger.error(f"Upload failed {r.status_code}: {r.text[:300]}"); return None
        except Exception as e: self.logger.error(f"Upload exception: {e}"); return None
    def queue_prompt(self, graph: Dict[str,Any]) -> Optional[str]:
        try:
            r = requests.post(f"{self.base}/prompt", json={"prompt": graph}, timeout=180)
            if r.status_code == 200: return r.json().get("prompt_id")
            self.logger.error(f"/prompt 400: {r.text[:300]}"); return None
        except Exception as e: self.logger.error(f"Queue exception: {e}"); return None
    def wait_history(self, pid: str, timeout: float = 3600.0, poll: float = 1.0) -> Optional[Dict[str,Any]]:
        start = time.time()
        url = f"{self.base}/history/{pid}"
        while True:
            try:
                r = requests.get(url, timeout=self.timeout)
                if r.status_code == 200:
                    j = r.json()
                    if j and pid in j: return j[pid]
            except Exception as e:
                self.logger.warn(f"history poll error: {e}")
            if time.time() - start > timeout:
                self.logger.error(f"Timeout waiting for prompt {pid}")
                return None
            time.sleep(poll)
    def view_image(self, filename: str, subfolder: str = "", typ: str = "output") -> Optional[bytes]:
        try:
            params = {"filename": filename, "type": typ}
            if subfolder: params["subfolder"] = subfolder
            r = requests.get(f"{self.base}/view", params=params, timeout=180)
            return r.content if r.status_code == 200 else None
        except Exception as e: self.logger.warn(f"/view error: {e}"); return None

# ---- Graphs (vanilla-safe) ----
def graph_esrgan(upload_name: str, model_name: str, extra_scale: float, out_prefix: str) -> Dict[str,Any]:
    g = {
        "1": {"class_type": "LoadImage", "inputs": {"image": upload_name}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"image": ["1",0], "upscale_model": ["2",0]}},
    }
    last = "3"
    if extra_scale != 1.0:
        g["10"] = {"class_type": "ImageScale", "inputs": {
            "image": [last, 0], "upscale_method": "bicubic", "scale_by": extra_scale,
            "width": 0, "height": 0, "crop": "disabled"
        }}
        last = "10"
    g["4"] = {"class_type": "SaveImage", "inputs": {"images": [last,0], "filename_prefix": out_prefix}}
    return g

def graph_sdxl(upload_name: str, ckpt_name: str, pos: str, neg: str, steps: int, cfg: float,
               sampler: str, scheduler: str, denoise: float, scale_by: float, seed: int,
               use_tiled_vae: bool, tile_size: int, out_prefix: str) -> Dict[str,Any]:
    vae_encode = "TiledVAEEncode" if use_tiled_vae else "VAEEncode"
    vae_decode = "TiledVAEDecode" if use_tiled_vae else "VAEDecode"
    vae_args   = ({"tile_size": tile_size} if use_tiled_vae else {})
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": upload_name}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["2",1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["2",1]}},
        "5": {"class_type": vae_encode, "inputs": {"pixels": ["1",0], "vae": ["2",2], **vae_args}},
        "6": {"class_type": "LatentUpscaleBy", "inputs": {"samples": ["5",0], "scale_by": scale_by}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
            "denoise": denoise, "model": ["2",0], "positive": ["3",0], "negative": ["4",0], "latent_image": ["6",0]
        }},
        "8": {"class_type": vae_decode, "inputs": {"samples": ["7",0], "vae": ["2",2], **vae_args}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8",0], "filename_prefix": out_prefix}}
    }

# ---- Core app ----
class BGGGApp:
    VIDEO_EXTS = {".mp4",".mov",".mkv",".avi",".webm",".mpg",".mpeg",".m4v"}

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        ensure_dir(Path("logs"))
        self.log = Log(Path("logs")/"bggg.log", debug_console=cfg.debug)
        self.comfy = ComfyClient(cfg.comfy_host, cfg.comfy_port, self.log)
        self.stop_event = threading.Event()
        self._watcher = None; self._poll_thread = None

    # ---- deps & comfy ----
    def deps_ok(self, require_gui=False) -> bool:
        missing = set(_MISSING); fatal=[]
        for pkg in ("requests","flask"):
            if pkg in missing: fatal.append(pkg)
        if require_gui and ("pygame" in missing): fatal.append("pygame")
        if not find_ffmpeg(): fatal.append("ffmpeg (binary)")
        if fatal:
            print(CRT_MAG+"[!] Missing: "+", ".join(fatal)+CRT_RESET); return False
        return True
    def comfy_ok(self) -> bool:
        ok = self.comfy.ping()
        if not ok: self.log.error("ComfyUI not reachable. Please start ComfyUI.")
        return ok

    # ---- discovery ----
    def comfy_paths(self) -> Dict[str,Path]:
        base = Path(self.cfg.comfy_root) if self.cfg.comfy_root else None
        if not base or not base.exists(): return {}
        return {"upscale_models": base/"models"/"upscale_models",
                "checkpoints": base/"models"/"checkpoints",
                "output": base/"output"}
    def discover_models(self) -> Tuple[List[str],List[str]]:
        ups, ck = [], []
        p = self.comfy_paths()
        if p.get("upscale_models", None) and p["upscale_models"].is_dir():
            ups = sorted([f.name for f in p["upscale_models"].iterdir() if f.is_file()])
        if p.get("checkpoints", None) and p["checkpoints"].is_dir():
            ck = sorted([f.name for f in p["checkpoints"].iterdir() if f.is_file()])
        return ups, ck
    def comfy_output_dir(self) -> Optional[Path]:
        p = self.comfy_paths().get("output")
        if p and p.is_dir(): return p
        # permissive guesses
        for guess in [Path("ComfyUI")/"output", Path.cwd()/"output"]:
            if guess.is_dir(): return guess
        return None

    # ---- pipeline steps ----
    def extract_frames(self, video: Path, fps: Optional[float]) -> bool:
        ff = find_ffmpeg()
        if not ff: self.log.error("ffmpeg not found."); return False
        ensure_dir(Path(self.cfg.frames_dir))
        pattern = Path(self.cfg.frames_dir)/self.cfg.frames_glob
        cmd=[ff,"-y","-i",str(video)]
        if fps and fps>0: cmd += ["-r", str(fps)]
        cmd += [str(pattern)]
        self.log.info(f"Extracting frames from {video.name}")
        rc,_,_=run_cmd(cmd, self.log)
        return rc==0

    def recombine_frames(self, src_video: Path, fps: Optional[float]) -> bool:
        ff = find_ffmpeg()
        if not ff: self.log.error("ffmpeg not found."); return False
        pattern = Path(self.cfg.upscale_out_dir)/self.cfg.frames_glob
        outv = Path(self.cfg.output_video)
        cmd=[ff,"-y"]
        if fps and fps>0: cmd += ["-framerate", str(fps)]
        cmd += ["-i", str(pattern)]
        if self.cfg.audio_copy and src_video.exists() and has_audio(ff, src_video, self.log):
            cmd += ["-i", str(src_video), "-map","0:v:0","-map","1:a:0","-c:a","copy"]
        cmd += ["-c:v","libx264","-pix_fmt","yuv420p",str(outv)]
        self.log.info(f"Recombining → {outv}")
        rc,_,_=run_cmd(cmd, self.log)
        return rc==0

    def _copy_history_output(self, history: Dict[str,Any], save_node_ids: List[str], index: int) -> bool:
        # try to locate the newest image from our SaveImage node(s)
        img=None; sub=""
        outs = history.get("outputs") or {}
        for nid in save_node_ids:
            if nid in outs:
                imgs = outs[nid].get("images") or []
                if imgs:
                    e = imgs[-1]; img=e.get("filename") or e.get("name"); sub=e.get("subfolder") or ""
                    if img: break
        if not img:
            self.log.error("No image filename found in history."); return False
        target = Path(self.cfg.upscale_out_dir)/f"frame_{index:06d}.png"
        # try direct file copy from ComfyUI/output
        outdir = self.comfy_output_dir()
        if outdir:
            src = (outdir/sub/img) if sub else (outdir/img)
            if src.exists():
                try:
                    ensure_dir(target.parent); target.write_bytes(src.read_bytes())
                    self.log.debug(f"Copied {src} -> {target}"); return True
                except Exception as e:
                    self.log.warn(f"Direct copy failed ({e}); trying /view.")
        # fallback: /view download
        data = self.comfy.view_image(img, subfolder=sub, typ="output")
        if not data: self.log.error("Failed to fetch image via /view."); return False
        try:
            ensure_dir(target.parent); target.write_bytes(data); return True
        except Exception as e: self.log.error(f"Write failed for {target}: {e}"); return False

    def _batch(self, frames_dir: Path, mode: str) -> bool:
        if not self.comfy_ok(): return False
        files = sorted([*frames_dir.glob("*.png"), *frames_dir.glob("*.jpg"), *frames_dir.glob("*.jpeg")])
        if not files: self.log.error(f"No frames in {frames_dir}"); return False
        ensure_dir(Path(self.cfg.upscale_out_dir))
        total=len(files); self.log.info(f"Batch start ({mode}) — {total} frames")
        save_prefix = "BGGG_"
        for idx, fp in enumerate(files, 1):
            if self.stop_event.is_set(): self.log.warn("Stop requested."); return False
            up = self.comfy.upload_image(fp)
            if not up: self.log.error(f"Upload failed: {fp.name}"); continue
            if mode=="esrgan":
                graph = graph_esrgan(up, self.cfg.esrgan_model_name, self.cfg.esrgan_extra_scale, save_prefix)
                save_ids=["4"]
            else:
                seed = (int(time.time()*1000)&0xffffffff) if self.cfg.sdxl_seed==0 else self.cfg.sdxl_seed
                graph = graph_sdxl(up, self.cfg.sdxl_ckpt_name, self.cfg.sdxl_positive, self.cfg.sdxl_negative,
                                   self.cfg.sdxl_steps, self.cfg.sdxl_cfg, self.cfg.sdxl_sampler_name,
                                   self.cfg.sdxl_scheduler, self.cfg.sdxl_denoise, self.cfg.sdxl_latent_upscale_by,
                                   seed, self.cfg.sdxl_use_tiled_vae, self.cfg.sdxl_tile_size, save_prefix)
                save_ids=["9"]
            pid = self.comfy.queue_prompt(graph)
            if not pid: continue
            hist = self.comfy.wait_history(pid, timeout=3600, poll=1.0)
            if not hist: self.log.error(f"Timeout/fetch failed for {fp.name}"); continue
            if not self._copy_history_output(hist, save_ids, idx):
                self.log.error(f"Failed to store upscaled: {fp.name}")
                continue
            self.log.info(f"[{idx}/{total}] {fp.name}")
            if self.cfg.esrgan_batch_pause>0: time.sleep(self.cfg.esrgan_batch_pause)
        self.log.info("Batch complete."); return True

    # ---- public actions ----
    def action_extract(self) -> bool:
        p = Path(self.cfg.input_video)
        if p.is_dir():
            vids = sorted([x for x in p.iterdir() if x.suffix.lower() in self.VIDEO_EXTS])
            if not vids: self.log.error("No videos in folder."); return False
            p = vids[0]; self.log.warn(f"Folder provided; using first: {p.name}")
        if not p.exists(): self.log.error(f"Input not found: {p}"); return False
        ff=find_ffmpeg(); fps = self.cfg.fps_override if self.cfg.fps_override>0 else (get_video_fps(ff,p,self.log) or 30.0)
        return self.extract_frames(p,fps)
    def action_esrgan(self) -> bool: return self._batch(Path(self.cfg.frames_dir), "esrgan")
    def action_sdxl(self) -> bool:   return self._batch(Path(self.cfg.frames_dir), "sdxl")
    def action_recombine(self) -> bool:
        p=Path(self.cfg.input_video); 
        if p.is_dir():
            vids = sorted([x for x in p.iterdir() if x.suffix.lower() in self.VIDEO_EXTS]); p = vids[0] if vids else p
        ff=find_ffmpeg(); fps=self.cfg.fps_override if self.cfg.fps_override>0 else (get_video_fps(ff,p,self.log) or 30.0)
        return self.recombine_frames(p,fps)
    def action_test(self) -> bool:
        ok=self.comfy_ok(); 
        if ok: self.log.info("ComfyUI reachable.")
        return ok

    # ---- watcher ----
    class _WatchHandler(FileSystemEventHandler):
        def __init__(self, app:"BGGGApp", mode_getter):
            super().__init__(); self.app=app; self.mode_getter=mode_getter
        def on_created(self, event):
            if getattr(event,"is_directory",False): return
            path=Path(event.src_path)
            if path.suffix.lower() in self.app.VIDEO_EXTS:
                self.app.log.info(f"New video: {path.name} — pipeline starting")
                self.app.cfg.input_video=str(path); self.app.cfg.save()
                try: self.app.pipeline(path, self.mode_getter())
                except Exception as e: self.app.log.error(f"Auto pipeline failed: {e}")

    def start_watcher(self, mode_getter, poll_seconds=5):
        folder=Path(self.cfg.watch_folder)
        if not folder.exists(): self.log.error("Watch folder missing."); return
        if Observer:
            handler=BGGGApp._WatchHandler(self, mode_getter)
            obs=Observer(); obs.schedule(handler, str(folder), recursive=False); obs.start()
            self._watcher=obs; self.log.info(f"Watcher started: {folder}")
        else:
            self.log.warn("watchdog not installed — polling fallback.")
            seen={f.name for f in folder.iterdir() if f.is_file()}
            stop=self.stop_event
            def loop():
                nonlocal seen
                while not stop.is_set():
                    now={f.name for f in folder.iterdir() if f.is_file()}
                    new=[n for n in now-seen if Path(folder,n).suffix.lower() in self.VIDEO_EXTS]
                    for n in new:
                        path=Path(folder,n); self.log.info(f"New video: {path.name} — pipeline starting")
                        self.cfg.input_video=str(path); self.cfg.save()
                        try: self.pipeline(path, mode_getter())
                        except Exception as e: self.log.error(f"Auto pipeline failed: {e}")
                    seen=now; time.sleep(poll_seconds)
            t=threading.Thread(target=loop, daemon=True); t.start(); self._poll_thread=t

    def stop_watcher(self):
        if self._watcher:
            self._watcher.stop(); self._watcher.join(); self._watcher=None; self.log.info("Watcher stopped.")
        if self._poll_thread:
            self.stop_event.set(); self._poll_thread.join(timeout=1); self._poll_thread=None; self.stop_event.clear()
            self.log.info("Polling watcher stopped.")

    # ---- convenience full pipeline ----
    def pipeline(self, video_path: Path, mode: str) -> bool:
        ff=find_ffmpeg()
        fps=self.cfg.fps_override if self.cfg.fps_override>0 else (get_video_fps(ff,video_path,self.log) or 30.0)
        return self.extract_frames(video_path,fps) and self._batch(Path(self.cfg.frames_dir),mode) and self.recombine_frames(video_path,fps)

# ---- Help text ----
HELP_TEXT = f"""
{CRT_BOLD}BGGG Help / Walkthrough{CRT_RESET}

{CRT_CYAN}Pipeline{CRT_RESET}
1) Extract frames from your input video (ffmpeg).
2) Send each frame to ComfyUI:
   - ESRGAN: UpscaleModelLoader → ImageUpscaleWithModel → [optional ImageScale] → SaveImage
   - SDXL:   Checkpoint → (Tiled)VAEEncode → LatentUpscaleBy → KSampler → (Tiled)VAEDecode → SaveImage
   VAE is wired from the checkpoint to both encode & decode.
3) Download the result from ComfyUI history (or copy from ComfyUI/output) and save as:
   {CRT_BOLD}frames_upscaled/frame_%06d.png{CRT_RESET}
4) Recombine back to video. If the source has audio and Audio Copy is enabled, it is preserved.

{CRT_CYAN}Notes{CRT_RESET}
• ESRGAN tiling inputs aren’t supported by vanilla ComfyUI; this build avoids them.
• If you need extra scaling (e.g., 1.5x after 4x SR), we insert an {CRT_BOLD}ImageScale(bicubic){CRT_RESET}.
• SDXL tiled VAE requires a tiled-vae extension; otherwise we fall back automatically.

{CRT_CYAN}Tips{CRT_RESET}
• SDXL: Steps ~18–30, CFG 4–6, Denoise 0.3–0.5, Latent scale 2.0. Tiled VAE 512–768 if VRAM is tight.
• ESRGAN: 4x-UltraSharp.pth is a good general model; keep extra_scale 1.0 to use native SR scale.
• Use SSD/NVMe for frames dirs. Add batch pause if the GPU/CPU is saturated.
"""

# ---- TUI ----
def tui_banner():
    os.system("cls" if is_windows() else "clear")
    print(CRT_AMBER + ASCII_BANNER + CRT_RESET)
    print(CRT_DIM + f"{APP_NAME} v{VERSION}\n" + CRT_RESET)

def tui_menu(cfg: AppConfig):
    print(CRT_AMBER+"Main Menu"+CRT_RESET)
    print(f"{CRT_CYAN}1{CRT_RESET}. Configure")
    print(f"{CRT_CYAN}2{CRT_RESET}. Extract frames")
    print(f"{CRT_CYAN}3{CRT_RESET}. Upscale (ESRGAN)")
    print(f"{CRT_CYAN}4{CRT_RESET}. Upscale (SDXL + Tiled)")
    print(f"{CRT_CYAN}5{CRT_RESET}. Recombine frames → video")
    print(f"{CRT_CYAN}6{CRT_RESET}. Start/Stop Web UI (http://127.0.0.1:{cfg.web_port})")
    print(f"{CRT_CYAN}7{CRT_RESET}. Help manual")
    print(f"{CRT_CYAN}8{CRT_RESET}. Test ComfyUI")
    print(f"{CRT_CYAN}9{CRT_RESET}. Launch CRT GUI")
    print(f"{CRT_MAG}0{CRT_RESET}. Exit\n")

def ask(prompt: str) -> str:
    print(CRT_AMBER+prompt+CRT_RESET, end=" "); return input().strip()

def configure_interactive(app: BGGGApp):
    c=app.cfg
    v=ask(f"ComfyUI root (autodiscovery) [{c.comfy_root or 'unset'}]:");           c.comfy_root = v or c.comfy_root
    v=ask(f"Comfy host [{c.comfy_host}]:");                                         c.comfy_host = v or c.comfy_host
    v=ask(f"Comfy port [{c.comfy_port}]:");                                         c.comfy_port = int(v) if v.isdigit() else c.comfy_port
    v=ask(f"Input video (file or folder) [{c.input_video}]:");                       c.input_video = v or c.input_video
    v=ask(f"Frames dir [{c.frames_dir}]:");                                         c.frames_dir = v or c.frames_dir
    v=ask(f"Frames pattern [{c.frames_glob}]:");                                     c.frames_glob = v or c.frames_glob
    v=ask(f"Upscaled frames dir [{c.upscale_out_dir}]:");                            c.upscale_out_dir = v or c.upscale_out_dir
    v=ask(f"Output video path [{c.output_video}]:");                                 c.output_video = v or c.output_video
    v=ask(f"Watch folder (blank to disable) [{c.watch_folder or 'disabled'}]:");     c.watch_folder = v or c.watch_folder
    v=ask(f"FPS override (0=auto) [{c.fps_override}]:");                             
    if v:
        try: c.fps_override = float(v)
        except: pass
    v=ask(f"Copy audio if present? (y/n) [{'y' if c.audio_copy else 'n'}]:");        c.audio_copy = v.lower().startswith("y") if v else c.audio_copy

    ups, ck = app.discover_models()
    if ups:
        print(CRT_CYAN+f"Found {len(ups)} upscaler models:"+CRT_RESET); 
        for i,n in enumerate(ups,1): print(f"  {i}. {n}")
        v=ask(f"Pick upscaler [enter keeps {c.esrgan_model_name}]:"); 
        if v.isdigit() and 1<=int(v)<=len(ups): c.esrgan_model_name = ups[int(v)-1]
    else:
        v=ask(f"Upscale model name [{c.esrgan_model_name}]:");                        c.esrgan_model_name = v or c.esrgan_model_name
    if ck:
        print(CRT_CYAN+f"Found {len(ck)} SDXL checkpoints:"+CRT_RESET)
        for i,n in enumerate(ck,1): print(f"  {i}. {n}")
        v=ask(f"Pick checkpoint [enter keeps {c.sdxl_ckpt_name}]:"); 
        if v.isdigit() and 1<=int(v)<=len(ck): c.sdxl_ckpt_name = ck[int(v)-1]
    else:
        v=ask(f"SDXL checkpoint filename [{c.sdxl_ckpt_name}]:");                     c.sdxl_ckpt_name = v or c.sdxl_ckpt_name

    v=ask(f"[ESRGAN] Extra scale (1.0 none) [{c.esrgan_extra_scale}]:");             
    if v:
        try: c.esrgan_extra_scale = float(v)
        except: pass
    v=ask(f"[ESRGAN] Batch pause seconds [{c.esrgan_batch_pause}]:");
    if v:
        try: c.esrgan_batch_pause = float(v)
        except: pass

    v=ask(f"[SDXL] Steps [{c.sdxl_steps}]:");                                         c.sdxl_steps = int(v) if v.isdigit() else c.sdxl_steps
    v=ask(f"[SDXL] CFG [{c.sdxl_cfg}]:");                                            
    if v:
        try: c.sdxl_cfg = float(v)
        except: pass
    v=ask(f"[SDXL] Denoise 0..1 [{c.sdxl_denoise}]:");                                
    if v:
        try: c.sdxl_denoise = float(v)
        except: pass
    v=ask(f"[SDXL] Latent upscale by [{c.sdxl_latent_upscale_by}]:");
    if v:
        try: c.sdxl_latent_upscale_by = float(v)
        except: pass
    v=ask(f"[SDXL] Seed (0=random) [{c.sdxl_seed}]:");                                
    if v:
        try: c.sdxl_seed = int(v)
        except: pass
    v=ask(f"[SDXL] Use Tiled VAE? (y/n) [{'y' if c.sdxl_use_tiled_vae else 'n'}]:");  c.sdxl_use_tiled_vae = v.lower().startswith("y") if v else c.sdxl_use_tiled_vae
    v=ask(f"[SDXL] Tile size [{c.sdxl_tile_size}]:");                                 c.sdxl_tile_size = int(v) if v.isdigit() else c.sdxl_tile_size

    v=ask(f"Enable DEBUG console? (y/n) [{'y' if c.debug else 'n'}]:");               c.debug = v.lower().startswith("y") if v else c.debug
    c.save(); print(CRT_AMBER+"Configuration saved.\n"+CRT_RESET)

# ---- Web UI ----
WEB_HTML = r"""
<!doctype html><html><head><meta charset="utf-8"/>
<title>BGGG</title>
<style>
:root{--amber:#ffbf00;--cyan:#00ffff;--mag:#ff00ff;--bg:#000;--card:#111}
html,body{margin:0;padding:0;background:var(--bg);color:var(--amber);font-family:Consolas,Menlo,monospace}
.wrap{max-width:1100px;margin:24px auto;padding:0 16px}
.banner{white-space:pre;color:var(--amber);line-height:1}
.card{background:var(--card);border:1px solid #222;padding:16px;margin:16px 0;border-radius:8px}
.title{color:var(--mag);font-weight:bold;margin-bottom:8px}
label{display:block;color:var(--cyan);margin:6px 0 2px}
input,textarea{width:100%;background:#000;color:var(--amber);border:1px solid #333;border-radius:6px;padding:8px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.btn{background:#111;border:1px solid var(--mag);color:var(--amber);padding:8px 12px;border-radius:8px;cursor:pointer}
.btn:hover{border-color:var(--cyan)}
.logs{background:#000;border:1px solid #333;padding:8px;height:240px;overflow:auto;white-space:pre-wrap}
.footer{color:#777}
a{color:var(--cyan)}
</style></head><body>
<div class="wrap">
  <div class="banner">{{banner}}</div>

  <div class="card">
    <div class="title">Configuration</div>
    <form method="post" action="{{ url_for('save_cfg') }}">
      <div class="grid">
        <div>
          <label>ComfyUI root</label><input name="comfy_root" value="{{cfg.comfy_root}}"/>
          <label>Input video</label><input name="input_video" value="{{cfg.input_video}}"/>
          <label>Frames dir</label><input name="frames_dir" value="{{cfg.frames_dir}}"/>
          <label>Frames pattern</label><input name="frames_glob" value="{{cfg.frames_glob}}"/>
          <label>Upscaled frames dir</label><input name="upscale_out_dir" value="{{cfg.upscale_out_dir}}"/>
          <label>Output video</label><input name="output_video" value="{{cfg.output_video}}"/>
          <label>Watch folder</label><input name="watch_folder" value="{{cfg.watch_folder}}"/>
        </div>
        <div>
          <label>Comfy host</label><input name="comfy_host" value="{{cfg.comfy_host}}"/>
          <label>Comfy port</label><input name="comfy_port" value="{{cfg.comfy_port}}"/>
          <label>FPS override (0=auto)</label><input name="fps_override" value="{{cfg.fps_override}}"/>
          <label>Copy audio (y/n)</label><input name="audio_copy" value="{{ 'y' if cfg.audio_copy else 'n' }}"/>
          <label>Debug (y/n)</label><input name="debug" value="{{ 'y' if cfg.debug else 'n' }}"/>
          <label>Web port</label><input name="web_port" value="{{cfg.web_port}}"/>
        </div>
      </div>
      <div class="title" style="margin-top:10px">ESRGAN</div>
      <div class="grid">
        <div><label>Upscaler model filename</label><input name="esrgan_model_name" value="{{cfg.esrgan_model_name}}"/></div>
        <div>
          <label>Extra scale</label><input name="esrgan_extra_scale" value="{{cfg.esrgan_extra_scale}}"/>
          <label>Batch pause (s)</label><input name="esrgan_batch_pause" value="{{cfg.esrgan_batch_pause}}"/>
        </div>
      </div>
      <div class="title" style="margin-top:10px">SDXL + Tiled</div>
      <div class="grid">
        <div>
          <label>Checkpoint filename</label><input name="sdxl_ckpt_name" value="{{cfg.sdxl_ckpt_name}}"/>
          <label>Positive</label><textarea name="sdxl_positive">{{cfg.sdxl_positive}}</textarea>
          <label>Negative</label><textarea name="sdxl_negative">{{cfg.sdxl_negative}}</textarea>
        </div>
        <div>
          <label>Steps</label><input name="sdxl_steps" value="{{cfg.sdxl_steps}}"/>
          <label>CFG</label><input name="sdxl_cfg" value="{{cfg.sdxl_cfg}}"/>
          <label>Sampler</label><input name="sdxl_sampler_name" value="{{cfg.sdxl_sampler_name}}"/>
          <label>Scheduler</label><input name="sdxl_scheduler" value="{{cfg.sdxl_scheduler}}"/>
          <label>Denoise</label><input name="sdxl_denoise" value="{{cfg.sdxl_denoise}}"/>
          <label>Latent upscale by</label><input name="sdxl_latent_upscale_by" value="{{cfg.sdxl_latent_upscale_by}}"/>
          <label>Seed (0=random)</label><input name="sdxl_seed" value="{{cfg.sdxl_seed}}"/>
          <label>Use Tiled VAE (y/n)</label><input name="sdxl_use_tiled_vae" value="{{ 'y' if cfg.sdxl_use_tiled_vae else 'n' }}"/>
          <label>Tile size</label><input name="sdxl_tile_size" value="{{cfg.sdxl_tile_size}}"/>
        </div>
      </div>
      <p><button class="btn" type="submit">Save</button></p>
    </form>
  </div>

  <div class="card">
    <div class="title">Actions</div>
    <form method="post" action="{{ url_for('do_action') }}">
      <button class="btn" name="action" value="extract">Extract</button>
      <button class="btn" name="action" value="esrgan">Upscale ESRGAN</button>
      <button class="btn" name="action" value="sdxl">Upscale SDXL</button>
      <button class="btn" name="action" value="recombine">Recombine</button>
      <button class="btn" name="action" value="test">Test ComfyUI</button>
      <button class="btn" name="action" value="toggle_web">Stop Web</button>
    </form>
  </div>

  <div class="card">
    <div class="title">Logs</div>
    <div class="logs">{{logs}}</div>
  </div>

  <div class="footer">BGGG v{{version}} • <a href="{{ url_for('help_page') }}">Help</a></div>
</div>
</body></html>
"""
HELP_HTML = r"""<!doctype html><html><head><meta charset="utf-8"/><title>Help</title>
<style>body{background:#000;color:#ffbf00;font-family:Consolas,Menlo,monospace}.wrap{max-width:900px;margin:24px auto;padding:0 16px}a{color:#00ffff}</style>
<body><div class="wrap"><pre>{{help}}</pre><p><a href="{{ url_for('home') }}">« Back</a></p></div></body></html>"""

class WebThread(threading.Thread):
    def __init__(self, app: BGGGApp):
        super().__init__(daemon=True); self.app=app; self.flask=Flask(__name__); self._routes()
    def _routes(self):
        app=self.app; flask=self.flask
        @flask.route("/")
        def home():
            try: logs=Path("logs/bggg.log").read_text(encoding="utf-8")[-9000:]
            except Exception: logs=""
            return render_template_string(WEB_HTML, banner=ASCII_BANNER, cfg=app.cfg, version=VERSION, logs=logs)
        @flask.route("/help")
        def help_page(): return render_template_string(HELP_HTML, help=HELP_TEXT)
        # noinspection PyTypeChecker
        @flask.route("/save_cfg", methods=["POST"])
        def save_cfg():
            f=flask_request.form
            # bool cast
            def yn(x,d): 
                t=(x or "").strip().lower()
                if t in ("y","yes","true","1","on"): return True
                if t in ("n","no","false","0","off"): return False
                return d
            def to_int(x,d):
                try: return int(str(x).strip())
                except: return d
            def to_float(x,d):
                try: return float(str(x).strip())
                except: return d
            app.cfg.comfy_root=f.get("comfy_root", app.cfg.comfy_root)
            app.cfg.input_video=f.get("input_video", app.cfg.input_video)
            app.cfg.frames_dir=f.get("frames_dir", app.cfg.frames_dir)
            app.cfg.frames_glob=f.get("frames_glob", app.cfg.frames_glob)
            app.cfg.upscale_out_dir=f.get("upscale_out_dir", app.cfg.upscale_out_dir)
            app.cfg.output_video=f.get("output_video", app.cfg.output_video)
            app.cfg.watch_folder=f.get("watch_folder", app.cfg.watch_folder)
            app.cfg.comfy_host=f.get("comfy_host", app.cfg.comfy_host)
            app.cfg.comfy_port=to_int(f.get("comfy_port", app.cfg.comfy_port), app.cfg.comfy_port)
            app.cfg.web_port=to_int(f.get("web_port", app.cfg.web_port), app.cfg.web_port)
            app.cfg.fps_override=to_float(f.get("fps_override", app.cfg.fps_override), app.cfg.fps_override)
            app.cfg.audio_copy=yn(f.get("audio_copy"), app.cfg.audio_copy)
            app.cfg.debug=yn(f.get("debug"), app.cfg.debug)
            app.cfg.esrgan_model_name=f.get("esrgan_model_name", app.cfg.esrgan_model_name)
            app.cfg.esrgan_extra_scale=to_float(f.get("esrgan_extra_scale", app.cfg.esrgan_extra_scale), app.cfg.esrgan_extra_scale)
            app.cfg.esrgan_batch_pause=to_float(f.get("esrgan_batch_pause", app.cfg.esrgan_batch_pause), app.cfg.esrgan_batch_pause)
            app.cfg.sdxl_ckpt_name=f.get("sdxl_ckpt_name", app.cfg.sdxl_ckpt_name)
            app.cfg.sdxl_positive=f.get("sdxl_positive", app.cfg.sdxl_positive)
            app.cfg.sdxl_negative=f.get("sdxl_negative", app.cfg.sdxl_negative)
            app.cfg.sdxl_steps=to_int(f.get("sdxl_steps", app.cfg.sdxl_steps), app.cfg.sdxl_steps)
            app.cfg.sdxl_cfg=to_float(f.get("sdxl_cfg", app.cfg.sdxl_cfg), app.cfg.sdxl_cfg)
            app.cfg.sdxl_sampler_name=f.get("sdxl_sampler_name", app.cfg.sdxl_sampler_name)
            app.cfg.sdxl_scheduler=f.get("sdxl_scheduler", app.cfg.sdxl_scheduler)
            app.cfg.sdxl_denoise=to_float(f.get("sdxl_denoise", app.cfg.sdxl_denoise), app.cfg.sdxl_denoise)
            app.cfg.sdxl_latent_upscale_by=to_float(f.get("sdxl_latent_upscale_by", app.cfg.sdxl_latent_upscale_by), app.cfg.sdxl_latent_upscale_by)
            app.cfg.sdxl_seed=to_int(f.get("sdxl_seed", app.cfg.sdxl_seed), app.cfg.sdxl_seed)
            app.cfg.sdxl_use_tiled_vae=yn(f.get("sdxl_use_tiled_vae"), app.cfg.sdxl_use_tiled_vae)
            app.cfg.sdxl_tile_size=to_int(f.get("sdxl_tile_size", app.cfg.sdxl_tile_size), app.cfg.sdxl_tile_size)
            app.cfg.save(); app.log.info("Config saved via Web UI."); return redirect(url_for('home'))
        @flask.route("/do_action", methods=["POST"])
        def do_action():
            act=flask_request.form.get("action","")
            if act=="extract": self.app.action_extract()
            elif act=="esrgan": self.app.action_esrgan()
            elif act=="sdxl": self.app.action_sdxl()
            elif act=="recombine": self.app.action_recombine()
            elif act=="test": self.app.action_test()
            elif act=="toggle_web":
                func = flask_request.environ.get("werkzeug.server.shutdown")
                if func: func()
            return redirect(url_for('home'))
    def run(self, port:int):
        import logging as _log; _log.getLogger("werkzeug").setLevel(_log.ERROR)
        self.flask.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

def start_web(app: BGGGApp):
    wt = WebThread(app)
    t = threading.Thread(target=lambda: wt.run(app.cfg.web_port), daemon=True)
    t.start(); app.cfg.web_running=True; return t

def stop_web(cfg: AppConfig):
    try: requests.post(f"http://127.0.0.1:{cfg.web_port}/do_action", data={"action":"toggle_web"}, timeout=2)
    except Exception: pass
    cfg.web_running=False

# ---- CRT GUI (pygame) ----
class CRTGUI:
    W,H = 1200,720
    COL_AMBER=(255,191,0); COL_CYAN=(0,255,255); COL_MAG=(255,0,255); COL_BLACK=(0,0,0)
    COL_PANEL=(16,16,16); COL_DIM=(140,110,0)
    def __init__(self, app: BGGGApp):
        if pygame is None: raise RuntimeError("pygame not installed.")
        self.app=app; self.scale=app.cfg.gui_scale or 1.0
        pygame.init(); pygame.display.set_caption(f"{APP_NAME} v{VERSION}")
        self.screen=pygame.display.set_mode((int(self.W*self.scale), int(self.H*self.scale)))
        self.clock=pygame.time.Clock()
        self.font=pygame.font.SysFont("Consolas", int(18*self.scale))
        self.font_big=pygame.font.SysFont("Consolas", int(22*self.scale), bold=True)
        self.bg=self._img(ASSET_BG); self.logo=self._img(ASSET_LOGO)
        self.scan=self._scanlines(); self._build_layout()

    def _img(self, path: Path):
        try:
            if path.exists():
                img=pygame.image.load(str(path)).convert_alpha()
                return pygame.transform.smoothscale(img, (int(img.get_width()*self.scale), int(img.get_height()*self.scale)))
        except Exception: ...
        s=pygame.Surface((10,10)); s.fill(self.COL_BLACK); return s
    def _scanlines(self):
        s=pygame.Surface((self.W,self.H), pygame.SRCALPHA)
        for y in range(0,self.H,4):
            a=20+int(10*random.random()); pygame.draw.line(s,(255,255,255,a),(0,y),(self.W,y))
        return pygame.transform.smoothscale(s,(int(self.W*self.scale), int(self.H*self.scale)))

    def _build_layout(self):
        x0,y0=int(20*self.scale),int(130*self.scale); w,h=int(260*self.scale),int(40*self.scale); gap=int(10*self.scale)
        keys=["cfg","extract","esrgan","sdxl","recombine","web","help","test","exit","watch"]
        self.rects={k: pygame.Rect(x0, y0+i*(h+gap), w, h) for i,k in enumerate(keys)}
        self.logs = pygame.Rect(int(300*self.scale), int(130*self.scale), int(self.W*self.scale-320*self.scale), int(self.H*self.scale-160*self.scale))

    def _btn(self, rect, label, active=False):
        pygame.draw.rect(self.screen, self.COL_PANEL, rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COL_MAG if active else (40,40,40), rect, width=1, border_radius=8)
        self.screen.blit(self.font.render(label, True, self.COL_CYAN if active else self.COL_AMBER), (rect.x+int(12*self.scale), rect.y+int(10*self.scale)))

    def _render_logs(self):
        pygame.draw.rect(self.screen,(12,12,12),self.logs,border_radius=8)
        pygame.draw.rect(self.screen,(40,40,40),self.logs,width=1,border_radius=8)
        try: txt=Path("logs/bggg.log").read_text(encoding="utf-8")[-7000:]
        except Exception: txt=""
        y=self.logs.y+int(8*self.scale); maxw=self.logs.width-int(16*self.scale)
        for line in txt.splitlines()[-220:]:
            # simple wrap
            words=line.split(" "); cur=""
            while words:
                w=words.pop(0); test=(cur+" "+w).strip()
                if self.font.size(test)[0]>maxw and cur: 
                    self.screen.blit(self.font.render(cur, True, self.COL_AMBER),(self.logs.x+int(8*self.scale),y))
                    y+=self.font.get_height()+2; cur=w
                else: cur=test
            if cur:
                self.screen.blit(self.font.render(cur, True, self.COL_AMBER),(self.logs.x+int(8*self.scale),y))
                y+=self.font.get_height()+2
            if y>self.logs.bottom-int(12*self.scale): break

    def _choose_mode_popup(self) -> str:
        txt="Choose watcher mode:\n\n  1) ESRGAN\n  2) SDXL\n\nPress 1 or 2 (Esc/Click = cancel, defaults to ESRGAN)."
        surf=self._wrap(txt, int(900*self.scale))
        overlay=pygame.Surface(self.screen.get_size(),pygame.SRCALPHA); overlay.fill((0,0,0,200))
        rect=surf.get_rect(); rect.center=(self.screen.get_width()//2, self.screen.get_height()//2)
        box=rect.inflate(int(40*self.scale), int(40*self.scale))
        self.screen.blit(overlay,(0,0))
        pygame.draw.rect(self.screen,self.COL_PANEL,box,border_radius=12)
        pygame.draw.rect(self.screen,(40,40,40),box,width=1,border_radius=12)
        cap=self.font_big.render("Watcher Mode", True, self.COL_MAG); self.screen.blit(cap,(box.x+int(14*self.scale), box.y+int(10*self.scale)))
        self.screen.blit(surf,(box.x+int(20*self.scale), box.y+int(40*self.scale))); pygame.display.flip()
        while True:
            for e in pygame.event.get():
                if e.type==pygame.QUIT: return "esrgan"
                if e.type==pygame.KEYDOWN:
                    if e.key==pygame.K_1: return "esrgan"
                    if e.key==pygame.K_2: return "sdxl"
                    return "esrgan"
                if e.type==pygame.MOUSEBUTTONDOWN: return "esrgan"
            self.clock.tick(60)

    def _wrap(self, text, maxw):
        lines=[]
        for para in text.splitlines():
            words=para.split(" "); cur=""
            for w in words:
                test=(cur+" "+w).strip()
                if self.font.size(test)[0]>maxw and cur: lines.append(cur); cur=w
                else: cur=test
            lines.append(cur)
        h=self.font.get_height()+4
        surf=pygame.Surface((maxw, h*len(lines)), pygame.SRCALPHA); y=0
        for ln in lines: surf.blit(self.font.render(ln, True, self.COL_AMBER),(0,y)); y+=h
        return surf

    def run(self):
        boot=pygame.Surface(self.screen.get_size(),pygame.SRCALPHA); boot.fill((255,255,255,40))
        t0=time.time()
        while time.time()-t0<0.2:
            self._frame(draw_overlay=False); self.screen.blit(boot,(0,0)); pygame.display.flip(); self.clock.tick(60)
        while True:
            action=self._frame()
            if action=="exit": break
            if action: self._handle(action)
        pygame.quit()

    def _frame(self, draw_overlay=True):
        for e in pygame.event.get():
            if e.type==pygame.QUIT: return "exit"
            if e.type==pygame.KEYDOWN:
                m={pygame.K_1:"cfg",pygame.K_2:"extract",pygame.K_3:"esrgan",pygame.K_4:"sdxl",pygame.K_5:"recombine",pygame.K_6:"web",pygame.K_7:"help",pygame.K_8:"test",pygame.K_0:"exit"}
                if e.key in m: return m[e.key]
            if e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
                for k,r in self.rects.items():
                    if r.collidepoint(e.pos): return k
        self.screen.fill(self.COL_BLACK)
        if self.bg: self.screen.blit(pygame.transform.smoothscale(self.bg,self.screen.get_size()),(0,0))
        if self.logo: self.screen.blit(self.logo,(int(20*self.scale), int(20*self.scale)))
        title=self.font_big.render(f"{APP_NAME} v{VERSION}", True, self.COL_AMBER); self.screen.blit(title,(int(220*self.scale), int(35*self.scale)))
        sub=self.font.render("Retro CRT • amber/cyan/magenta", True, self.COL_DIM); self.screen.blit(sub,(int(220*self.scale), int(65*self.scale)))
        labels={"cfg":"1 Configure","extract":"2 Extract","esrgan":"3 Upscale ESRGAN","sdxl":"4 Upscale SDXL",
                "recombine":"5 Recombine","web":("Stop Web" if self.app.cfg.web_running else "6 Start Web"),
                "help":"7 Help","test":"8 Test Comfy","exit":"0 Exit","watch":"[Toggle] Folder Watcher"}
        for k,r in self.rects.items(): self._btn(r, labels[k])
        self._render_logs()
        if draw_overlay:
            if random.random()<0.02: self.scan=self._scanlines()
            self.screen.blit(self.scan,(0,0), special_flags=pygame.BLEND_PREMULTIPLIED)
        pygame.display.flip(); self.clock.tick(60); return None

    def _handle(self, action: str):
        if action=="cfg":
            ups,ck=self.app.discover_models()
            if ups: self.app.cfg.esrgan_model_name=ups[0]
            if ck:  self.app.cfg.sdxl_ckpt_name=ck[0]
            self.app.cfg.save(); self.app.log.info("Config refreshed from autodiscovery.")
        elif action=="extract": self.app.action_extract()
        elif action=="esrgan":  self.app.action_esrgan()
        elif action=="sdxl":    self.app.action_sdxl()
        elif action=="recombine": self.app.action_recombine()
        elif action=="web":
            if self.app.cfg.web_running: stop_web(self.app.cfg); self.app.log.info("Stopping Web UI…")
            else: start_web(self.app); self.app.log.info(f"Web UI on http://127.0.0.1:{self.app.cfg.web_port}")
        elif action=="help":
            self._popup("HELP", HELP_TEXT)
        elif action=="test": self.app.action_test()
        elif action=="watch":
            if self.app._watcher or self.app._poll_thread: self.app.stop_watcher()
            else:
                mode=self._choose_mode_popup(); self.app.start_watcher(lambda: mode)

    def _popup(self, title: str, text: str):
        surf=self._wrap(text, int(900*self.scale))
        overlay=pygame.Surface(self.screen.get_size(), pygame.SRCALPHA); overlay.fill((0,0,0,200))
        rect=surf.get_rect(); rect.center=(self.screen.get_width()//2, self.screen.get_height()//2)
        box=rect.inflate(int(40*self.scale), int(40*self.scale))
        self.screen.blit(overlay,(0,0))
        pygame.draw.rect(self.screen,self.COL_PANEL,box,border_radius=12)
        pygame.draw.rect(self.screen,(40,40,40),box,width=1,border_radius=12)
        cap=self.font_big.render(title, True, self.COL_MAG); self.screen.blit(cap,(box.x+int(14*self.scale), box.y+int(10*self.scale)))
        self.screen.blit(surf,(box.x+int(20*self.scale), box.y+int(40*self.scale))); pygame.display.flip()
        waiting=True
        while waiting:
            for e in pygame.event.get():
                if e.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.QUIT): waiting=False
            self.clock.tick(60)

# ---- Orchestration ----
def menu_loop(app: BGGGApp):
    tui_banner(); app.log.info(f"Welcome to {APP_NAME} v{VERSION}")
    if not app.deps_ok(): print(CRT_MAG+"\nInstall missing dependencies and re-run."+CRT_RESET); return
    ensure_dir(Path(app.cfg.frames_dir)); ensure_dir(Path(app.cfg.upscale_out_dir))
    web_thread=None
    while True:
        tui_menu(app.cfg)
        ch=ask("Select option [1-9, 0=Exit]:")
        if ch=="1": configure_interactive(app)
        elif ch=="2": app.action_extract()
        elif ch=="3": app.action_esrgan()
        elif ch=="4": app.action_sdxl()
        elif ch=="5": app.action_recombine()
        elif ch=="6":
            if app.cfg.web_running: stop_web(app.cfg); app.log.info("Stopping Web UI…")
            else: start_web(app); app.log.info(f"Web UI on http://127.0.0.1:{app.cfg.web_port}")
        elif ch=="7": print(HELP_TEXT)
        elif ch=="8": app.action_test()
        elif ch=="9":
            if not app.deps_ok(require_gui=True): continue
            try: CRTGUI(app).run()
            except Exception as e: app.log.error(f"GUI error: {e}")
        elif ch=="0": app.log.info("Goodbye."); break
        else: print(CRT_MAG+"Unknown selection."+CRT_RESET)
        print("")

def main():
    cfg=AppConfig.load(); app=BGGGApp(cfg)
    args=set(sys.argv[1:])
    if "--debug" in args: cfg.debug=True
    if "--web" in args:
        if app.deps_ok(): start_web(app); print(CRT_CYAN+f"Web UI: http://127.0.0.1:{cfg.web_port}  (Ctrl+C to quit)"+CRT_RESET)
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt: stop_web(cfg)
        return
    if "--gui" in args:
        if not app.deps_ok(require_gui=True): return
        try: CRTGUI(app).run()
        except KeyboardInterrupt: pass
        return
    try: menu_loop(app)
    except KeyboardInterrupt: print("\n"+CRT_MAG+"Interrupted."+CRT_RESET)
    except Exception as e: print(CRT_MAG+f"Fatal: {e}"+CRT_RESET); traceback.print_exc()

if __name__ == "__main__":
    main()
