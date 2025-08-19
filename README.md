# 📺 BGGG Video Upscaler v5.5

**The BG Gremlin Group (BGGG)** Batch Upscaler is a powerful, flexible tool for upscaling videos on Windows using **ComfyUI**. It extracts frames with `ffmpeg`, upscales them with **ESRGAN** or **SDXL + Tiled VAE**, and recombines them into a high-quality video with optional audio preservation. With three slick interfaces—**TUI** (retro Command Prompt/Windows Terminal), **Web UI** (browser-based), and **CRT GUI** (retro-styled with scanlines)—BGGG brings a nostalgic yet modern vibe to video processing. 🚀

**Developed by the BG Gremlin Group: *Creating Unique Tools for Unique Individuals***

---

## ✨ Features

- **End-to-End Pipeline**:
  - 🖼️ **Extract**: Splits videos into frames using `ffmpeg` (`frames\frame_%06d.png`).
  - 🔍 **Upscale**: Processes frames via ComfyUI using ESRGAN or SDXL workflows.
  - 🎥 **Recombine**: Rebuilds the video (`output_upscaled.mp4`), preserving audio if desired.

- **Upscaling Modes**:
  - **ESRGAN**: Fast, model-based upscaling with optional bicubic scaling for extra resolution.
  - **SDXL + Tiled VAE**: Advanced upscaling with latent diffusion, supporting tiled VAE for low-VRAM systems.

- **Interfaces**:
  - **TUI**: Retro number-menu interface with colorful ANSI output in Windows Terminal. 🖥️
  - **Web UI**: Browser-based control at `http://127.0.0.1:7860`. 🌐
  - **CRT GUI**: Pygame-powered retro interface with scanlines, logo, and amber/cyan/magenta aesthetics. 📟

- **Automation**:
  - 📂 **Folder Watcher**: Automatically processes new videos in a watched folder using `watchdog` or polling fallback.
  - 🔄 **Configurable**: Save/load settings via JSON, with autodiscovery for ComfyUI models.

- **Robustness**:
  - 🛡️ Lazy imports for optional dependencies (`flask`, `pygame`, `watchdog`).
  - 📜 Detailed logging to `logs\bggg.log` with color-coded output in Windows Terminal.
  - 🔄 Graceful error handling for missing files, ComfyUI failures, or dependency issues.

---

## 🛠️ Requirements (Windows)

| Component          | Requirement                              | Notes                                                                 |
|--------------------|------------------------------------------|----------------------------------------------------------------------|
| **Python**        | 3.9+                                    | Install from [python.org](https://www.python.org/downloads/). Add to PATH. |
| **ffmpeg**        | Installed and in PATH                   | Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/). Add `ffmpeg.exe` to PATH. |
| **ComfyUI**       | Running (default `http://127.0.0.1:8188`)| Models in `ComfyUI\models\upscale_models` or `checkpoints`.           |
| **Dependencies**  | `requests`, `flask`                     | Required. Install via `pip install -r requirements.txt`.              |
| **Optional**      | `colorama`, `pygame`, `watchdog`        | For TUI colors, CRT GUI, and watcher. Install via `requirements.txt`. |

### Optional Assets
- 📁 `assets\logo_bggg.png`: Logo displayed in CRT GUI (top-left).
- 📁 `assets\bg_crt.png`: Amber tech grid background for CRT GUI.
- *Note*: Missing assets are handled gracefully (replaced with black surfaces).

### Model Files
- **ESRGAN**: `4x-UltraSharp.pth` in `ComfyUI\models\upscale_models`.
- **SDXL**: `sd_xl_base_1.0.safetensors` in `ComfyUI\models\checkpoints`.

---

## 🚀 Installation (Windows)

1. **Install Python 3.9+**:
   - Download from [python.org](https://www.python.org/downloads/).
   - Check "Add Python to PATH" during installation.
   - Verify with:
     ```cmd
     python --version
     ```

2. **Clone or Download**:
   ```cmd
   git clone https://github.com/your-repo/bggg.git
   cd bggg
   ```
   *Note*: If Git is not installed, download the ZIP from the repository and extract it.

3. **Install Dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

4. **Install ffmpeg**:
   - Download a Windows build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (e.g., `ffmpeg-release-essentials.zip`).
   - Extract to `C:\ffmpeg`.
   - Add `C:\ffmpeg\bin` to the system PATH:
     1. Open Control Panel → System → Advanced system settings → Environment Variables.
     2. Under "System variables," edit `Path` and add `C:\ffmpeg\bin`.
   - Verify with:
     ```cmd
     ffmpeg -version
     ```

5. **Set Up ComfyUI**:
   - Clone or download ComfyUI from [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI).
   - Place models in the correct folders:
     - `4x-UltraSharp.pth` → `ComfyUI\models\upscale_models`
     - `sd_xl_base_1.0.safetensors` → `ComfyUI\models\checkpoints`
   - Start ComfyUI:
     ```cmd
     cd C:\path\to\ComfyUI
     python main.py
     ```

6. **Create Assets Directory** (optional):
   ```cmd
   mkdir assets
   ```
   - Place `logo_bggg.png` and `bg_crt.png` in `assets\`.

---

## 📚 Usage (Windows)

### Running BGGG
Run in Command Prompt, PowerShell, or Windows Terminal (recommended for UTF-8 and color support):

- **TUI (Default)**:
  ```cmd
  python bggg.py
  ```
  Displays a retro number-menu interface. Select options (1-9, 0 to exit).

- **Web UI**:
  ```cmd
  python bggg.py --web
  ```
  Opens a browser interface at `http://127.0.0.1:7860`. Configure settings, trigger actions, and view logs.

- **CRT GUI**:
  ```cmd
  python bggg.py --gui
  ```
  Launches a retro-styled GUI with scanlines and clickable buttons. Requires `pygame`.

- **Debug Mode**:
  ```cmd
  python bggg.py --debug
  ```
  Enables verbose console logging for debugging.

### Pipeline Steps
1. **Extract Frames**:
   - Splits input video into PNG frames (`frames\frame_%06d.png`).
   - Uses `ffmpeg` with optional FPS override (default: auto-detected or 30.0).

2. **Upscale Frames**:
   - **ESRGAN**: Uses `UpscaleModelLoader` → `ImageUpscaleWithModel` → (optional) `ImageScale(bicubic)`.
   - **SDXL**: Uses `CheckpointLoaderSimple` → `(Tiled)VAEEncode` → `LatentUpscaleBy` → `KSampler` → `(Tiled)VAEDecode`.
   - Saves upscaled frames to `frames_upscaled\frame_%06d.png`.

3. **Recombine Frames**:
   - Rebuilds video (`output_upscaled.mp4`) with `ffmpeg`.
   - Copies audio from source if enabled.

### Configuration
Settings are stored in `bggg_config.json`. Default values:

| Category       | Key                       | Default Value                          | Description                                    |
|----------------|---------------------------|----------------------------------------|------------------------------------------------|
| **ComfyUI**    | `comfy_host`             | `127.0.0.1`                           | ComfyUI server host.                           |
|                | `comfy_port`             | `8188`                                | ComfyUI server port.                           |
|                | `comfy_root`             | `""`                                  | ComfyUI root (e.g., `C:\ComfyUI`).             |
| **Paths**      | `input_video`            | `""`                                  | Input video file or folder (e.g., `C:\Videos\video.mp4`). |
|                | `frames_dir`             | `frames`                              | Directory for extracted frames.                |
|                | `frames_glob`            | `frame_%06d.png`                      | Frame filename pattern.                        |
|                | `upscale_out_dir`        | `frames_upscaled`                     | Directory for upscaled frames.                 |
|                | `output_video`           | `output_upscaled.mp4`                 | Output video path.                             |
|                | `watch_folder`           | `""`                                  | Folder to watch (e.g., `C:\Videos\watch`).     |
| **Video**      | `fps_override`           | `0.0`                                 | FPS for extraction/recombination (0=auto).     |
|                | `audio_copy`             | `True`                                | Copy audio from source video.                  |
| **ESRGAN**     | `esrgan_model_name`      | `4x-UltraSharp.pth`                   | Upscaler model filename.                       |
|                | `esrgan_extra_scale`     | `1.0`                                 | Additional bicubic scaling factor.             |
|                | `esrgan_batch_pause`     | `0.0`                                 | Pause (seconds) between frames to avoid overload. |
| **SDXL**       | `sdxl_ckpt_name`         | `sd_xl_base_1.0.safetensors`          | SDXL checkpoint filename.                      |
|                | `sdxl_positive`          | `highly detailed, sharp, ...`         | Positive prompt for SDXL.                      |
|                | `sdxl_negative`          | `blur, noise, artifacts, ...`         | Negative prompt for SDXL.                      |
|                | `sdxl_steps`             | `18`                                  | Sampling steps.                                |
|                | `sdxl_cfg`               | `4.5`                                 | CFG scale.                                     |
|                | `sdxl_sampler_name`      | `dpmpp_2m`                           | Sampler name.                                  |
|                | `sdxl_scheduler`         | `karras`                              | Scheduler name.                                |
|                | `sdxl_denoise`           | `0.4`                                 | Denoise strength (0-1).                        |
|                | `sdxl_latent_upscale_by` | `2.0`                                 | Latent upscale factor.                         |
|                | `sdxl_seed`              | `0`                                   | Seed (0=random).                               |
|                | `sdxl_use_tiled_vae`     | `True`                                | Use tiled VAE for low VRAM.                    |
|                | `sdxl_tile_size`         | `512`                                 | Tiled VAE tile size.                           |
| **Misc**       | `debug`                  | `False`                               | Enable verbose console logging.                |
|                | `gui_scale`              | `1.0`                                 | CRT GUI scaling factor.                        |
|                | `web_port`               | `7860`                                | Web UI port.                                   |

### Example Workflow
1. **Configure**:
   - Set `input_video` to `C:\Videos\video.mp4`.
   - Set `comfy_root` to `C:\ComfyUI`.
   - Choose `esrgan_model_name` or `sdxl_ckpt_name` based on available models.

2. **Run TUI**:
   ```cmd
   python bggg.py
   ```
   - Select **1** to configure.
   - Select **2** to extract frames.
   - Select **3** (ESRGAN) or **4** (SDXL) to upscale.
   - Select **5** to recombine into `output_upscaled.mp4`.

3. **Run Web UI**:
   ```cmd
   python bggg.py --web
   ```
   - Open `http://127.0.0.1:7860` in a browser.
   - Enter settings, click **Save**, then trigger actions (Extract, Upscale, Recombine).

4. **Run CRT GUI**:
   ```cmd
   python bggg.py --gui
   ```
   - Click buttons or press 1-9, 0 to navigate.

5. **Watch Folder**:
   - Set `watch_folder` to `C:\Videos\watch` in config.
   - Select **Watch** in CRT GUI or enable via TUI/Web.
   - Copy videos to `watch_folder` to auto-process.

---

## 🎨 Interfaces

### TUI (Command Prompt/Windows Terminal)
- Retro number-menu with ANSI colors (amber, cyan, magenta; best in Windows Terminal with `colorama`).
- Options: Configure, Extract, Upscale (ESRGAN/SDXL), Recombine, Web UI toggle, Help, Test ComfyUI, CRT GUI, Exit.
- Logs to `logs\bggg.log` and console.

### Web UI
- Accessible at `http://127.0.0.1:7860` (requires `flask`).
- Features:
  - Form-based configuration with grid layout.
  - Action buttons for pipeline steps.
  - Live log display (last 9000 characters).
  - Help page with pipeline details.

### CRT GUI
- Retro-styled interface with pygame, featuring:
  - Scanline overlay for CRT aesthetic.
  - Logo and background image support (`assets\logo_bggg.png`, `assets\bg_crt.png`).
  - Clickable buttons and keyboard shortcuts (1-9, 0).
  - Log panel with basic text wrapping.
- Requires `pygame`.

---

## 🛠️ Pipeline Details

### ESRGAN Workflow
- **Nodes**: `UpscaleModelLoader` → `ImageUpscaleWithModel` → (optional) `ImageScale(bicubic)` → `SaveImage`.
- **Notes**:
  - Uses vanilla ComfyUI nodes (no tiling inputs).
  - `esrgan_extra_scale` adds bicubic scaling if ≠ 1.0.
  - Default model: `4x-UltraSharp.pth`.

### SDXL + Tiled Workflow
- **Nodes**: `CheckpointLoaderSimple` → `(Tiled)VAEEncode` → `LatentUpscaleBy` → `KSampler` → `(Tiled)VAEDecode` → `SaveImage`.
- **Notes**:
  - VAE correctly wired from checkpoint to encode/decode.
  - Tiled VAE (optional) reduces VRAM usage.
  - Default checkpoint: `sd_xl_base_1.0.safetensors`.
  - Configurable prompts, steps, CFG, denoise, etc.

### Tips
- **SDXL**: Use 18–30 steps, CFG 4–6, denoise 0.3–0.5, latent scale 2.0. Tiled VAE with 512–768 tile size for low VRAM.
- **ESRGAN**: Set `esrgan_extra_scale=1.0` for native model scaling. Use `esrgan_batch_pause` (e.g., 0.5s) if GPU/CPU is overloaded.
- **Storage**: Use SSD/NVMe for `frames_dir` and `upscale_out_dir` to speed up IO.

---

## 🐛 Troubleshooting (Windows)

| Issue                              | Solution                                                                 |
|------------------------------------|--------------------------------------------------------------------------|
| **Missing ffmpeg**                 | Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), extract to `C:\ffmpeg`, add `C:\ffmpeg\bin` to PATH. Check with `ffmpeg -version`. |
| **ComfyUI not reachable**          | Start ComfyUI (`cd C:\ComfyUI && python main.py`). Verify `comfy_host`/`comfy_port` in config. |
| **Missing dependencies**           | Run `pip install -r requirements.txt` in Command Prompt or PowerShell.   |
| **CRT GUI fails**                  | Install `pygame` (`pip install pygame`). Ensure `assets\` has `logo_bggg.png` and `bg_crt.png`. |
| **Watcher not detecting files**    | Install `watchdog` (`pip install watchdog`). Verify `watch_folder` path (e.g., `C:\Videos\watch`). |
| **Unicode display issues**         | Use Windows Terminal (not CMD) for proper UTF-8 and color support.       |
| **Web UI port conflict**           | Change `web_port` in `bggg_config.json` or stop conflicting service (e.g., `netstat -ano | findstr :7860`). |

### Common Errors
- **Log file too large**: CRT GUI may lag with large `logs\bggg.log`. Delete or truncate manually.
- **ComfyUI timeout**: Increase `timeout` in `ComfyClient` methods or check ComfyUI server load.
- **Model not found**: Place `4x-UltraSharp.pth` in `ComfyUI\models\upscale_models` and `sd_xl_base_1.0.safetensors` in `ComfyUI\models\checkpoints`.

---

## 📝 Example Configuration (`bggg_config.json`)

```json
{
  "comfy_host": "127.0.0.1",
  "comfy_port": 8188,
  "web_port": 7860,
  "comfy_root": "C:\\ComfyUI",
  "input_video": "C:\\Videos\\video.mp4",
  "frames_dir": "frames",
  "frames_glob": "frame_%06d.png",
  "upscale_out_dir": "frames_upscaled",
  "output_video": "output_upscaled.mp4",
  "watch_folder": "C:\\Videos\\watch",
  "fps_override": 0.0,
  "audio_copy": true,
  "esrgan_model_name": "4x-UltraSharp.pth",
  "esrgan_extra_scale": 1.0,
  "esrgan_batch_pause": 0.0,
  "sdxl_ckpt_name": "sd_xl_base_1.0.safetensors",
  "sdxl_positive": "highly detailed, sharp, natural texture, faithful upscale",
  "sdxl_negative": "blur, noise, artifacts, distortion, oversharpen",
  "sdxl_steps": 18,
  "sdxl_cfg": 4.5,
  "sdxl_sampler_name": "dpmpp_2m",
  "sdxl_scheduler": "karras",
  "sdxl_denoise": 0.4,
  "sdxl_latent_upscale_by": 2.0,
  "sdxl_seed": 0,
  "sdxl_use_tiled_vae": true,
  "sdxl_tile_size": 512,
  "debug": false,
  "gui_scale": 1.0,
  "config_path": "bggg_config.json",
  "web_running": false
}
```

---

## 🌟 Notes
- **Version**: 5.5 (stable, with fixed Web UI).
- **License**: MIT
- **Contributing**: Submit pull requests or issues to [your-repo](https://github.com/BGGremlin-Group/upscaler5.5)
- **Support**: Check `logs\bggg.log` for detailed error messages.
- **Recommendation**: Use Windows Terminal for the best TUI experience (UTF-8 and color support).

Enjoy upscaling your videos with a retro twist! 🎞️

---

### Developed by the BG Gremlin Group
***Creating Unique Tools for Unique Individuals***

### ***Presented as is, with no promises nor guarantees of working.***
