# 📜 Changelog for BGGG Video Upscaler

## [5.5] - 2025-08-19

### 🛠️ Fixed
- **Web UI**: Fixed incomplete `yn` function in `WebThread._routes` for proper boolean parsing in `save_cfg` route, ensuring reliable configuration saving on Windows. 🔧
- Improved `find_ffmpeg` to prioritize Windows paths (e.g., `C:\ffmpeg\bin\ffmpeg.exe`), enhancing compatibility. 🖥️
- Enhanced error logging for ComfyUI HTTP failures, with clear diagnostics in `logs\bggg.log`. 📜

### ✨ Added
- Windows-specific paths in configuration (e.g., `C:\ComfyUI`, `C:\Videos\watch`) for easier setup. 📂
- Enhanced Web UI with grid-based layout and live log display (last 9000 characters), optimized for Windows browsers. 🌐
- Improved CRT GUI log rendering with basic text wrapping, maintaining retro aesthetic on Windows. 📟
- Added `esrgan_batch_pause` to prevent GPU/CPU overload during batch processing. ⚙️

### 🔄 Changed
- Updated default SDXL parameters for better upscaling quality (`steps=18`, `cfg=4.5`, `denoise=0.4`). 🎨
- Optimized `ComfyClient` to prefer direct file copying from `ComfyUI\output` on Windows, reducing reliance on `/view` endpoint. 🚀
- Refined folder watcher to handle polling fallback gracefully when `watchdog` is unavailable, with Windows path support. 📂
- Improved TUI display recommendations, suggesting Windows Terminal for UTF-8 and color support. 🖥️

### 🗑️ Removed
- Removed redundant `ffmpeg` path guesses for non-Windows platforms, streamlining Windows detection. 🧹

### 📝 Notes
- Version 5.5 should be fully compatible with Windows 10/11, Python 3.9+, and ComfyUI’s vanilla nodes.
- Ensure `4x-UltraSharp.pth` and `sd_xl_base_1.0.safetensors` are in `ComfyUI\models\` directories.
- Use Windows Terminal for optimal TUI experience (CMD may not display colors or Unicode correctly).

### 💀 previous builds 
- Redacted
