#!/usr/bin/env bash
set -euxo pipefail

ts(){ echo "[$(date '+%H:%M:%S')] $*"; }

# Uso
if [ $# -lt 1 ]; then
  echo "Uso: $0 <archivo.mp4>"
  exit 1
fi
INPUT="$1"
[ -f "$INPUT" ] || { echo "ERROR: '$INPUT' no existe."; exit 1; }
BASE="${INPUT%.mp4}"

# 1) Denoise
if [ -f "denoise_${BASE}.mp4" ]; then
  ts "⏭️ Omitiendo denoise_${BASE}.mp4 (ya existe)"
else
  ts "🎬 Fase 1: Denoise (hqdn3d)"
  ffmpeg -hide_banner -y -stats -i "$INPUT" \
    -vf "hqdn3d=4:3:6:4" \
    -c:v libx264 -crf 18 -preset slow \
    -an denoise_${BASE}.mp4
  ts "✅ denoise_${BASE}.mp4 listo"
fi

# 2A) Extraer frames
ts "🖼️ Fase 2A: Extracción de frames"
rm -rf frames upscaled_frames && mkdir -p frames upscaled_frames
ffmpeg -hide_banner -y -stats -i denoise_${BASE}.mp4 \
  -vsync 0 frames/frame_%08d.png
ts "✅ frames/ listos"

# 2B) IA Upscale de frames (GPU MPS si hay, si no CPU)
ts "🔍 Fase 2B: IA Upscale de frames (GPU MPS si hay, CPU si no)"
python3 << 'PYCODE'
import os, cv2, torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer

print("🚀 Iniciando upscaler…", flush=True)
# 1) Elegimos dispositivo
device = torch.device('mps') if (torch.backends.mps.is_available()) else torch.device('cpu')
use_gpu = (device.type != 'cpu')

# 2) Arquitectura
model = RRDBNet(3, 3, scale=2)

# 3) Init upscaler: tile=0 en GPU (full frame), half precision sólo en GPU
upscaler = RealESRGANer(
    scale=2,
    model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    model=model,
    tile=0 if use_gpu else 512,
    tile_pad=10,
    pre_pad=10,
    half=use_gpu,
    device=device
)

# 4) Procesamiento secuencial
for fname in sorted(os.listdir("frames")):
    inp = os.path.join("frames", fname)
    outp = os.path.join("upscaled_frames", fname)
    img = cv2.imread(inp, cv2.IMREAD_COLOR)
    res, _ = upscaler.enhance(img, outscale=2)
    cv2.imwrite(outp, res)
    print(f"[Upscale] {fname} completado", flush=True)
PYCODE
ts "✅ upscaled_frames/ listos"

# 2C) Reensamblar vídeo
ts "⚙️ Fase 2C: Reensamblar video"
FPS=$(ffprobe -v error -select_streams v:0 \
      -show_entries stream=r_frame_rate \
      -of default=noprint_wrappers=1:nokey=1 denoise_${BASE}.mp4 | \
      awk -F'/' '{printf "%.2f", $1/$2}')
ffmpeg -hide_banner -y -stats -framerate "$FPS" \
  -i upscaled_frames/frame_%08d.png \
  -c:v libx264 -crf 16 -preset slow \
  -an upscaled_${BASE}.mp4
ts "✅ upscaled_${BASE}.mp4 listo"

# 3) Color + nitidez
ts "🎨 Fase 3: Color & nitidez"
ffmpeg -hide_banner -y -stats -i upscaled_${BASE}.mp4 \
  -vf "eq=contrast=1.1:brightness=0.02:saturation=1.1,unsharp=3:3:0.8" \
  -c:v libx264 -crf 16 -preset slow \
  -c:a copy final_${BASE}.mp4
ts "✅ final_${BASE}.mp4 listo"

# 4) Audio (priorizando voces)
ts "🔊 Fase 4: Denoise y normalización de audio"
ffmpeg -hide_banner -y -stats -i final_${BASE}.mp4 \
  -af "afftdn,acompressor=threshold=-20dB:ratio=4:attack=5:release=50,dynaudnorm,loudnorm=I=-16:LRA=7:TP=-1.5" \
  -c:v copy final_audio_${BASE}.mp4
ts "✅ final_audio_${BASE}.mp4 listo"

ts "🎉 Pipeline completo. Archivos generados:"
echo " - denoise_${BASE}.mp4"
echo " - upscaled_${BASE}.mp4"
echo " - final_${BASE}.mp4"
echo " - final_audio_${BASE}.mp4"
