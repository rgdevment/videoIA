#!/usr/bin/env bash
set -euxo pipefail

ts(){ echo "[$(date '+%H:%M:%S')] $*"; }

# ----------------------------------------------------------------
# Uso: ./enhance.sh videos/input.mp4
# ----------------------------------------------------------------
if [ $# -lt 1 ]; then
  echo "Uso: $0 <archivo.mp4>"
  exit 1
fi

INPUT="$1"
[ -f "$INPUT" ] || { echo "ERROR: '$INPUT' no existe."; exit 1; }

# Directorio y nombre base
DIR="$(dirname "$INPUT")"
NAME="$(basename "$INPUT" .mp4)"

# ----------------------------------------------------------------
# 1) Denoise
# ----------------------------------------------------------------
DENOISE="$DIR/denoise_${NAME}.mp4"
if [ -f "$DENOISE" ]; then
  ts "⏭️ Omitiendo denoise (ya existe)"
else
  ts "🎬 Fase 1: Denoise (hqdn3d)"
  ffmpeg -hide_banner -y -stats -i "$INPUT" \
    -vf "hqdn3d=4:3:6:4" \
    -c:v libx264 -crf 18 -preset slow \
    -an "$DENOISE"
  ts "✅ $DENOISE listo"
fi

# ----------------------------------------------------------------
# 2A) Extraer frames
# ----------------------------------------------------------------
ts "🖼️ Fase 2A: Extracción de frames"
FRAMES="$DIR/frames"
UPSCALED="$DIR/upscaled_frames"
rm -rf "$FRAMES" "$UPSCALED"
mkdir -p "$FRAMES" "$UPSCALED"
ffmpeg -hide_banner -y -stats -i "$DENOISE" \
  -vsync 0 "$FRAMES/frame_%08d.png"
ts "✅ frames extraídos en $FRAMES"

# ----------------------------------------------------------------
# 2B) IA Upscale de frames
#     GPU MPS = tile=0 + half
#     CPU    = tile=512 + multiprocessing (máx 4 hilos)
# ----------------------------------------------------------------
# detectamos MPS
HAS_MPS=$(python3 - <<PYCODE
import torch; print(torch.backends.mps.is_available())
PYCODE
)

if [ "$HAS_MPS" = "True" ]; then
  ts "⚡ Dispositivo: GPU MPS (tile=0, half precision)"
  DEVICE="mps"; TILE=0; HALF="True"; NPROC=1
else
  CORES=$(sysctl -n hw.ncpu)
  NPROC=$(( CORES<4 ? CORES : 4 ))
  ts "⚙️ Dispositivo: CPU (tile=512, procesos=$NPROC)"
  DEVICE="cpu"; TILE=512; HALF="False"
fi

ts "🔍 Fase 2B: Upscale IA de frames"
python3 <<PYCODE
import os, cv2, torch
from multiprocessing import Pool
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer

print("🚀 Iniciando upscaler…", flush=True)

device = torch.device("${DEVICE}")
model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2)
upscaler = RealESRGANer(
    scale=2,
    model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    model=model,
    tile=${TILE},
    tile_pad=10,
    pre_pad=10,
    half=${HALF},
    device=device
)

def process_frame(fname):
    inp = os.path.join("${FRAMES}", fname)
    outp = os.path.join("${UPSCALED}", fname)
    # corregir azul: BGR→RGB → enhance → RGB→BGR
    bgr = cv2.imread(inp, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res, _ = upscaler.enhance(rgb, outscale=2)
    bgr_out = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outp, bgr_out)
    print(f"[Upscale] {fname} completado", flush=True)

files = sorted(os.listdir("${FRAMES}"))
if device.type == "mps":
    for f in files:
        process_frame(f)
else:
    with Pool(${NPROC}) as p:
        p.map(process_frame, files)
PYCODE

ts "✅ frames upscalados en $UPSCALED"

# ----------------------------------------------------------------
# 2C) Reensamblar vídeo
# ----------------------------------------------------------------
ts "⚙️ Fase 2C: Reensamblar video"

# extraemos "num/den" y calculamos con bc para tener un float con 2 decimales
R_FRAME_RATE=$(ffprobe -v error -select_streams v:0 \
  -show_entries stream=r_frame_rate \
  -of default=noprint_wrappers=1:nokey=1 "$DENOISE")
NUM=${R_FRAME_RATE%%/*}
DEN=${R_FRAME_RATE##*/}
FPS=$(echo "scale=2; $NUM / $DEN" | bc)

ffmpeg -hide_banner -y -stats -framerate "$FPS" \
  -i "$UPSCALED/frame_%08d.png" \
  -c:v libx264 -crf 16 -preset slow \
  -an "$DIR/upscaled_${NAME}.mp4"
ts "✅ $DIR/upscaled_${NAME}.mp4 listo"

# ----------------------------------------------------------------
# 3) Corrección de color + nitidez
# ----------------------------------------------------------------
ts "🎨 Fase 3: Color & nitidez"
ffmpeg -hide_banner -y -stats -i "$DIR/upscaled_${NAME}.mp4" \
  -vf "eq=contrast=1.1:brightness=0.02:saturation=1.1,unsharp=3:3:0.8" \
  -c:v libx264 -crf 16 -preset slow \
  -c:a copy "$DIR/final_${NAME}.mp4"
ts "✅ $DIR/final_${NAME}.mp4 listo"

# ----------------------------------------------------------------
# 4) Audio (priorizando voces)
# ----------------------------------------------------------------
ts "🔊 Fase 4: Denoise y normalización de audio"
ffmpeg -hide_banner -y -stats -i "$DIR/final_${NAME}.mp4" \
  -af "afftdn,acompressor=threshold=-20dB:ratio=4:attack=5:release=50,dynaudnorm,loudnorm=I=-16:LRA=7:TP=-1.5" \
  -c:v copy "$DIR/final_audio_${NAME}.mp4"
ts "✅ $DIR/final_audio_${NAME}.mp4 listo"

# ----------------------------------------------------------------
ts "🎉 Pipeline completo. Archivos generados:"
echo " - $DENOISE"
echo " - $DIR/upscaled_${NAME}.mp4"
echo " - $DIR/final_${NAME}.mp4"
echo " - $DIR/final_audio_${NAME}.mp4"
