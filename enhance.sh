#!/usr/bin/env bash
set -euxo pipefail

# Temporal si hay soporte MPS y CUDA se le pasa al CPU (Muy lento, desactivar es recomendado)
# Ademas no es confiable en los M1 o chip ARM genera pixeles negros.
# export PYTORCH_ENABLE_MPS_FALLBACK=1

# controla si se aplica escalado IA (true/false)
UPSCALE_ENABLED=true

# Función de timestamp
ts(){ echo "[$(date '+%H:%M:%S')] $*"; }

# ----------------------------------------------------------------
# Uso: ./enhance.sh <archivo.mp4>
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

# Rutas de salida
DENOISE_VIDEO="$DIR/denoise_${NAME}.mp4"
ORIG_AUDIO="$DIR/orig_${NAME}.aac"
UPSCALED_VIDEO="$DIR/upscaled_${NAME}.mp4"
FINAL_NO_AUDIO="$DIR/final_${NAME}.mp4"
FINAL_OUTPUT="$DIR/final_audio_${NAME}.mp4"
FRAMES_DIR="$DIR/frames"
UPSCALED_FRAMES_DIR="$DIR/upscaled_frames"

# ----------------------------------------------------------------
# 0) Extraer audio original
# ----------------------------------------------------------------
ts "🔊 Fase 0: Extracción de audio original"
ffmpeg -hide_banner -y -stats -i "$INPUT" -vn -acodec copy "$ORIG_AUDIO"
ts "✅ Audio extraído en $ORIG_AUDIO"

# ----------------------------------------------------------------
# 1) Denoise
# ----------------------------------------------------------------
ts "🎬 Fase 1: Denoise (hqdn3d)"
ffmpeg -hide_banner -y -stats -i "$INPUT" -vf "hqdn3d=4:3:6:4" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart -an "$DENOISE_VIDEO"
ts "✅ $DENOISE_VIDEO listo"

# ----------------------------------------------------------------
# 2) Escalado IA (opcional)
# ----------------------------------------------------------------
if [ "$UPSCALE_ENABLED" != true ]; then
  ts "⏭️ Escalado IA desactivado. Saltando fases 2A-2C"
else
  # 2A) Extraer frames
  ts "🖼️ Fase 2A: Extracción de frames"
  rm -rf "$FRAMES_DIR" "$UPSCALED_FRAMES_DIR"
  mkdir -p "$FRAMES_DIR" "$UPSCALED_FRAMES_DIR"
  ffmpeg -hide_banner -y -stats -i "$DENOISE_VIDEO" -vsync 0 "$FRAMES_DIR/frame_%08d.png"
  ts "✅ frames extraídos en $FRAMES_DIR"

  # 2B) Upscale con Python
  ts "🔍 Fase 2B: Upscale IA de frames"
  HAS_MPS=$(python3 - <<PYCODE
import torch; print(torch.backends.mps.is_available())
PYCODE
  )
  if [ "$HAS_MPS" = "True" ]; then
    DEVICE="mps"; TILE=0; HALF_FLAG="--half"
    ts "⚡ Dispositivo: GPU MPS"
  else
    DEVICE="cpu"; TILE=512; HALF_FLAG=""
    ts "⚙️ Dispositivo: CPU"
  fi

  python3 process_frames.py \
    --input-dir "$FRAMES_DIR" \
    --output-dir "$UPSCALED_FRAMES_DIR" \
    --stages deblurgan edvr realesrgan \
    --device "$DEVICE" --tile "$TILE" $HALF_FLAG
  ts "✅ frames procesados en $UPSCALED_FRAMES_DIR"

  # 2C) Reensamblar vídeo
  ts "⚙️ Fase 2C: Reensamblar video"
  R_FRAME_RATE=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=r_frame_rate \
    -of default=noprint_wrappers=1:nokey=1 "$DENOISE_VIDEO")
  NUM=${R_FRAME_RATE%%/*}; DEN=${R_FRAME_RATE##*/}
  FPS=$(echo "scale=2; $NUM / $DEN" | bc)

  ffmpeg -hide_banner -y -stats -framerate "$FPS" -i "$UPSCALED_FRAMES_DIR/frame_%08d.png" -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -movflags +faststart -an "$UPSCALED_VIDEO"
  ts "✅ $UPSCALED_VIDEO listo"
fi

# ----------------------------------------------------------------
# 3) Color y nitidez
# ----------------------------------------------------------------
if [ "$UPSCALE_ENABLED" = true ]; then
  SOURCE_VIDEO="$UPSCALED_VIDEO"
else
  SOURCE_VIDEO="$DENOISE_VIDEO"
fi

ts "🎨 Fase 3: Color y nitidez"
ffmpeg -hide_banner -y -stats -i "$SOURCE_VIDEO" -vf "eq=contrast=1.1:brightness=0.02:saturation=1.1,unsharp=3:3:0.8" -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -movflags +faststart -an "$FINAL_NO_AUDIO"
ts "✅ $FINAL_NO_AUDIO listo"

# ----------------------------------------------------------------
# 4) Mux y normalización de audio
# ----------------------------------------------------------------
ts "🔊 Fase 4: Normalize y mux audio"
ffmpeg -hide_banner -y -stats -i "$FINAL_NO_AUDIO" -i "$ORIG_AUDIO" -map 0:v -map 1:a -af "afftdn,acompressor=threshold=-20dB:ratio=4:attack=5:release=50,dynaudnorm,loudnorm=I=-16:LRA=7:TP=-1.5" -c:v copy -c:a aac -movflags +faststart "$FINAL_OUTPUT"
ts "✅ $FINAL_OUTPUT listo"

# ----------------------------------------------------------------
# Fin del pipeline
# ----------------------------------------------------------------
ts "🎉 Pipeline completo. Archivos generados:"
echo " - $DENOISE_VIDEO"
echo " - $UPSCALED_VIDEO"
echo " - $FINAL_NO_AUDIO"
echo " - $FINAL_OUTPUT"
