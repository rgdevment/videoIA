#!/usr/bin/env python3
"""
process_frames.py: Aplica en cascada DeblurGAN, EDVR y Real-ESRGAN a una carpeta de frames.
"""
import argparse
import os
import sys
from glob import glob

import cv2
import torch

# Añadir ruta local de DeblurGANv2 si existe
deblurgan_path = os.path.expanduser("~/DeblurGANv2")
if os.path.isdir(deblurgan_path) and deblurgan_path not in sys.path:
    sys.path.insert(0, deblurgan_path)

# Intentar importar DeblurGAN Predictor local
try:
    from predict import Predictor as DeblurModel  # clase Predictor en predict.py de DeblurGANv2
except ImportError:
    DeblurModel = None

# BasicSR imports para EDVR
try:
    from basicsr.archs.edvr_arch import EDVRNet
    from basicsr.utils import img2tensor, tensor2img
except ImportError:
    EDVRNet = None

# Real‑ESRGAN imports
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer


def load_deblurgan_model(device):
    if DeblurModel is None:
        print("WARN: DeblurGAN no está disponible; omitiendo etapa de deblur.")
        return None
    # Ruta a pesos de DeblurGAN descargados manualmente
    weights_path = os.path.expanduser("~/DeblurGANv2/fpn_inception.pth")
    if not os.path.isfile(weights_path):
        print(f"WARN: Peso DeblurGAN no encontrado en {weights_path}; omitiendo etapa.")
        return None
    # Inicializar Predictor sin pasar device
    model = DeblurModel(weights_path)
    # Intentar mover al dispositivo seleccionado (cpu, mps o cuda)
    try:
        model.model.to(device)
    except Exception:
        pass
    return model


def load_edvr_model(device):
    if EDVRNet is None:
        print("WARN: EDVR no está disponible; omitiendo etapa EDVR.")
        return None
    # Configuración EDVR: 5 frames, escala x2
    model = EDVRNet(
        num_in_ch=3, num_out_ch=3,
        num_blocks=20, num_groups=10,
        num_frame=5, center_frame_idx=2,
        scale=2
    )
    weights_path = os.path.expanduser("~/models/EDVR_L_x2.pth")
    if not os.path.isfile(weights_path):
        print(f"WARN: Peso EDVR no encontrado en {weights_path}; omitiendo EDVR.")
        return None
    checkpoint = torch.load(weights_path, map_location=device)
    state = checkpoint.get("params", checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_realesrgan_model(device, tile, half):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2)
    upscaler = RealESRGANer(
        scale=2,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=10,
        half=half,
        device=device
    )
    return upscaler


def process_frame(fname, args, models):
    in_path = os.path.join(args.input_dir, fname)
    out_path = os.path.join(args.output_dir, fname)
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"WARN: No se pudo leer {in_path}")
        return
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pipeline en cascada
    if models.get("deblurgan"):
        frame = models["deblurgan"].predict(frame)
    if models.get("edvr"):
        tensor = img2tensor(frame, bgr2rgb=False, float32=True).unsqueeze(0).to(args.device)
        with torch.no_grad():
            output = models["edvr"](tensor)
        frame = tensor2img(output.squeeze(0), out_type="uint8", min_max=(0, 1))
    if models.get("realesrgan"):
        frame, _ = models["realesrgan"].enhance(frame, outscale=2)

    bgr_out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr_out)


def main():
    parser = argparse.ArgumentParser(
        description="Process frames with DeblurGAN, EDVR and Real-ESRGAN"
    )
    parser.add_argument("--input-dir", required=True, help="Carpeta de frames de entrada")
    parser.add_argument("--output-dir", required=True, help="Carpeta de frames procesados")
    parser.add_argument(
        "--stages", nargs='+', choices=["deblurgan", "edvr", "realesrgan"],
        default=["deblurgan", "edvr", "realesrgan"],
        help="Etapas a aplicar"
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu", help="Dispositivo para inferencia")
    parser.add_argument("--tile", type=int, default=0, help="Tile para Real‑ESRGAN (0=full frame)")
    parser.add_argument("--half", action='store_true', help="Usar fp16 en Real‑ESRGAN")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.device = torch.device(args.device)

    # Cargar modelos según stages
    models = {}  
    models["deblurgan"] = load_deblurgan_model(args.device) if "deblurgan" in args.stages else None
    models["edvr"]      = load_edvr_model(args.device)       if "edvr" in args.stages      else None
    models["realesrgan"]= load_realesrgan_model(args.device, args.tile, args.half) if "realesrgan" in args.stages else None

    files = sorted([os.path.basename(f) for f in glob(os.path.join(args.input_dir, "*.png"))])
    for fname in files:
        print(f"Procesando {fname}...")
        process_frame(fname, args, models)


if __name__ == "__main__":
    main()
