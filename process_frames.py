#!/usr/bin/env python3
"""
process_frames.py: Aplica en cascada DeblurGAN, EDVR y Real‑ESRGAN a una carpeta de frames.
"""
import argparse
import os
import sys
from glob import glob

import cv2
import torch

# Rutas de pesos (manuales en ./models)
FPIN_PATH = os.path.abspath(os.path.join(os.getcwd(), "models/fpn_inception.h5"))
FPMOB_PATH = os.path.abspath(os.path.join(os.getcwd(), "models/fpn_mobilenet.h5"))
if os.path.isfile(FPIN_PATH):
    DEBLURGAN_PATH = FPIN_PATH
    print("INFO: Usando DeblurGANv2 Inception")
elif os.path.isfile(FPMOB_PATH):
    DEBLURGAN_PATH = FPMOB_PATH
    print("INFO: Usando DeblurGANv2 MobileNet")
else:
    DEBLURGAN_PATH = FPIN_PATH  # por defecto

# EDVR: apuntar al modelo descargado
EDVR_PATH = os.path.abspath(os.path.join(
    os.getcwd(),
    "models/EDVR_M_x4_SR_REDS_official-32075921.pth"
))

# Añadir ruta local de DeblurGANv2
proj_root = os.path.abspath(os.getcwd())
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Importar DeblurGAN Predictor
try:
    from predict import Predictor as DeblurModel
except ImportError:
    DeblurModel = None

# EDVR
tensor_imported = True
try:
    from basicsr.archs.edvr_arch import EDVR as EDVRModel
    from basicsr.utils import img2tensor, tensor2img
except ImportError:
    EDVRModel = None
    tensor_imported = False

# Real-ESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer


def check_weight(path, name):
    if not os.path.isfile(path):
        print(f"WARN: Peso para {name} no encontrado en {path}.")
        print("Por favor, descarga manualmente y coloca en esa ruta.")
        return False
    return True


def load_deblurgan_model(device):
    """
    Carga el modelo DeblurGANv2 solo si se dispone de GPU CUDA.
    En otros dispositivos (MPS/CPU) se omite.
    """
    if device.type != 'cuda':
        print("WARN: DeblurGANv2 requiere GPU CUDA; omitiendo etapa de deblur.")
        return None
    if DeblurModel is None:
        print("WARN: DeblurGANv2 no instalado; omitiendo etapa de deblur.")
        return None
    if not check_weight(DEBLURGAN_PATH, "DeblurGANv2"):
        return None
    # Cargar modelo en CUDA
    cwd = os.getcwd()
    try:
        os.chdir(os.path.expanduser("~/DeblurGANv2"))
        model = DeblurModel(DEBLURGAN_PATH)
    finally:
        os.chdir(cwd)
    model.model.cuda()
    model.model.eval()
    return model


def load_edvr_model(device):
    """
    Carga EDVR. En MPS permite fallback a CPU si PYTORCH_ENABLE_MPS_FALLBACK=1.
    """
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == '1'
    if device.type == 'mps' and not fallback:
        print("WARN: EDVR no soportado en MPS; omitiendo etapa EDVR.")
        return None
    # determinar dispositivo para EDVR
    edvr_dev = torch.device('cpu') if (device.type == 'mps' and fallback) else device
    if EDVRModel is None or not tensor_imported:
        print("WARN: EDVR no disponible; omitiendo etapa EDVR.")
        return None
    if not check_weight(EDVR_PATH, "EDVR"):
        return None
    mdl = EDVRModel(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=10,
        center_frame_idx=2, hr_in=False,
        with_predeblur=False, with_tsa=True
    )
    ckpt = torch.load(EDVR_PATH, map_location=edvr_dev)
    state = ckpt.get("params", ckpt)
    mdl.load_state_dict(state)
    mdl.to(edvr_dev).eval()
    # guardar dispositivo en el modelo para usar en process_frame
    mdl.device = edvr_dev
    return mdl


def load_realesrgan_model(device, tile, half):
    REAL_PATH = os.path.abspath(os.path.join(os.getcwd(), "models/RealESRGAN_x4plus.pth"))
    print("WARN: RealESRGAN local no encontrado; descarga automática.")
    model_path = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/"
        "RealESRGAN_x2plus.pth"
    )
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2)
    return RealESRGANer(
        scale=2, model_path=model_path,
        model=model, tile=tile,
        tile_pad=10, pre_pad=10,
        half=half, device=device
    )


def process_frame(fname, args, models):
    in_path = os.path.join(args.input_dir, fname)
    out_path = os.path.join(args.output_dir, fname)
    img = cv2.imread(in_path)
    if img is None:
        print(f"WARN: no pude leer {in_path}")
        return
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # DeblurGANv2
    if models.get("deblurgan"):
        deblurer = models["deblurgan"]
        try:
            frame = deblurer(frame)
        except TypeError:
            frame = deblurer(frame, None)

        # EDVR: requiere tensor 5D (B, T, C, H, W)
    if models.get("edvr"):
        edvr_model = models["edvr"]
        raw = img2tensor(frame, bgr2rgb=False, float32=True).unsqueeze(0)
        t = raw.unsqueeze(1).repeat(1, 5, 1, 1, 1)
        # enviar tensor al dispositivo del modelo EDVR (CPU o MPS)
        t = t.to(getattr(edvr_model, 'device', args.device))
        with torch.no_grad():
            out = edvr_model(t)
        frame = tensor2img(out.squeeze(0), out_type="uint8", min_max=(0,1))

    # RealESRGAN
    if models.get("realesrgan"):
        frame, _ = models["realesrgan"].enhance(frame, outscale=2)

    # Guardar
    cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def main():
    p = argparse.ArgumentParser(description="DeblurGAN+EDVR+RealESRGAN")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stages", nargs='+', default=["deblurgan","edvr","realesrgan"])
    p.add_argument("--device", choices=["cpu","mps","cuda"], default="cpu")
    p.add_argument("--tile", type=int, default=0)
    p.add_argument("--half", action='store_true')
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.device = torch.device(args.device)
    md = {}
    if "deblurgan" in args.stages:
        md["deblurgan"] = load_deblurgan_model(args.device)
    if "edvr" in args.stages:
        md["edvr"] = load_edvr_model(args.device)
    if "realesrgan" in args.stages:
        md["realesrgan"] = load_realesrgan_model(args.device, args.tile, args.half)
    for f in sorted(glob(os.path.join(args.input_dir, "*.png"))):
        print(f"Procesando {os.path.basename(f)}...")
        process_frame(os.path.basename(f), args, md)

if __name__ == "__main__":
    main()
