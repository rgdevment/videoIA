#!/usr/bin/env python3
"""
process_frames.py: Aplica en cascada DeblurGAN, EDVR y Real-ESRGAN a una carpeta de frames.
Optimizado para videos antiguos con movimiento, baja resolución y múltiples tipos de rostros.
"""
import argparse
import os
import sys
from glob import glob

import cv2
import torch

# Rutas de pesos (manuales en ./models)
FPIN_PATH = os.path.abspath("models/fpn_inception.h5")
FPMOB_PATH = os.path.abspath("models/fpn_mobilenet.h5")
EDVR_PATH = os.path.abspath("models/EDVR_M_x4_SR_REDS_official-32075921.pth")
REAL_PATH = os.path.abspath("models/RealESRGAN_x4plus.pth")  # modelo más suave
GFPGAN_PATH = os.path.abspath("models/GFPGANv1.4.pth")

# Detectar peso disponible de DeblurGAN
if os.path.isfile(FPIN_PATH):
    DEBLURGAN_PATH = FPIN_PATH
    print("INFO: Usando DeblurGANv2 Inception")
elif os.path.isfile(FPMOB_PATH):
    DEBLURGAN_PATH = FPMOB_PATH
    print("INFO: Usando DeblurGANv2 MobileNet")
else:
    DEBLURGAN_PATH = None
    print("WARN: No se encontraron pesos de DeblurGAN")

# Agregar ruta local para DeblurGAN
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

# Real-ESRGAN y GFPGAN
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

# Constante para aplicar o no mejora facial
ENABLE_FACE_ENHANCE_BY_DEFAULT = False

def check_weight(path, name):
    if not os.path.isfile(path):
        print(f"WARN: Peso para {name} no encontrado en {path}.")
        return False
    return True

def load_deblurgan_model(device):
    if device.type != 'cuda':
        print("WARN: DeblurGANv2 requiere GPU CUDA; omitiendo etapa de deblur.")
        return None
    if DeblurModel is None or not check_weight(DEBLURGAN_PATH, "DeblurGANv2"):
        return None
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
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == '1'
    if device.type == 'mps' and not fallback:
        print("WARN: EDVR no soportado en MPS; omitiendo etapa EDVR.")
        return None
    if device.type != 'cuda':
        print("WARN: EDVR no soportado correctamente sin CUDA; omitiendo etapa EDVR.")
        return None
    edvr_dev = torch.device('cpu') if (device.type == 'mps' and fallback) else device
    if EDVRModel is None or not tensor_imported or not check_weight(EDVR_PATH, "EDVR"):
        return None
    mdl = EDVRModel(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=10,
        center_frame_idx=2, hr_in=False,
        with_predeblur=False, with_tsa=True
    )
    ckpt = torch.load(EDVR_PATH, map_location=edvr_dev)
    mdl.load_state_dict(ckpt.get("params", ckpt))
    mdl.to(edvr_dev).eval()
    mdl.device = edvr_dev
    return mdl

def load_realesrgan_model(device, tile, half):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4)
    return RealESRGANer(
        scale=4, model_path=REAL_PATH,
        model=model, tile=tile,
        tile_pad=10, pre_pad=10,
        half=half, device=device
    )

def load_gfpgan_model(device):
    if not GFPGAN_AVAILABLE:
        print("WARN: GFPGAN no disponible; omitiendo mejora de rostros")
        return None
    if not os.path.isfile(GFPGAN_PATH):
        print(f"WARN: Modelo GFPGAN no encontrado en {GFPGAN_PATH}")
        return None
    try:
        return GFPGANer(
            model_path=GFPGAN_PATH,
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=str(device)
        )
    except Exception as e:
        print(f"WARN: Fallo cargando GFPGAN: {e}")
        return None

def process_frame(fname, args, models):
    in_path = os.path.join(args.input_dir, fname)
    out_path = os.path.join(args.output_dir, fname)
    img = cv2.imread(in_path)
    if img is None:
        print(f"WARN: no pude leer {in_path}")
        return
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if models.get("deblurgan"):
        try:
            frame = models["deblurgan"](frame)
        except TypeError:
            frame = models["deblurgan"](frame, None)

    if models.get("edvr"):
        edvr_model = models["edvr"]
        raw = img2tensor(frame, bgr2rgb=False, float32=True).unsqueeze(0)
        t = raw.unsqueeze(1).repeat(1, 5, 1, 1, 1).to(edvr_model.device)
        with torch.no_grad():
            out = edvr_model(t)
        img_array = tensor2img(out.squeeze(0), out_type="uint8", min_max=(0,1))
        if (img_array == 0).all():
            print(f"WARN: EDVR produjo frame negro en {fname}, omitiendo salida")
        else:
            frame = img_array

    if models.get("realesrgan"):
        frame, _ = models["realesrgan"].enhance(frame, outscale=4)

    if (args.face_enhance or ENABLE_FACE_ENHANCE_BY_DEFAULT) and models.get("gfpgan"):
        try:
            if frame.std() > 10:
                frame, _ = models["gfpgan"].enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                print(f"Frame {fname} descartado para mejora facial por baja información")
        except Exception as e:
            print(f"WARN: GFPGAN fallo al procesar {fname}: {e}")

    cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def main():
    p = argparse.ArgumentParser(description="DeblurGAN+EDVR+RealESRGAN+GFPGAN")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stages", nargs='+', default=["deblurgan","edvr","realesrgan"])
    p.add_argument("--device", choices=["cpu","mps","cuda"], default="cpu")
    p.add_argument("--tile", type=int, default=0)
    p.add_argument("--half", action='store_true')
    p.add_argument("--face-enhance", action='store_true', help="Usar GFPGAN para restaurar rostros")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.device = torch.device(args.device)

    models = {}
    if "deblurgan" in args.stages:
        models["deblurgan"] = load_deblurgan_model(args.device)
    if "edvr" in args.stages:
        models["edvr"] = load_edvr_model(args.device)
    if "realesrgan" in args.stages:
        models["realesrgan"] = load_realesrgan_model(args.device, args.tile, args.half)
    if args.face_enhance or ENABLE_FACE_ENHANCE_BY_DEFAULT:
        models["gfpgan"] = load_gfpgan_model(args.device)

    for f in sorted(glob(os.path.join(args.input_dir, "*.png"))):
        print(f"Procesando {os.path.basename(f)}...")
        process_frame(os.path.basename(f), args, models)

if __name__ == "__main__":
    main()
