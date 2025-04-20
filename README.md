# README Video Enhancer

## Requisitos previos

    - macOS en MacBook ARM
    - Homebrew instalado

## Instalación de dependencias

    1. Abrir Terminal
    2. Instalar herramientas con Homebrew:
    	~~~bash
    	brew install python@3.11 ffmpeg opencv
    	~~~
    3. Crear y activar un entorno virtual con Python 3.11:
    	~~~bash
    	/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv ~/venv/video-enhancer
    	source ~/venv/video-enhancer/bin/activate
    	~~~
    4. Actualizar pip y setuptools:
    	~~~bash
    	pip install --upgrade pip setuptools wheel
    	~~~

## Instalación de librerías Python

    En el entorno virtual activo, ejecutar:
    ~~~bash
    pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
    pip install basicsr realesrgan
    ~~~

## Verificación rápida

    ~~~bash
    python - <<PY
    import basicsr, realesrgan
    print("✔ BasicSR y Real‑ESRGAN cargados OK")
    PY
    ~~~

## Uso del script

    1. Colocar el video a procesar en una ruta accesible.
    2. Ejecutar:
    	~~~bash
    	./enhance.sh videos/mi_video.mp4
    	~~~
    3. Los archivos generados se guardarán junto al original:
    	- denoise_mi_video.mp4
    	- upscaled_mi_video.mp4
    	- final_mi_video.mp4
    	- final_audio_mi_video.mp4

## Instalar BasicSR con soporte EDVR

### 1) Clonar el repositorio de BasicSR

Abre Terminal y ejecuta:

```bash
git clone https://github.com/xinntao/BasicSR.git ~/BasicSR
```

### 2) Instalar dependencias

```bash
pip install -r ~/BasicSR/requirements.txt
```

### 3) Instalar BasicSR en modo editable

```
pip install --use-pep517 -e .
```

## Descarga de modelos adicionales

### DeblurGAN‑v2

Clona el repositorio y crea el directorio de modelos:

```bash
git clone https://github.com/VITA-Group/DeblurGANv2.git ~/DeblurGANv2
mkdir -p ~/DeblurGANv2/models
```

Descarga los pesos pre‑entrenados (Inception‑ResNet‑v2 y MobileNet):

```bash
curl -L -o ~/DeblurGANv2/models/fpn_inception.h5 "https://drive.google.com/uc?export=download&id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR"
curl -L -o ~/DeblurGANv2/models/fpn_mobilenet.h5 "https://drive.google.com/uc?export=download&id=1JhnT4BBeKBBSLqTo6UsJ13HeBXevarrU"
```

:contentReference[oaicite:0]{index=0}

### EDVR (BasicSR)

Sitúate en la carpeta del repositorio de BasicSR y ejecuta el script de descarga:

```bash
cd ~/BasicSR
python scripts/download_pretrained_models.py EDVR
```

Esto descargará automáticamente todos los modelos EDVR (Vimeo90K, REDS, deblur, etc.) en  
`experiments/pretrained_models/` :contentReference[oaicite:1]{index=1}

Si prefieres hacerlo manualmente, por ejemplo para el track REDS_SR_L:

```bash
mkdir -p ~/BasicSR/experiments/pretrained_models
curl -L -o ~/BasicSR/experiments/pretrained_models/EDVR_REDS_SR_L.zip \
  "https://drive.google.com/uc?export=download&id=1h6E0QVZyJ5SBkcnYaT1puxYYPVbPsTLt"
unzip ~/BasicSR/experiments/pretrained_models/EDVR_REDS_SR_L.zip \
  -d ~/BasicSR/experiments/pretrained_models/
```

:contentReference[oaicite:2]{index=2}

---

Con esto tendrás todos los pesos necesarios para las etapas de DeblurGAN, EDVR y RealESRGAN en tu pipeline de vídeo.
::contentReference[oaicite:3]{index=3}

### 5) Verificar la importación

En tu entorno virtual, prueba:

```bash
python3 - <<PY
from basicsr.archs.edvr_arch import EDVR
print("✅ EDVR OK")
PY
```

---

## Instalación local de DeblurGAN‑v2

### 1) Clonar el repositorio

- Abrir Terminal y ejecutar:
  ```bash
  git clone https://github.com/VITA-Group/DeblurGANv2.git ~/DeblurGANv2
  ```

### 2) Instalar dependencias del repositorio

- Entrar en la carpeta del proyecto:
  ```bash
  cd ~/DeblurGANv2
  ```
- Instalar librerías desde requirements.txt (si existe):
  ```bash
  pip install -r requirements.txt
  ```

### 3) Configurar PYTHONPATH

- Añadir la ruta local para que Python reconozca el paquete:
  ```bash
  export PYTHONPATH="$HOME/DeblurGANv2:$PYTHONPATH"
  ```

## Crear paquete instalable con setup.py

En la raíz de ~/DeblurGANv2 crear un archivo setup.py con:

```python
from setuptools import setup, find_packages

setup(
    name="deblurgan",
    version="0.1.0",
    description="DeblurGAN‑v2 para deblurring de frames de vídeo",
    author="Orest Kupyn et al.",
    packages=find_packages(),
    install_requires=[
		"torch>=1.0,<2.0",
		"opencv-python>=4.5",
		"numpy>=1.19,<2.0"
    ],
)
```

- find_packages() detecta los módulos automáticamente  
- install_requires lista las dependencias mínimas

## Instalación en modo editable

Con el entorno virtual activo y en la carpeta ~/DeblurGANv2:

```bash
pip install -e .
```

Esto instala el paquete local en modo editable, de forma que cualquier cambio en el código se refleje al instante.

## Notas

    - Ajustar constantes dentro de enhance.sh para opcionalidad de escalado
    - El script detecta automáticamente GPU MPS o CPU
