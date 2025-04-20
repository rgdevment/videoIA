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
