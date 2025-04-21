# Video Enhancer README

## Requisitos previos

    - macOS con chip ARM
    - Homebrew instalado

## Instalación de dependencias

    1. Abrir Terminal
    2. Instalar herramientas con Homebrew:
    	brew install python@3.11 ffmpeg opencv
    3. Crear y activar un entorno virtual:
    	/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv ~/venv/video-enhancer
    	source ~/venv/video-enhancer/bin/activate
    4. Actualizar pip y setuptools:
    	pip install --upgrade pip setuptools wheel

## Instalación de librerías Python

    Con el entorno virtual activo, ejecutar:
    	pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
    	pip install basicsr realesrgan

## Verificación rápida

    python - <<PY
    import basicsr, realesrgan
    print("✔ BasicSR y Real‑ESRGAN cargados OK")
    PY

## Uso del script

    1. Colocar el video en una ruta accesible
    2. Ejecutar:
    	./enhance.sh videos/mi_video.mp4
    3. Archivos generados:
    	- denoise_mi_video.mp4
    	- upscaled_mi_video.mp4
    	- final_mi_video.mp4
    	- final_audio_mi_video.mp4

## Instalación de BasicSR y modelo EDVR

    1. Clonar repositorio:
    	git clone https://github.com/xinntao/BasicSR.git ~/BasicSR
    2. Instalar dependencias:
    	pip install -r ~/BasicSR/requirements.txt
    3. Instalar en modo editable:
    	cd ~/BasicSR
    	pip install --use-pep517 -e .
    4. Descargar modelo EDVR:
    	python scripts/download_pretrained_models.py EDVR
    5. Verificación:
    	python3 - <<PY
    	from basicsr.archs.edvr_arch import EDVR
    	print("✅ EDVR OK")
    	PY

## Instalación de DeblurGAN‑v2

    1. Clonar repositorio:
    	git clone https://github.com/VITA-Group/DeblurGANv2.git ~/DeblurGANv2
    2. Crear carpeta de modelos:
    	mkdir -p ~/DeblurGANv2/models
    3. Descargar pesos:
    	curl -L -o ~/DeblurGANv2/models/fpn_inception.h5 "https://drive.google.com/uc?export=download&id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR"
    	curl -L -o ~/DeblurGANv2/models/fpn_mobilenet.h5 "https://drive.google.com/uc?export=download&id=1JhnT4BBeKBBSLqTo6UsJ13HeBXevarrU"
    4. Instalar dependencias (si existe requirements.txt):
    	cd ~/DeblurGANv2
    	pip install -r requirements.txt
    5. Configurar PYTHONPATH:
    	export PYTHONPATH="$HOME/DeblurGANv2:$PYTHONPATH"
    6. Crear archivo setup.py en ~/DeblurGANv2 con:

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

    7. Instalar en modo editable:
    	pip install -e .

## Instalación y modelos Real‑ESRGAN

    1. Clonar repositorio:
    	git clone https://github.com/xinntao/Real-ESRGAN.git ~/Real-ESRGAN
    2. Instalar dependencias:
    	cd ~/Real-ESRGAN
    	pip install basicsr facexlib gfpgan
    	pip install -r requirements.txt
    	python setup.py develop
    3. Descargar pesos:
    	wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ~/Real-ESRGAN/weights
    4. Prueba opcional:
    	python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance

## Notas

    - Ajusta las constantes del script enhance.sh según necesidad
    - El script detecta automáticamente si hay GPU MPS o CPU y ajusta los parámetros
