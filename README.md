# BullE-AI

**Resumen**
- Proyecto para entrenamiento y despliegue local de modelos YOLOv8 (clasificación) sobre el dataset FBSI.

**Requisitos**
- Python 3.8 o superior
- Espacio en disco suficiente para dataset y pesos

**Instalación rápida**
1. Crear y activar un entorno virtual:

	 - Windows (PowerShell):

		 ```powershell
		 python -m venv .venv
		 .\.venv\Scripts\Activate.ps1
		 python -m pip install --upgrade pip
		 ```

2. Instalar dependencias mínimas:

	 ```powershell
	 pip install ultralytics opencv-python pandas lxml pillow
	 ```

**Estructura relevante del repo**
- `FBSI/dataset/` : carpetas por clase con imágenes (ej: `Score 0`, `Score 1`, ...)
- `FBSI/split_dataset.py` : script para crear `train/` y `val/` a partir del dataset por clases
- `Machine_Learning_Solutions/Yolov8.py` : ejemplo de llamada a `ultralytics.YOLO.train()` (puedes editar la ruta de datos)
- `Machine_Learning_Solutions/Yolov8_Prediction.py` : ejemplo para inferencia en cámara (webcam) usando un modelo entrenado

**Preparar los datos**
1. Organiza tus imágenes en carpetas por clase dentro de `FBSI/dataset/`. Ejemplo:

	 ```text
	 FBSI/dataset/
		 Score 0/
			 img001.jpg
		 Score 1/
			 img002.jpg
		 ...
	 ```

2. Ejecuta el script de split para generar `train/` y `val/` (por defecto copia archivos):

	 ```powershell
	 python FBSI/split_dataset.py --source FBSI/dataset --output FBSI --ratio 0.8 --seed 42
	 ```

Esto creará `FBSI/train/` y `FBSI/val/` con subcarpetas por clase.

**Configurar `data.yaml` (opcional pero recomendado)**
- Para mayor control crea un archivo `FBSI/data.yaml` con el siguiente formato (ajusta `names` y `nc`):

```yaml
train: FBSI/train
val:   FBSI/val
nc: 5
names: ['Score 0', 'Score 1', 'Score 1-2', 'Score 2', 'Score 3']
```

Si no quieres crear `data.yaml`, la API de Ultralytics también acepta la ruta del directorio que contiene `train/` y `val/`.

**Entrenamiento**
Puedes entrenar de dos formas: usando la CLI `yolo` de Ultralytics o el script Python.

- Usando la CLI (recomendado para rapidez):

	```powershell
	yolo task=classify mode=train model=yolov8n-cls.pt data=FBSI/data.yaml epochs=50 imgsz=224 batch=32 device=0
	```

	- `device=0` usa GPU CUDA 0 (si está disponible). Para CPU usa `device=cpu`.

- Usando el script de ejemplo (edita las rutas dentro del script si es necesario):

	```powershell
	python Machine_Learning_Solutions/Yolov8.py
	```

Tras el entrenamiento, los pesos estarán en `runs/classify/` (p. ej. `runs/classify/train/weights/best.pt`).

**Despliegue local (Webcam / cámara)**
Se provee `Machine_Learning_Solutions/Yolov8_Prediction.py` para inferencia en tiempo real desde webcam.

Ejemplo de uso:

```powershell
python Machine_Learning_Solutions/Yolov8_Prediction.py --model runs/classify/train/weights/best.pt --source 0
```

- `--model` : ruta al `.pt` generado durante el entrenamiento
- `--source` : `0` para webcam por defecto, o ruta a archivo de video

Presiona `ESC` para salir de la ventana de visualización.

**Inferencia sobre un conjunto de imágenes (lote)**
Opción A — usar la CLI de Ultralytics:

```powershell
yolo task=classify mode=predict model=runs/classify/train/weights/best.pt source=path/to/images save=True
```

Esto procesará todas las imágenes en la carpeta `path/to/images` y guardará resultados en `runs/predict`.

Opción B — script rápido con OpenCV (ejemplo):

```python
from ultralytics import YOLO
import cv2
import pathlib

model = YOLO('runs/classify/train/weights/best.pt')
src = pathlib.Path('path/to/images')
for img_path in sorted(src.glob('*.*')):
		img = cv2.imread(str(img_path))
		res = model(img)
		# procesar res[0].probs o guardar imagen con anotaciones

```

**Consejos prácticos**
- Ajusta `imgsz` y `batch` según memoria GPU.
- Si tienes muchas clases o pocas imágenes, experimenta con aumentos de datos o preentrenamiento.
- Mantén copias de seguridad antes de usar `--move` en `split_dataset.py`.