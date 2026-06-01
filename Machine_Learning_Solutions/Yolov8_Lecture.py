import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def load_image_paths(folder: Path):
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    return sorted([p for p in folder.rglob('*') if p.suffix.lower() in extensions and p.is_file()])


def main():
    parser = argparse.ArgumentParser(description='Clasificación por lotes de imágenes en FBSI/Test')
    parser.add_argument(
        '--model',
        type=str,
        default=r'C:\Users\santi\Documents\LA\CubeRT_Cubesat\Software\Machine_Learning_Solutions\runs\classify\train-2\weights\best.pt',
        help='Ruta al modelo entrenado de clasificación',
    )
    parser.add_argument(
        '--folder',
        type=str,
        default=r'C:\Users\santi\Documents\LA\FBSI\Test',
        help='Carpeta que contiene las imágenes de prueba',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Mostrar cada imagen con la predicción en una ventana',
    )
    parser.add_argument(
        '--wait',
        type=int,
        default=1500,
        help='Milisegundos que se muestra cada imagen cuando se usa --show',
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    folder = Path(args.folder)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f'No se encontró la carpeta de prueba: {folder}')

    image_paths = load_image_paths(folder)
    if not image_paths:
        raise FileNotFoundError(f'No se encontraron imágenes en: {folder}')

    print(f'Usando modelo: {args.model}')
    print(f'Procesando {len(image_paths)} imágenes en: {folder}\n')

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'No se pudo leer la imagen: {image_path.name}')
            continue

        results = model(image)
        result = results[0]

        label = None
        confidence = None
        if hasattr(result, 'probs') and result.probs is not None:
            probs = result.probs
            if len(probs) > 0:
                if hasattr(probs, 'top1'):
                    class_id = int(probs.top1)
                    confidence = float(probs.top1conf)
                else:
                    probs_array = probs.numpy() if hasattr(probs, 'numpy') else probs
                    class_id = int(probs_array.argmax())
                    confidence = float(probs_array[class_id])

                if hasattr(result, 'names') and result.names is not None:
                    label = result.names[class_id]
                elif hasattr(model, 'names') and model.names is not None:
                    label = model.names[class_id]
                else:
                    label = str(class_id)

        if label is None:
            print(f'{image_path.name}: no se obtuvo predicción')
        else:
            print(f'{image_path.name}: {label} ({confidence:.2f})')

            if args.show:
                text = f'{label}: {confidence:.2f}'
                cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Yolov8 Lecture', image)
                if cv2.waitKey(args.wait) & 0xFF == 27:
                    break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
