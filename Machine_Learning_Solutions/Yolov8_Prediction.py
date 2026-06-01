# Importar librerías

import argparse
import cv2
import logging
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Real-time classification con YOLOv8")
    parser.add_argument(
        "--model",
        type=str,
        default=r"C:...\Machine_Learning_Solutions\runs\classify\train-2\weights\best.pt",
        help="Ruta al modelo entrenado de clasificación",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Fuente de entrada: 0 para cámara, o ruta de archivo de video.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Ancho de la ventana de visualización.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Alto de la ventana de visualización.",
    )
    args = parser.parse_args()

    logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
    model = YOLO(args.model)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {args.source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        result = results[0]

        label = None
        confidence = None

        if hasattr(result, "probs") and result.probs is not None:
            probs = result.probs
            if len(probs) > 0:
                class_id = int(probs.argmax())
                confidence = float(probs[class_id])
                label = result.names[class_id] if hasattr(result, "names") else str(class_id)

        if label is not None:
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("YOLOv8 Classification", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
