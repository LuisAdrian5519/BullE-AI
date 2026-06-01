import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def split_dataset(source_dir: Path, output_dir: Path, ratio: float, seed: int, move: bool = False):
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    random_generator = random.Random(seed)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    total_train = 0
    total_val = 0

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        images = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            print(f"Advertencia: no se encontraron imágenes en la clase {class_name}")
            continue

        random_generator.shuffle(images)
        split_index = int(len(images) * ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        class_train_dir = train_dir / class_name
        class_val_dir = val_dir / class_name
        class_train_dir.mkdir(parents=True, exist_ok=True)
        class_val_dir.mkdir(parents=True, exist_ok=True)

        for src_path in train_images:
            dst_path = class_train_dir / src_path.name
            if move:
                shutil.move(str(src_path), str(dst_path))
            else:
                shutil.copy2(str(src_path), str(dst_path))
        for src_path in val_images:
            dst_path = class_val_dir / src_path.name
            if move:
                shutil.move(str(src_path), str(dst_path))
            else:
                shutil.copy2(str(src_path), str(dst_path))

        summary.append((class_name, len(train_images), len(val_images)))
        total_train += len(train_images)
        total_val += len(val_images)

    print("Split completado")
    print(f"Directorio origen: {source_dir}")
    print(f"Directorio destino: {output_dir}")
    print(f"Ratio de training: {ratio:.2f}")
    print(f"Semilla: {seed}")
    print("\nResumen por clase:")
    for class_name, train_count, val_count in summary:
        print(f"  {class_name}: {train_count} train, {val_count} val")
    print(f"\nTotal: {total_train} train, {total_val} val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Divide un dataset de clasificación en carpetas train/val con ratio 80/20."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset"),
        help="Directorio origen con carpetas de clase (por ejemplo FBSI/dataset).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="Directorio de salida donde se crearán train/ y val/.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Porcentaje de imágenes para train (valor entre 0 y 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para la selección aleatoria.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Mover archivos en lugar de copiarlos.",
    )

    args = parser.parse_args()
    split_dataset(args.source, args.output, args.ratio, args.seed, args.move)
