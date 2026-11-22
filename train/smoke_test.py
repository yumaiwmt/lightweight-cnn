"""Train, validate, and reload each architecture on a few real dataset images."""

import argparse
import gc
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
sys.path.insert(0, str(ROOT / "vendor/ultralytics"))

import torch
import yaml
from ultralytics import YOLO, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or a GPU index, such as 0")
    parser.add_argument("--imgsz", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(2)
    settings.update({key: False for key in (
        "wandb", "tensorboard", "mlflow", "clearml", "comet", "dvc", "neptune", "raytune", "hub"
    )})

    config = yaml.safe_load((ROOT / "train/cfg/v3.yaml").read_text(encoding="utf-8"))
    dataset_yaml = ROOT / config["data"]
    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    dataset_root = dataset_yaml.parent.resolve()
    output = ROOT / "runs" / f"smoke_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True)

    # Small manifests leave the full dataset and its source YAML untouched.
    for split, count in (("train", 4), ("val", 2)):
        folder = dataset_root / dataset[split]
        images = sorted(path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[:count]
        if len(images) < count:
            raise FileNotFoundError(f"Need at least {count} images in {folder}")
        manifest = output / f"{split}.txt"
        manifest.write_text("\n".join(str(path.resolve()) for path in images) + "\n", encoding="utf-8")
        dataset[split] = manifest.name
    dataset["path"] = str(output)
    dataset.pop("test", None)
    data = output / "data.yaml"
    data.write_text(yaml.safe_dump(dataset), encoding="utf-8")

    for architecture in sorted((ROOT / "models").glob("*.yaml")):
        print(f"Checking {architecture.name}", flush=True)
        model = YOLO(str(architecture), task="detect")
        model.train(
            data=str(data), epochs=1, imgsz=args.imgsz, batch=2, device=args.device,
            workers=0, amp=False, cache=False, pretrained=False, plots=False,
            project=str(output), name=architecture.stem.replace(" ", "_"),
        )
        checkpoint = str(model.trainer.best)
        del model
        gc.collect()
        reloaded = YOLO(checkpoint, task="detect")
        metrics = reloaded.val(
            data=str(data), imgsz=args.imgsz, batch=2, device=args.device,
            workers=0, plots=False, project=str(Path(checkpoint).parent.parent), name="reload_validation",
        )
        print(f"PASS: {architecture.name}; reloaded checkpoint validated (mAP50={metrics.box.map50:.4f})", flush=True)
        del reloaded, metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"All models passed. Outputs: {output}")


if __name__ == "__main__":
    main()
