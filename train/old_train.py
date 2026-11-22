#!/usr/bin/env python3
"""

██╗   ██╗████████╗███████╗██████╗         ██████╗ ██╗   ██╗
██║   ██║╚══██╔══╝██╔════╝██╔══██╗        ██╔══██╗██║   ██║
██║   ██║   ██║   █████╗  ██████╔╝        ██║  ██║██║   ██║
██║   ██║   ██║   ██╔══╝  ██╔══██╗        ██║  ██║╚██╗ ██╔╝
╚██████╔╝   ██║   ██║     ██║  ██║███████╗██████╔╝ ╚████╔╝
 ╚═════╝    ╚═╝   ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝   ╚═══╝

Training script for YOLOv11n model with Roboflow and Ultralytics YOLO and WandB integration.
Author: Daniel Asadi

Usage:
  (default)      Train the model
  --no-train     Skip training
  --validate     Run validation
  --export       Export to TensorRT
  --benchmark    Benchmark the model
  --predict      Run inference on an image
  --image PATH   Path to image for prediction (default: utfr_01148.jpg)
  --config PATH  Path to configuration file (default: ../cfg/v3.yaml)
"""

import argparse
from roboflow import Roboflow
from ultralytics import YOLO, settings
from ultralytics.utils.benchmarks import benchmark
import os
import shutil
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# Ultralytics parameters
MODEL_PATH = "../models/yolo11n.pt"  # Doesn't work
PARAMETER_FILE = "../cfg/v3.yaml"  # Default parameter file to use

# Roboflow parameters
ROBOFLOW_API_KEY = "T"
DS_VERSION = 3
WORKSPACE = "utfrdv"
ROBOFLOW_PROJECT = "cone-detection-v7sp1"
DATASET_PATH = "../data"

# WandB parameters
ENTITY = "utfr_dv"
WANDB_PROJECT = "yolo11n-training"
LOG_PATH = "yolo11n-training"


class YoloTrainer:
    def __init__(self, model_path: str, parameter_file: str):
        self.parameter_file = parameter_file
        self.model_path = model_path
        self.engine_model_path = None
        self.pt_model_path = None
        settings.update({"wandb": True})
        self.model = YOLO(model=model_path, task="detect")

        self.init_dataset()

        with open(self.parameter_file, "r") as f:
            self.config = yaml.safe_load(f)

    def init_dataset(self):
        if os.path.exists(DATASET_PATH):
            print("Dataset already exists locally. Skipping download from Roboflow.")
            return

        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(WORKSPACE).project(ROBOFLOW_PROJECT)
        version = project.version(DS_VERSION)
        version.download("yolov11", location=DATASET_PATH)

    def train(self):
        """
        Train the YOLOv11 model with the specified parameters.
        """
        print("Starting training.")
        start_time = datetime.now()

        metrics = self.model.train(cfg=self.parameter_file)

        elapsed = datetime.now() - start_time
        # print(metrics)
        print(f"Training completed in {elapsed}.")

    def tune(self): # WIP
        """
        Hyperparameter tuning using Ray Tune.
        """
        print("Starting Ray Tune hyperparameter optimization.")
        start_time = datetime.now()
        # ray tune needs absolute path to data file
        data_path = str(Path(__file__).resolve().parent.parent / "data" / "data.yaml")

        result = self.model.tune(
            use_ray=True,
            grace_period=10,  # default
            gpu_per_trial=1,  # default
            iterations=10,  # default
            data=data_path,
            **self.config,  # Unpack the config dictionary
        )

        elapsed = datetime.now() - start_time
        print(f"Hyperparameter tuning completed in {elapsed}.")
        print("Results:", result)

        best_result = result.get_best_result(metric="metrics/mAP50", mode="max")
        print("Best result:", best_result.config)

    def validate(self):
        """
        Validate the YOLOv11 model on the validation dataset.
        """
        print("Starting validation.")
        start_time = datetime.now()

        metrics = self.model.val(cfg=self.parameter_file)

        elapsed = datetime.now() - start_time
        # print(metrics)
        print(f"Validation completed in {elapsed}.")

    def export(self):
        print("Starting export.")
        start_time = datetime.now()

        self.model.export(
            cfg=self.parameter_file,
            data=self.config.get("data", "../data/data.yaml"),  # must pass explicitly
        )

        latest_train = get_latest_train_folder_with_engine()
        src_path = os.path.join(latest_train, "weights", "best.engine")
        self.pt_model_path = os.path.join(latest_train, "weights", "best.pt")

        if not os.path.exists(src_path):
            raise FileNotFoundError(f"{src_path} not found.")

        os.makedirs("../models", exist_ok=True)
        date_tag = datetime.now().strftime("%Y-%m-%d")
        self.engine_model_path = f"../models/{date_tag}-yolo11.engine"

        shutil.move(src_path, self.engine_model_path)
        print(f"Moved TensorRT engine to: {self.engine_model_path}")

        elapsed = datetime.now() - start_time
        print(f"Export completed in {elapsed}.")

    def example_predict(self, image_path="test.jpg"):
        # Run inference
        self.model = YOLO(model=self.engine_model_path, task="detect")
        results = self.model.predict(
            source=image_path,
            imgsz=self.config.get("imgsz", "832"),
            conf=0.4,
            half=True,
        )

        # Get result image with bounding boxes drawn
        result_image = results[0].plot()

        # Visualize using matplotlib
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title("YOLOv11n TensorRT Inference")
        plt.axis("off")
        plt.show()

    def benchmark(self):
        if self.pt_model_path is None or not os.path.exists(self.pt_model_path):
            raise ValueError(
                "Exported .pt model not found. Run export() before benchmarking."
            )

        print("Starting benchmark.")
        benchmark(
            model=self.pt_model_path,
            data=self.config.get("data", "../data/data.yaml"),
            imgsz=self.config.get("imgsz", 832),
            half=self.config.get("half", True),
            int8=self.config.get("int8", True),
            format=self.config.get("format", "engine"),
        )
        print("Benchmark completed.")


def get_latest_train_folder_with_engine(base_dir="yolo11n-training"):
    """Returns the latest folder inside base_dir that contains 'weights/best.engine'."""
    candidates = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    candidates_with_engine = [
        folder
        for folder in candidates
        if os.path.exists(os.path.join(folder, "weights", "best.engine"))
    ]

    if not candidates_with_engine:
        raise FileNotFoundError("No train folders with best.engine found.")

    latest = max(candidates_with_engine, key=os.path.getmtime)
    return latest


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv11n Trainer with optional steps."
    )
    parser.add_argument("--no-train", action="store_true", help="Skip training")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument(
        "--export", action="store_true", help="Export model to TensorRT"
    )
    parser.add_argument("--predict", action="store_true", help="Run example prediction")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument(
        "--tune", action="store_true", help="Run Ray Tune hyperparameter tuning"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="utfr_01148.jpg",
        help="Path to image for prediction",
    )
    parser.add_argument(
        "--config", type=str, default=PARAMETER_FILE, help="Path to config file"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        print(f"Using configuration file: {args.config}")
        PARAMETER_FILE = (
            os.path.join("../cfg", args.config)
            if not args.config.startswith("../")
            else args.config
        )
    else:
        print(f"Using default configuration file: {PARAMETER_FILE}")

    trainer = YoloTrainer(
        model_path=MODEL_PATH,
        parameter_file=PARAMETER_FILE,
    )

    if args.tune:
        settings.update({"wandb": False})
        trainer.tune()
    elif not args.no_train:
        trainer.train()
    if args.validate:
        trainer.validate()
    if args.export:
        trainer.export()
    if args.benchmark:
        trainer.benchmark()
    if args.predict:
        trainer.example_predict(image_path=args.image)


if __name__ == "__main__":
    main()
