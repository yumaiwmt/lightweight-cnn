# Training the three models

Use Python 3.10 or newer. On RunPod, select an environment with CUDA-enabled
PyTorch and upload the dataset images and labels as well as this repository.
The dataset splits are ignored by Git and will not arrive through a clone alone.

From the repository root:

```bash
python train/setup_ultralytics.py
python -m pip install -e ./vendor/ultralytics "timm==1.0.31" wandb
```

The setup script downloads a pinned Ultralytics source checkout, applies the
small integration patch, and copies the open-source modules into it. Sources,
licenses, name mappings, and compatibility changes are documented in
[custom_modules/SOURCES.md](custom_modules/SOURCES.md).

Check the installation in the same Python environment as the notebook:

```python
import torch
import ultralytics

print(ultralytics.__file__)  # should point inside vendor/ultralytics
print(torch.cuda.is_available())  # should be True on a GPU RunPod
```

Run the small training/validation/checkpoint test before a full comparison:

```bash
python train/smoke_test.py --device 0 --imgsz 960
```

For a local CPU functionality check, use `--device cpu --imgsz 128`. The test
uses four training images and two validation images per model for one epoch;
these results are not accuracy or full-training performance estimates.

Open `train/train.ipynb`, run its installation cell, restart the kernel after the
first installation, then choose `MODEL_NAME` in Setup:

```python
MODEL_NAME = "yolov8_baseline.yaml"
# MODEL_NAME = "ADown _C2fFaster.yaml"  # includes a space after ADown
# MODEL_NAME = "GSConv_VoV-GSCP1.yaml"
# MODEL_NAME = None  # train all architectures sequentially
```

The default trains the baseline alone. Run Setup, Check selected architectures,
and Train selected models. To train another model later, change `MODEL_NAME` and
rerun those cells. Each Setup creates a separate output directory and preserves
previous runs. All three architectures use the nano scale and the dataset's five
classes.

The shared settings in `train/cfg/v3.yaml` specify 500 maximum epochs, 960-pixel
images, SGD, automatic batch sizing, and validation during training. Early
stopping can finish before 500 epochs. For a strictly fixed batch across models,
replace `batch: -1` with a size that fits every model. `workers: 4` is a reasonable
starting point on RunPod; `workers: 0` supports Windows notebooks.

Each comparison has its own `runs/comparison_<timestamp>/` directory containing:

- A portable dataset YAML with the dataset root resolved for that machine.
- A folder per model with `weights/best.pt`, `weights/last.pt`, and `results.csv`.
- `comparison.csv` with training times and validation metrics.

The optional separate-validation cell reevaluates checkpoints trained in the
current run. The export, prediction, and TensorRT benchmark cells default to the
first model trained in this run through `SELECTED_MODEL`; change that selection
to process another model from the same run. TensorRT
requires a suitable NVIDIA GPU and runtime and has not been checked by the CPU
test. Build deployment engines on compatible deployment hardware.

The notebook's training cell starts new runs when executed again. It does not
automatically resume interrupted runs. Copy the complete run directory to
persistent storage before terminating a RunPod.
