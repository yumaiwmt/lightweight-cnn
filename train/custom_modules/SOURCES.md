# Open-source model blocks

These are experimental model adaptations assembled from the sources below.
They are not verified reproductions of a particular paper, and no knowledge
distillation or pretrained weights are included. Preserve this attribution and
the accompanying AGPL-3.0 license when redistributing the code.

| Code | Pinned source |
| --- | --- |
| `GSConv` | [Slim-Neck by GSConv, `conv.py`](https://github.com/alanli1997/slim-neck-by-gsconv/blob/56813dd43394e608ee49f00d4dfdff5eb666ad18/gsconv-yolov8_9_10_11/ultralytics/nn/modules/conv.py) |
| `GSBottleneck`, `VoVGSCSP`, `Partial_conv3`, `Faster_Block`, `C2f_Faster` | [BerryNet, `block.py`](https://github.com/UGA-BSAIL/BerryNet/blob/3efef5ec5e32e215f62ee5995dc6178fe658e774/yolov8-BerryNet/ultralytics/nn/extra_modules/block.py) |
| `Conv_GN`, `DEConv_GN`, `Scale`, `Detect_LSDECD` | [EAUAV-YOLO, `head.py`](https://github.com/123456shiro/EAUAV-YOLO/blob/82c2bbb4be22ee844195ba9542a90430e32979a3/ultralytics/nn/modules/head.py) |
| `DEConv` and its difference-convolution helpers | [EAUAV-YOLO, `deconv.py`](https://github.com/123456shiro/EAUAV-YOLO/blob/82c2bbb4be22ee844195ba9542a90430e32979a3/ultralytics/nn/modules/deconv.py) |
| Framework | [Ultralytics v8.3.0, commit `6e43d1e`](https://github.com/ultralytics/ultralytics/tree/6e43d1e1e5db72afbf686dee6745669bcb124b0a) |

## Naming and integration

- `C2fFaster` aliases upstream `C2f_Faster`, preserving the YAML name.
- `VoVGSCSP1` aliases the upstream simple `VoVGSCSP` implementation. The selected
  upstream files do not define a separate `VoVGSCSP1` class. Both YAML spellings
  intentionally select the same block; a different variant requires its own
  implementation and comparison.
- The Slim-Neck port uses the author's YOLOv8 `GSConv` with SiLU and channel
  shuffle, plus BerryNet's repeated `GSBottleneck` aggregation. Unused helper
  classes from the upstream files are omitted. The upstream unused `res`
  convolution in `VoVGSCSP` is retained.
- The Faster port retains its upstream partial-convolution, MLP, and stochastic
  depth behavior. `DropPath` comes from `timm.layers` instead of the deprecated
  `timm.models.layers` import.
- `Detect_LSDECD` is named `LSDECD` and subclasses `Detect` for Ultralytics stride
  initialization, device conversion, and export handling. Its constructor calls
  `nn.Module.__init__` because it supplies its own detection branches. Its
  upstream prediction, decoding, and bias initialization are retained.
- Difference-convolution weight allocation uses `weight.new_zeros`, preserving
  the device and dtype for mixed precision. Native `flatten`/`reshape` replace
  equivalent `einops.Rearrange` operations; unused radial convolution is omitted.
- The parser supplies channel counts, block repetitions, and scaled LSDECD
  hidden channels. `[nc, 256]` becomes 64 hidden channels at nano width 0.25.
- Explicit YAML `scale: n` is preserved when filenames do not encode a scale.
- The patch restores the original YOLOv8 classification branch from
  [Ultralytics v8.2.103](https://github.com/ultralytics/ultralytics/blob/v8.2.103/ultralytics/nn/modules/head.py),
  using two standard convolutions rather than the v8.3.0 depthwise branch.
  This checkout is intended for this project's YOLOv8 experiments.

## Reproducible setup

`train/setup_ultralytics.py` checks out the pinned framework, applies
`train/ultralytics.patch`, and copies these three Python modules into it. The
generated `vendor/` directory is ignored by the parent repository. Keep the
setup script, patch, these sources, and license in version control; a fresh
RunPod can recreate the modified framework without copying a nested Git checkout.

The setup command can be rerun after editing these modules. It refreshes the
copied files and does not apply the patch twice. Restart an active notebook
kernel after changing imported code.
