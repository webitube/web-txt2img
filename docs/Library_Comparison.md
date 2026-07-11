# Lightweight CNN Library Comparison: MobileNet, SqueezeNet, FER+, and Ultralytics

## Executive Summary

This document compares four lightweight/deep learning libraries found in the workspace:

| Library | Framework | Primary Focus | Model Size | Year | License |
|---------|-----------|---------------|------------|------|---------|
| **MobileNet** | TensorFlow 1.x | Image classification | ~3.3M params | 2017 | Apache 2.0 |
| **SqueezeNet** | Caffe | Image classification | ~1M params | 2016 | Apache 2.0 |
| **FER+** | CNTK | Facial expression recognition | ~1M params (VGG13) | 2016 | MIT |
| **Ultralytics** | PyTorch | Multi-task (detection, segmentation, classification, pose) | 2.4M–55.7M params | 2023–present | AGPL-3.0 |

---

## 1. MobileNet (TensorFlow)

### Overview
- **Repo**: `D:\Dev\mobilenet`
- **Paper**: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- **Framework**: TensorFlow 1.x with `tf.contrib.slim`
- **Status**: Legacy (TF 1.x APIs deprecated)

### Architecture
- **Core innovation**: Depthwise separable convolutions
- **Structure**:
  - Initial 3×3 convolution (32 filters)
  - 14 depthwise separable convolution blocks
  - Global average pooling (7×7)
  - Fully connected layer (1000 classes)
- **Width multiplier**: Supports α ∈ {0.25, 0.5, 0.75, 1.0} for model scaling
- **Input**: 224×224×3 RGB images

### Key Files
```
nets/mobilenet.py              # Core model definition
nets/mobilenetdet.py           # Object detection variant (KITTI)
preprocessing/mobilenet_preprocessing.py  # Image augmentation
train_image_classifier.py      # Training script with YellowFin optimizer
tools/freeze_graph.py          # Graph freezing for deployment
tools/quantize_graph.py        # 8-bit quantization
optimizer/yellowfin/           # YellowFin optimizer implementation
configs/                       # Training configurations
```

### Performance
| Metric | Value |
|--------|-------|
| Top-1 Accuracy (ImageNet) | 66.51% |
| Top-5 Accuracy (ImageNet) | 87.09% |
| Parameters | 3.3M (base) |
| CPU Inference (TF 1.1, AVX2) | 19ms |
| GPU Inference (GTX 1060) | 3ms |
| Training Data | ImageNet-2012 |

### Training Setup
```bash
bash ./script/train_mobilenet_on_imagenet.sh
```
- Optimizer: RMSprop or YellowFin
- Batch size: Configurable via `num_clones`
- Learning rate: Exponential/polynomial decay
- Preprocessing: Inception-style with color distortion

### Export & Deployment
- **Freeze graph**: Converts checkpoints to standalone GraphDef
- **Quantization**: 8-bit weight quantization for CPU acceleration
- **Fused batch norm**: Bakes BN into conv weights for faster inference

### Limitations
- TensorFlow 1.x is end-of-life
- `tf.contrib` modules removed in TF 2.x
- No native TF 2.x / Keras migration
- Limited to image classification and basic detection

---

## 2. SqueezeNet (Caffe)

### Overview
- **Repo**: `D:\Dev\SqueezeNet`
- **Paper**: [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters](http://arxiv.org/abs/1602.07360)
- **Framework**: Caffe (prototxt + caffemodel)
- **Status**: Legacy (Caffe largely superseded)

### Architecture
- **Core innovation**: Fire modules (squeeze → expand)
- **Structure**:
  - Initial convolution (96 filters v1.0 / 64 filters v1.1)
  - 9 Fire modules with squeeze (1×1) and expand (1×1 + 3×3) layers
  - Final convolution and loss layer
- **Input**: 227×227×3 RGB images

### Versions
| Version | Params | FLOPs | Notes |
|---------|--------|-------|-------|
| v1.0 | ~1M | ~0.8M | Original paper model |
| v1.1 | ~1M | ~0.3M | 2.4× less computation, same accuracy |

### Key Files
```
SqueezeNet_v1.0/
  train_val.prototxt              # Model architecture
  solver.prototxt                 # Training hyperparameters
  squeezenet_v1.0.caffemodel      # Pretrained weights
  deploy.prototxt                 # Inference graph

SqueezeNet_v1.1/
  train_val.prototxt              # Optimized architecture
  solver.prototxt                 # Training hyperparameters
  squeezenet_v1.1.caffemodel      # Pretrained weights
  deploy.prototxt                 # Inference graph
```

### Training Configuration (v1.0)
```
Base Learning Rate: 0.04
Batch Size: 32 × 16 (iter_size) = 512
Max Iterations: 170,000
LR Policy: Polynomial (linear decay)
Momentum: 0.9
Weight Decay: 0.0002
```

### Performance
| Metric | Value |
|--------|-------|
| Top-1 Accuracy (ImageNet) | ~58.1% (AlexNet-level) |
| Model Size | 4.8 MB uncompressed |
| Parameters | ~1M |
| Training Data | ImageNet-1K |

### Community Ports
- MXNet, Chainer, Keras, TensorFlow, PyTorch, CoreML
- Compressed variants and residual connections available

### Limitations
- Caffe framework is largely unmaintained
- Prototxt format not human-friendly
- Limited flexibility for custom architectures
- No built-in data augmentation beyond cropping

---

## 3. FER+ (Facial Expression Recognition)

### Overview
- **Repo**: `D:\Dev\ferplus`
- **Paper**: [Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution](https://arxiv.org/abs/1608.01041)
- **Framework**: Microsoft Cognitive Toolkit (CNTK)
- **Status**: Legacy (CNTK discontinued by Microsoft)

### Architecture
- **Model**: VGG13-like network adapted for emotion data
- **Structure**:
  - 4 convolutional blocks (64→128→256→256 filters)
  - Max pooling after each block
  - 2 fully connected layers (1024 units)
  - Output layer (8 emotion classes)
- **Input**: 64×64×1 grayscale face images

### Key Innovation
- **Multi-label training modes**:
  1. **Majority voting**: Single label per image
  2. **Probability**: Soft labels from crowd-sourced distributions
  3. **Cross-entropy**: Direct probability optimization
  4. **Multi-target**: Binary relevance per emotion

### Key Files
```
src/
  models.py              # VGG13 model definition
  train.py               # Training loop with CNTK
  ferplus.py             # Data reader and augmentation
  img_util.py            # Image preprocessing utilities
  rect_util.py           # Face rectangle utilities
  generate_training_data.py  # Dataset preparation

data/
  FER2013Train/          # Training split
  FER2013Valid/          # Validation split
  FER2013Test/           # Test split
```

### Emotion Classes
| Index | Emotion |
|-------|---------|
| 0 | Neutral |
| 1 | Happiness |
| 2 | Surprise |
| 3 | Sadness |
| 4 | Anger |
| 5 | Disgust |
| 6 | Fear |
| 7 | Contempt |

### Training Setup
```bash
# Generate training data
python generate_training_data.py -d <base_folder> -fer <fer2013.csv> -ferplus <fer2013new.csv>

# Train with majority voting
python train.py -d <base_folder> -m majority

# Train with probability mode
python train.py -d <base_folder> -m probability
```

### Data Augmentation
- Random shift (±8%)
- Random scale (up to 1.05×)
- Random rotation (±20°)
- Random skew (±5%)
- Horizontal flipping
- Z-score normalization

### Limitations
- CNTK is discontinued
- FER2013 dataset has known biases
- Limited to facial expression task
- Requires manual dataset preparation from Kaggle

---

## 4. Ultralytics YOLO

### Overview
- **Repo**: `D:\Dev\ultralytics`
- **Docs**: [docs.ultralytics.com](https://docs.ultralytics.com)
- **Framework**: PyTorch
- **Status**: Actively maintained (v8.4.92)
- **License**: AGPL-3.0

### Architecture
- **Models**: YOLOv3, YOLOv5, YOLOv8, YOLO11, YOLO26, RT-DETR, SAM, NAS
- **Tasks**: Detection, segmentation, classification, pose estimation, tracking
- **Backbone**: CSPDarknet variants with modern improvements

### YOLO26 Detection Models (COCO)
| Model | Size | mAP val 50-95 | Params | FLOPs | CPU Speed | GPU Speed |
|-------|------|---------------|--------|-------|-----------|-----------|
| YOLO26n | 640 | 40.9% | 2.4M | 5.4B | 38.9ms | 1.7ms |
| YOLO26s | 640 | 48.6% | 9.5M | 20.7B | 87.2ms | 2.5ms |
| YOLO26m | 640 | 53.1% | 20.4M | 68.2B | 220ms | 4.7ms |
| YOLO26l | 640 | 55.0% | 24.8M | 86.4B | 286ms | 6.2ms |
| YOLO26x | 640 | 57.5% | 55.7M | 193.9B | 526ms | 11.8ms |

### Key Files
```
ultralytics/
  models/           # Model implementations (YOLO, SAM, RT-DETR, etc.)
  engine/           # Training, validation, prediction, export engines
  nn/               # Neural network modules, backends, and heads
  data/             # Data loading, augmentation, and dataset utilities
  optim/            # Optimizers and learning rate schedulers
  trackers/         # Multi-object tracking implementations
  solutions/        # High-level vision solutions
  utils/            # Logging, exports, checks, and helpers
  cfg/              # Default configurations
  hub/              # Ultralytics Hub integration

Key architecture files:
  nn/autobackend.py     # AutoBackend: dynamic backend selection (20+ formats)
  nn/backends/onnx.py   # ONNX Runtime with CUDA/CoreML/IO binding support
  nn/modules/head.py    # Detect, Segment, Pose heads with end2end support
  engine/exporter.py    # Export pipeline for all supported formats
  engine/results.py     # Results class for detection/segmentation/classification
  utils/nms.py          # Non-maximum suppression with end2end support
  utils/export/onnx.py  # ONNX INT8 quantization with calibration
```

### Features
- **CLI and Python API**:
  ```bash
  yolo predict model=yolo26n.pt source='bus.jpg'
  yolo train model=yolo26n.pt data=coco.yaml epochs=100
  ```
  ```python
  from ultralytics import YOLO
  model = YOLO("yolo26n.pt")
  results = model("path/to/image.jpg")
  ```

- **Export formats**: 20+ formats including ONNX, TensorRT, CoreML, TFLite/LiteRT, OpenVINO, PaddlePaddle, ncnn, MNN, RKNN, IMX, ExecuTorch, Axelera, DeepX, QNN
- **AutoBackend system**: Dynamic backend selection based on file suffix; supports PyTorch, TorchScript, ONNX (Runtime + OpenCV DNN), TensorRT, OpenVINO, CoreML, TensorFlow, and embedded AI accelerators
- **ONNX Runtime optimizations**: IO binding for zero-copy GPU inference, CUDA/CoreML execution providers, dynamic shape support
- **Quantization**: FP16, INT8 (static with calibration data), W8A16 (INT8 weights + INT16 activations), W8A32 (dynamic INT8)
- **Data augmentation**: Mosaic, mixup, copy-paste, autoaugment
- **Mixed precision**: FP16/BF16 training support
- **Multi-GPU**: DDP training out of the box
- **Tracking**: BoT-SORT, ByteTrack, Kalman filter-based
- **End-to-end detection**: NMS-free inference mode with one-to-one and one-to-many heads

### Dependencies
```
Python >= 3.8
PyTorch >= 1.8
torchvision >= 0.9
numpy, matplotlib, opencv-python, pillow
pyyaml, requests, psutil, polars
```

### Strengths
- Actively developed with regular updates
- Comprehensive documentation and examples
- Multi-task support in single codebase
- Production-ready export pipelines
- Large community and enterprise support

### Considerations
- AGPL-3.0 license requires open-sourcing derivative works
- Larger dependency footprint
- Heavier than pure inference libraries

### Ultralytics Internals (Source Code Analysis)

#### AutoBackend Architecture
The `AutoBackend` class (`nn/autobackend.py`) provides a unified inference interface across 20+ backends:
- **Backend dispatch**: Maps file suffixes (`.pt`, `.onnx`, `.engine`, etc.) to backend classes
- **ONNX Runtime**: Uses IO binding for zero-copy GPU inference when shapes are static
- **Fallback chain**: CUDA → CoreML → CPU execution providers
- **Metadata propagation**: Class names, stride, and task info embedded in exported models

#### Detect Head (`nn/modules/head.py`)
- **One-to-many head**: Traditional YOLO head with NMS post-processing (cv2 for boxes, cv3 for classes)
- **One-to-one head**: End-to-end mode bypasses NMS via top-k selection
- **DFL (Distribution Focal Loss)**: Reg_max=16 channels for refined box regression
- **Legacy mode**: Backward compatibility for v3/v5/v8/v9/v11 models

#### Export Pipeline (`engine/exporter.py`)
- **Environment isolation**: Each export format has isolated Python environments (version pinning)
- **Precision support matrix**:
  - FP16: TorchScript, ONNX, OpenVINO, TensorRT, CoreML, MNN, NCNN, RKNN
  - INT8: ONNX, OpenVINO, TensorRT, CoreML, SavedModel, EdgeTPU, MNN, NCNN, IMX, RKNN, Axelera, DeepX, LiteRT
  - W8A16: CoreML, IMX, QNN, LiteRT
- **INT8 calibration**: Uses COCO8/COCO128 subsets with configurable fractions
- **Validation**: Asserts output model size > 0.1 MB to catch corrupt exports

#### Results System (`engine/results.py`)
- **Unified Results class**: Handles detection, segmentation, classification, pose, OBB
- **BaseTensor**: Lazy coordinate scaling between image size and original shape
- **Export formats**: JSON, CSV, DataFrame (Polars), text files, cropped images

#### NMS (`utils/nms.py`)
- **Configurable thresholds**: Confidence (0.25), IoU (0.45), max detections (300)
- **End-to-end mode**: Skips NMS when model outputs pre-sorted top-k detections
- **Rotated boxes**: OBB support via Probabilistic IoU (ProIoU)

---

## Comparative Analysis

### Framework Maturity
| Library | Framework | Maintenance | Modernity |
|---------|-----------|-------------|-----------|
| MobileNet | TF 1.x | ❌ EOL | Legacy |
| SqueezeNet | Caffe | ❌ EOL | Legacy |
| FER+ | CNTK | ❌ Discontinued | Legacy |
| Ultralytics | PyTorch | ✅ Active | Modern |

### Model Efficiency
| Library | Params | Input Size | Top-1/mAP | FLOPs |
|---------|--------|------------|-----------|-------|
| MobileNet 1.0 | 3.3M | 224×224 | 66.51% | ~534M |
| SqueezeNet 1.1 | 1M | 227×227 | 58.1% | ~0.3M |
| FER+ VGG13 | ~1M | 64×64 | N/A (8-class) | ~0.1B |
| YOLO26n | 2.4M | 640×640 | 40.9% mAP | 5.4B |

### Task Coverage
| Library | Classification | Detection | Segmentation | Pose | Tracking |
|---------|----------------|-----------|--------------|------|----------|
| MobileNet | ✅ | ✅ (basic) | ❌ | ❌ | ❌ |
| SqueezeNet | ✅ | ❌ | ❌ | ❌ | ❌ |
| FER+ | ✅ (8-class) | ❌ | ❌ | ❌ | ❌ |
| Ultralytics | ✅ | ✅ | ✅ | ✅ | ✅ |

### Deployment Options
| Library | Frozen Graph | Quantization | ONNX | TFLite | TensorRT | CoreML |
|---------|--------------|--------------|------|--------|----------|--------|
| MobileNet | ✅ | ✅ (8-bit) | ❌ | ❌ | ❌ | ❌ |
| SqueezeNet | ✅ (caffemodel) | ❌ | ❌ | ❌ | ❌ | ❌ |
| FER+ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ultralytics | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Data Augmentation
| Library | Color Jitter | Geometric | Mosaic | Mixup | AutoAugment |
|---------|-------------|-----------|--------|-------|-------------|
| MobileNet | ✅ | ✅ | ❌ | ❌ | ❌ |
| SqueezeNet | ❌ | ✅ (crop) | ❌ | ❌ | ❌ |
| FER+ | ❌ | ✅ (extensive) | ❌ | ❌ | ❌ |
| Ultralytics | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Recommendations

### For Learning/Research
- **MobileNet**: Study depthwise separable convolutions and width multipliers
- **SqueezeNet**: Understand Fire modules and parameter efficiency
- **FER+**: Explore multi-label training and crowd-sourced label distributions

### For Production
- **Ultralytics**: Only actively maintained option with modern deployment support
- Consider YOLO26n (2.4M params) for edge deployment
- Use export pipelines for target platform optimization

### For Mobile/Edge
1. **Ultralytics YOLO26n** → TFLite/CoreML export
2. **MobileNet** → Frozen graph + 8-bit quantization (legacy)
3. **SqueezeNet** → Smallest model but limited framework support

### Migration Path
```
Legacy (TF1/Caffe/CNTK)          Modern (PyTorch)
─────────────────────    →       ─────────────────
MobileNet (TF1)                Ultralytics YOLO
SqueezeNet (Caffe)             Ultralytics YOLO
FER+ (CNTK)                    Ultralytics YOLO (custom head)
```

---

## References

1. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv:1704.04861, 2017
2. Iandola et al., "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters," arXiv:1602.07360, 2016
3. Barsoum et al., "Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution," ICMI 2016, arXiv:1608.01041
4. Ultralytics, "YOLO26 Documentation," https://docs.ultralytics.com
