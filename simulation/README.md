## Installation

1. **Install Modified Ultralytics Package**  
   Install the custom Ultralytics package in editable mode:
   ```bash
   pip install -e ./ultralytics/
   ```

2. **Set Dynamic Quantization Backend Path**  
   Export the backend path before running any scripts (add this to your shell profile for convenience):
   ```bash
   export DQPATH=$PWD/DynamicQuantizationBackend
   ```

## Quantization Parameters

Configure quantization using the following environment variables:

| Variable    | Values                                                                 | Default |
|-------------|-----------------------------------------------------------------------|---------|
| `dq`        | `0` (Floating Point), `1` (Estimate), `2` (Dynamic), `3` (Static)     | `0`     |
| `corrupt`   | `1` (Enable corruption), `0` (Disable corruption)                    | `1`     |
| `ch`        | `1` (Per-channel quantization), `0` (Per-tensor quantization)        | `1`     |
| `stride`    | Sampling stride for quantization (integer value)                     | `1`     |
| `cal_size`  | Number of samples in calibration dataset (integer value)             | `16`    |

## Usage Examples

### 1. Validate on COCO Dataset
```bash
# Example: Dynamic quantization with per-channel quantization
dq=2 ch=1 yolo val detect data=coco8.yaml
```

### 2. Validate on ImageNet
```bash
# Example: Validate ResNet50 with batch size 128 on GPU 2
python imagenet.py \
  --data /datasets/imagenet \
  --gpu 2 \
  --b 128 \
  --wk 4 \
  --model resnet50

# For MobileNetV2
python imagenet.py \
  --data /datasets/imagenet \
  --model mobilenetv2
```

### 3. Validate on CIFAR-100
```bash
# Example: Test ResNet50 with pretrained weights
python test.py \
  -net resnet50 \
  -weights resnet50-best.pth
```

## Notes
- Ensure the `DQPATH` environment variable is set before running any quantization scripts.
- Dataset paths (e.g., `/datasets/imagenet`) should be updated to match your local setup.
