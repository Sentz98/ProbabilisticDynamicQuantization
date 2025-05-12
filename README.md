# A probablistic framework for dynamic quantization

This repository contains code for simulating and deploying our quantization approach, as presented in the paper.

## 📂 Repository Structure

### `simulation/`
Code for simulating our quantization method on:
- **Ultralytics models** (YOLO-based architectures)
- Common classification models (e.g., ResNet, MobileNet) evaluated on:
  - ImageNet (1K classes)
  - CIFAR-100 (100 classes)

### `deployment/`
Modified CMSIS-NN 5 code for microcontroller deployment

Tested on the microcontroller used in our paper:
- STM32L4 series (Cortex-M4)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ with PyTorch 1.12+
- CMSIS 5.8.0+ (for deployment)
- ARM GCC toolchain (for cross-compilation)

### Installation
```bash
git clone https://github.com/Sentz98/ProbabilisticDynamicQuantization.git
cd ProbabilisticDynamicQuantization

# Install simulation requirements
pip install -r simulation/requirements.txt

# CMSIS setup (see deployment/README.md for detailed instructions)
