# DynamicQuantizationBackend

This tool converts PyTorch models by replacing standard Linear and Convolution layers with their quantized versions.

## Features

- **Supported reQuantization Modes:**  
  - **Dynamic**
  - **Estimate**
  - **Static**

- **Per-Layer Customization:**  
  - Override the default quantization mode or parameters for specific layers.
  - Skip conversion for particular layers using a `skip` flag.
  - Use block-level settings so that if a block name (e.g., `"features"`) is defined, its configuration applies to all sublayers unless explicitly overridden.

## Usage

To convert a model, run the conversion script with the following command:

```bash
python convert_model.py \
    --input_model my_model.pth \
    --output_dir ./converted_models \
    --config custom_config.py
```

- **--input_model:** Path to your input PyTorch model file (e.g., `my_model.pth`).
- **--config:** Path to your custom configuration file (default: `config.py`).

## Configuration File Structure

The configuration file is a Python file that must define a dictionary (or variable) with two top-level keys: `"global"` and `"layers"`.

### 1. Global Defaults

Under the key `"global"`, you specify the default quantization parameters for each conversion mode. The default are:

```python
"global": {
    QuantMode.DYNAMIC: {},
    QuantMode.ESTIMATE: {
        'default': {
            'e_std': 2,
            'sampling_stride': 1,
        },
        'Conv2d': {
            'e_std': 2,
            'sampling_stride': 1,
        },
        'Linear': {
            'e_std': 2,
        }
    },
    QuantMode.STATIC: {}    
}
```

> **Note:** Ensure that any references to `QuantMode` are imported or defined in your configuration file. 

### 2. Per-Layer Overrides

Under the key `"layers"`, define overrides for individual layers or blocks using their full names. For example:

```python
"layers": {
    # Skip converting the first linear layer
    'sub.fc1': {'skip': True},
    
    # Custom parameters for second convolutional layer
    'features': {
        'e_std': 3,
        'sampling_stride': 4,
    },

    'features.1': {
        'e_std': 1,
        'sampling_stride': 1,
    },
    
    # Special configuration for final classifier
    'classifier': {
        'mode' : QuantMode.STATIC
    }
}
```

- **Block-Level Overrides:** If a layer name corresponds to a block (e.g., `"features"`), its settings will be propagated to all submodules unless individually overridden.

## How It Works

1. **Configuration Loading:**  
   A configuration loader merges the global defaults with the per-layer (or block) overrides. When determining a layer’s conversion mode, it first checks for a layer-specific `"mode"`, then a block-level setting, and finally falls back to the default. The corresponding base parameters are then retrieved from the `"global"` section and merged with any extra options from the overrides.

2. **Model Conversion:**  
   The conversion script (`convert_model.py`) uses the processed configuration to replace standard `nn.Linear` and `nn.Conv2d` layers with their quantized counterparts. The conversion function recursively applies the configuration so that block-level settings automatically affect all sublayers.

3. **Separation of Concerns:**  
   All merging and mode resolution are handled during configuration loading. The conversion functions simply apply the resulting quantization parameters to each eligible layer.

## Additional Notes

- **Layer Names:**  
  Ensure that the keys in your `"layers"` configuration match the full layer names in your model. Use dot notation for nested modules (e.g., `"block1.conv"`).

- **Supported Layers:**  
  Currently, the tool supports `nn.Linear` and `nn.Conv2d` layers. Support for additional layers (e.g., `nn.Conv1d` or `nn.Conv3d`) can be added as needed.

- **Error Handling:**  
  The script will print an error message if the configuration file cannot be loaded. Double-check the file path and structure if you encounter issues.

By following these instructions, you can customize and run your model conversion with the desired quantization parameters.

---