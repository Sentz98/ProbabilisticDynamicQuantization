import copy
import torch.nn as nn

from backend.layers import QLinear, QConv2d
from backend.quantizers import QuantMode
from backend.utils import create_config


def convert_model(model: nn.Module, mode: QuantMode, config: dict = None, inplace: bool = False, verbose: bool = False) -> nn.Module:
    """
    Converts all Linear and Convolution layers in the model to their corresponding 
    Quantizer versions. This function simply loads the quantization parameters (under 
    the key "quant_params") and mode (if overridden by "q_mode") from the processed 
    config.
    
    Args:
        model (nn.Module): The PyTorch model to convert.
        mode (QuantMode): The default quantization mode to use.
        config (dict, optional): Processed configuration from create_config. If not provided,
                                 an empty config is used.
        inplace (bool): If False, a deep copy of the model is converted.
    
    Returns:
        nn.Module: The converted model.
    """
    if not inplace:
        model = copy.deepcopy(model)

    config = create_config() if config is None else config

    if verbose:
        print(config)
    
    _convert_module(model, '', config, mode, verbose)
    return model

def _convert_module(module: nn.Module, prefix: str, config: dict, quant_mode: QuantMode, verbose: bool):
    """
    Recursively traverses the module tree and converts eligible layers using the 
    pre-merged configuration.
    
    For each layer, the configuration is determined by checking for an entry in
    config["layers"] keyed by the full layer name; if absent, block-level settings
    (if any) are applied, falling back to global defaults.
    
    Args:
        module (nn.Module): The current module to process.
        prefix (str): The hierarchical name (dot-separated) for lookup in the config.
        config (dict): Processed configuration with "global" and "layers" keys.
        quant_mode (QuantMode): Default quantization mode.
    """
    global_cfg = config.get("global", {})
    layers_cfg = config.get("layers", {})

    # Get block-level config for current prefix (if present)
    block_cfg = layers_cfg.get(prefix, {})

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        
        # Get layer-specific config (if present)
        layer_cfg = layers_cfg.get(full_name, {})

        # Check if skip is requested at either block or layer level
        if layer_cfg.get("skip", False) or block_cfg.get("skip", False):
            print(f'!!!! - SKIPPED {full_name} - !!!!')
            continue

        # Determine the effective quantization mode:
        # layer-level overrides block-level, which overrides the default.
        effective_mode = layer_cfg.get("mode", block_cfg.get("mode", quant_mode))
        
        # Lookup base quant_params from the global config using the effective mode.
        base_params = global_cfg.get(effective_mode, {}).copy()
        base_params = base_params.get(child.__class__.__name__, base_params).copy()
        base_params.update(global_cfg.get('def'))

        # Merge additional settings: block-level then layer-level (ignoring skip and mode keys)
        block_override = {k: v for k, v in block_cfg.items() if k not in {"skip", "mode"}}
        layer_override = {k: v for k, v in layer_cfg.items() if k not in {"skip", "mode"}}
        quant_params = {**base_params, **block_override, **layer_override}
        quant_params['full_name'] = full_name

        if isinstance(child, nn.Linear):
            if verbose:
                print(f"Applying config to {full_name}: {effective_mode} {quant_params}")

            new_linear = QLinear(
                child,
                effective_mode,
                **quant_params
            )
            setattr(module, name, new_linear)
            
        elif isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if verbose:
                print(f"Applying config to {full_name}: {effective_mode} {quant_params}")
            
            if isinstance(child, nn.Conv1d):
                raise NotImplementedError(f'Conv1d Not Implemented: Layer {full_name} must be skipped')
            elif isinstance(child, nn.Conv2d):
                qConvClass = QConv2d
            elif isinstance(child, nn.Conv3d):
                raise NotImplementedError(f'Conv3d Not Implemented: Layer {full_name} must be skipped')
            else:
                continue

            new_conv = qConvClass(
                            child,
                            effective_mode, 
                            **quant_params
            )
            setattr(module, name, new_conv)
        else:
            # Recurse into submodules.
            _convert_module(child, full_name, config, quant_mode, verbose)

