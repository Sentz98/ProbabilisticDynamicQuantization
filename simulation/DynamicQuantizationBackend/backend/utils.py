import importlib.util
import sys
from backend.quantizers import QuantMode

# Default configurations
def_global_cfgs = {
    'def': {'per_channel': True},
    QuantMode.DYNAMIC: {},
    QuantMode.ESTIMATE: {
        'Conv2d': {
            'e_std': 3,
            'sampling_stride': 1,
        },
        'Linear': {
            'e_std': 3,
        }
    },
    QuantMode.STATIC: {'cal_size': 16}    
}

def create_config(config_path: str = None) -> dict:
    """
    Loads and processes a Python configuration file, merging global defaults with
    layer-specific overrides. All mode configuration logic is handled here.
    
    The returned dictionary is structured as:
        {
            "global": { ... global defaults ... },
            "layers": { layer_full_name: {"skip": ..., "mode": ..., ...}
        }
    
    Args:
        config_path (str): Path to the Python configuration file.
        
    Returns:
        dict: The processed configuration dictionary.
    """
    global_configs = {}
    layer_configs = {}
    if config_path:
        try:
            spec = importlib.util.spec_from_file_location("custom_config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            sys.modules["custom_config"] = config_module
            spec.loader.exec_module(config_module)

            if hasattr(config_module, 'global_configs'):
                global_configs = config_module.global_configs
            
            if hasattr(config_module, 'layer_configs'):
                layer_configs = config_module.layer_configs
        except Exception as e:
            print(f"Error loading config file: {e}")
    else:
        print("No config loaded")
    # Determine the base global configuration:
    global_cfg = global_configs.get("global", def_global_cfgs)

    #TODO controlla che le config caricate siano corrette 
    
    return {
        "global": global_cfg,
        "layers": layer_configs
    }

if __name__ == "__main__":
    cfg = create_config("config.py")
    print(cfg)