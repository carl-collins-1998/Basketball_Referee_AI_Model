import torch
import torch.serialization

# Fix for PyTorch compatibility
try:
    from ultralytics.nn.tasks import DetectionModel

    # Check if add_safe_globals exists (PyTorch 2.0+)
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([DetectionModel])
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Could not add DetectionModel to safe globals: {e}")

# Monkey patch torch.load for YOLO models
_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    # For YOLO model files, disable weights_only restriction
    if args and isinstance(args[0], str) and args[0].endswith('.pt'):
        # Check PyTorch version
        torch_version = torch.__version__.split('.')
        major_version = int(torch_version[0])
        minor_version = int(torch_version[1].split('+')[0])  # Handle versions like 1.13.1+cu117

        # For PyTorch >= 1.13, handle weights_only parameter
        if major_version > 1 or (major_version == 1 and minor_version >= 13):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False

        # For older versions, remove weights_only if present
        elif 'weights_only' in kwargs:
            del kwargs['weights_only']

    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

print("YOLO loader fix applied successfully")