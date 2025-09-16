import torch
import gc

def clear_gpu_memory():
    """Clear GPU memory to prevent memory leaks"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # For Apple Silicon
        torch.mps.empty_cache()