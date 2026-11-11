"""Bidirectional parameter conversion between PyTorch state_dict and Flax nested PyTree."""

from pathlib import Path
from typing import Any, Dict
import numpy as np
import safetensors.numpy as st_np

from timestral.config import TimestralConfig


def _set_nested(d: dict, keys: list[str], val: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val


def _get_nested(d: dict, keys: list[str]) -> Any:
    for k in keys:
        d = d[k]
    return d


def torch_to_flax_params(torch_state_dict: Dict[str, Any], config: TimestralConfig) -> Dict[str, Any]:
    """Convert PyTorch state_dict to Flax nested dictionary structure.

    Dense weights are transposed from `(out_features, in_features)` to `(in_features, out_features)`.
    Layer list indices are mapped: `layers.0` -> `layers_0`.
    """
    params: dict = {}

    for k, v in torch_state_dict.items():
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)

        # Map key tokens
        parts = k.split(".")
        new_parts = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                new_parts.append(f"layers_{parts[i + 1]}")
                i += 2
                continue
            elif part in ("attn_norms", "attns") and i + 1 < len(parts) and parts[i + 1].isdigit():
                new_parts.append(f"{part}_{parts[i + 1]}")
                i += 2
                continue
            elif part == "weight" and i == len(parts) - 1:
                # Dense weights in PyTorch become kernels in Flax (transposed)
                # except if 1D (like LayerNorm / RMSNorm weights, but RMSNorm uses 'scale' here)
                if arr.ndim == 2:
                    new_parts.append("kernel")
                    arr = np.transpose(arr, (1, 0))
                else:
                    new_parts.append("weight")
            else:
                new_parts.append(part)
            i += 1

        _set_nested(params, new_parts, arr)

    return params


def flax_to_torch_params(flax_params: Dict[str, Any], config: TimestralConfig) -> Dict[str, np.ndarray]:
    """Convert Flax nested parameters PyTree back to a flat PyTorch state dict mapping.

    Kernels are transposed back to `(out_features, in_features)`.
    """
    flat: Dict[str, np.ndarray] = {}

    def _recurse(d: dict, prefix: list[str]):
        for k, v in d.items():
            if isinstance(v, dict):
                _recurse(v, prefix + [k])
            else:
                arr = np.asarray(v)
                # Transform name back to PyTorch
                py_parts = []
                for p in prefix + [k]:
                    if p.startswith("layers_") and p[len("layers_") :].isdigit():
                        py_parts.extend(["layers", p[len("layers_") :]])
                    elif p.startswith("attn_norms_") and p[len("attn_norms_") :].isdigit():
                        py_parts.extend(["attn_norms", p[len("attn_norms_") :]])
                    elif p.startswith("attns_") and p[len("attns_") :].isdigit():
                        py_parts.extend(["attns", p[len("attns_") :]])
                    elif p == "kernel" and arr.ndim == 2:
                        py_parts.append("weight")
                        arr = np.transpose(arr, (1, 0))
                    else:
                        py_parts.append(p)
                flat[".".join(py_parts)] = arr

    _recurse(flax_params, [])
    return flat


def load_flax_model_from_safetensors(path_or_file: str | Path, config: TimestralConfig) -> Dict[str, Any]:
    """Load Flax parameters from a safetensors file."""
    p = Path(path_or_file)
    if p.is_dir():
        p = p / "model.safetensors"
    tensors = st_np.load_file(str(p))
    return torch_to_flax_params(tensors, config)


def save_flax_model_to_safetensors(flax_params: Dict[str, Any], path_or_file: str | Path, config: TimestralConfig) -> None:
    """Save Flax parameters to a PyTorch-compatible safetensors file."""
    p = Path(path_or_file)
    if p.is_dir():
        p = p / "model.safetensors"
    flat = flax_to_torch_params(flax_params, config)
    st_np.save_file(flat, str(p))
