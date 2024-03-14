from __future__ import annotations
from pathlib import Path
import torch

from .gtl import MODELS_DIR

DEFAULT_SAVED_MODEL_NAME = f"{MODELS_DIR}/gtl_base_model.pt"
DEFAULT_GUIDANCE_MATRIX_NAME = f"{MODELS_DIR}/guidance_matrix.pt"


def save_base(model: torch.nn.Module, path: str = "", name: str = "") -> Path:
    path = _get_file_path(path, name, DEFAULT_SAVED_MODEL_NAME)
    torch.save(model.state_dict(), path)
    return path


def load_base(model: torch.nn.Module, path: str = "", name: str = ""):
    path = _get_file_path(path, name, DEFAULT_SAVED_MODEL_NAME)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return state_dict


def save_guidance_matrix(
    guidance_matrix: torch.nn.Module, path: str = "", name: str = ""
) -> Path:
    path = _get_file_path(path, name, DEFAULT_GUIDANCE_MATRIX_NAME)
    torch.save(guidance_matrix, path)
    return path


def load_guidance_matrix(path: str = "", name: str = ""):
    path = _get_file_path(path, name, DEFAULT_GUIDANCE_MATRIX_NAME)
    return torch.load(path)


def _get_file_path(path: str, name: str, default_name: str) -> Path:
    path = Path.cwd() if path == "" else Path(path)
    name = default_name if name == "" else name
    return path / name
