"""Model loading, configuration, and determinism setup."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .utils import StegoModelError

# Default model and fallback
PRIMARY_MODEL = "HuggingFaceTB/SmolLM-135M"
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Default prompt that seeds the generation context
DEFAULT_PROMPT = "The following is a passage from a book:"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a recommended model."""

    name: str
    description: str
    parameters: str
    gated: bool = False


MODEL_REGISTRY: tuple[ModelInfo, ...] = (
    ModelInfo(
        name="HuggingFaceTB/SmolLM-135M",
        description="Tiny, fast default model (recommended)",
        parameters="135M",
    ),
    ModelInfo(
        name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="Small chat model used as fallback",
        parameters="1.1B",
    ),
    ModelInfo(
        name="HuggingFaceTB/SmolLM-360M",
        description="Larger SmolLM variant, better prose quality",
        parameters="360M",
    ),
    ModelInfo(
        name="Qwen/Qwen2.5-0.5B",
        description="Compact multilingual model",
        parameters="0.5B",
    ),
    ModelInfo(
        name="meta-llama/Llama-3.2-1B",
        description="High-quality Meta model (requires HF_TOKEN)",
        parameters="1B",
        gated=True,
    ),
)


def list_models() -> tuple[ModelInfo, ...]:
    """Return all recommended models."""
    return MODEL_REGISTRY


def get_model_info(model_name: str) -> ModelInfo | None:
    """Look up a model by name. Returns ``None`` if not in the registry."""
    for info in MODEL_REGISTRY:
        if info.name == model_name:
            return info
    return None


# ---------------------------------------------------------------------------
# HuggingFace token helpers
# ---------------------------------------------------------------------------


def _get_hf_token() -> str | None:
    """Load ``.env`` and return the ``HF_TOKEN`` environment variable, if set."""
    load_dotenv()
    return os.environ.get("HF_TOKEN") or None


def _resolve_device(device: str) -> torch.device:
    """Resolve 'auto' device string to an actual torch device.

    Args:
        device: One of 'auto', 'cpu', 'cuda', 'cuda:0', etc.

    Returns:
        A ``torch.device`` instance.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _setup_determinism() -> None:
    """Configure PyTorch for maximum determinism."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def load_model(
    model_name: str,
    device: str = "auto",
    token: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device]:
    """Load a causal language model and tokenizer.

    Tries the requested model first; if it fails (e.g. gated access), falls
    back to ``FALLBACK_MODEL``.

    Args:
        model_name: HuggingFace model identifier.
        device: Device string ('auto', 'cpu', 'cuda', etc.).
        token: HuggingFace API token for gated models.  When ``None``,
            falls back to the ``HF_TOKEN`` environment variable / ``.env``.

    Returns:
        A tuple of (model, tokenizer, resolved_device).

    Raises:
        StegoModelError: If both primary and fallback models fail to load.
    """
    _setup_determinism()
    resolved_device = _resolve_device(device)
    hf_token = token or _get_hf_token()

    for name in (model_name, FALLBACK_MODEL):
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, token=hf_token)
            model = AutoModelForCausalLM.from_pretrained(
                name,
                dtype=torch.float32,  # float32 for determinism
                token=hf_token,
            )
            model = model.to(resolved_device)
            model.eval()

            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            return model, tokenizer, resolved_device
        except Exception as exc:
            if name == model_name and name != FALLBACK_MODEL:
                # Primary failed, will try fallback
                continue
            raise StegoModelError(f"Failed to load model '{name}': {exc}") from exc

    # Should not reach here, but satisfy type checker
    raise StegoModelError("No model could be loaded")
