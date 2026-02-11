"""Model loading, configuration, and determinism setup."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .utils import StegoModelError

# Default model and fallback
PRIMARY_MODEL = "meta-llama/Llama-3.2-1B"
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Default prompt that seeds the generation context
DEFAULT_PROMPT = "The following is a passage from a book:"


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
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device]:
    """Load a causal language model and tokenizer.

    Tries the requested model first; if it fails (e.g. gated access), falls
    back to ``FALLBACK_MODEL``.

    Args:
        model_name: HuggingFace model identifier.
        device: Device string ('auto', 'cpu', 'cuda', etc.).

    Returns:
        A tuple of (model, tokenizer, resolved_device).

    Raises:
        StegoModelError: If both primary and fallback models fail to load.
    """
    _setup_determinism()
    resolved_device = _resolve_device(device)

    for name in (model_name, FALLBACK_MODEL):
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.float32,  # float32 for determinism
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
            raise StegoModelError(
                f"Failed to load model '{name}': {exc}"
            ) from exc

    # Should not reach here, but satisfy type checker
    raise StegoModelError("No model could be loaded")
