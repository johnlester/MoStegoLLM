"""Command-line interface for MoStegoLLM."""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from .codec import StegoCodec
from .encoder import TOP_K
from .model import DEFAULT_PROMPT, PRIMARY_MODEL
from .utils import (
    StegoDecodeError,
    StegoError,
    StegoModelError,
    StegoCryptoError,
)


# Separator written between cover texts when encoding with --chunk-size, and
# split on again during decode to reconstruct the chunk list.
CHUNK_SEPARATOR = "\n---\n"


def _add_global_args(p: argparse.ArgumentParser) -> None:
    """Add options accepted either before *or* after the subcommand.

    ``argparse`` normally only accepts parent-parser options before the
    subcommand.  Adding them to both the parent and each subparser with
    ``default=argparse.SUPPRESS`` lets them appear in either position without
    the subparser's copy clobbering a value parsed by the parent.  Real
    defaults are resolved in :func:`main` via ``getattr``.
    """
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="print diagnostics to stderr",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="suppress model loading output on stderr",
    )
    p.add_argument("--model", default=argparse.SUPPRESS, help="HuggingFace model name")
    p.add_argument("--device", default=argparse.SUPPRESS, help="torch device (auto, cpu, cuda, …)")
    p.add_argument("--top-k", type=int, default=argparse.SUPPRESS, help="top-k filtering width")
    p.add_argument("--prompt", default=argparse.SUPPRESS, help="seed prompt for generation")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mostegollm",
        description="Hide secret data inside LLM-generated English prose.",
    )
    _add_global_args(parser)

    sub = parser.add_subparsers(dest="command")

    # -- encode --------------------------------------------------------
    enc = sub.add_parser("encode", help="encode secret data into cover text")
    _add_global_args(enc)
    enc.add_argument("text", nargs="?", default=None, help="string to encode")
    enc.add_argument("-f", "--file", default=None, help="file to encode")
    enc.add_argument("-o", "--output", default=None, help="write cover text to file")
    enc.add_argument(
        "--sentence-boundary",
        action="store_true",
        default=False,
        help="continue generating until cover text ends at a sentence boundary",
    )
    enc.add_argument("-p", "--password", default=None, help="encrypt payload with AES-256-GCM")
    enc.add_argument("--chunk-size", type=int, default=None, help="split into chunks of N bytes")
    enc.add_argument(
        "--topic", default=None, help="cover-story topic for the opener (see 'topics')"
    )
    enc.add_argument("--stats", action="store_true", help="print encoding stats to stderr")

    # -- models --------------------------------------------------------
    sub.add_parser("models", help="list recommended models")

    # -- topics --------------------------------------------------------
    sub.add_parser("topics", help="list cover-story topics")

    # -- decode --------------------------------------------------------
    dec = sub.add_parser("decode", help="decode cover text back to secret data")
    _add_global_args(dec)
    dec.add_argument("text", nargs="?", default=None, help="cover text to decode")
    dec.add_argument("-f", "--file", default=None, help="file containing cover text")
    dec.add_argument("-o", "--output", default=None, help="write decoded bytes to file")
    dec.add_argument("-p", "--password", default=None, help="decrypt payload with AES-256-GCM")
    dec.add_argument(
        "-t",
        "--text",
        dest="text_mode",
        action="store_true",
        help="decode and print as UTF-8 text",
    )

    return parser


def _log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, file=sys.stderr)


def _read_input(args: argparse.Namespace) -> str | bytes:
    """Return the user-supplied input as str (for decode) or bytes (for encode -f)."""
    if args.text is not None:
        return args.text
    if args.file is not None:
        if args.command == "encode":
            with open(args.file, "rb") as fh:
                return fh.read()
        with open(args.file, encoding="utf-8") as fh:
            return fh.read()
    # stdin
    if not sys.stdin.isatty():
        if args.command == "encode":
            return sys.stdin.buffer.read()
        return sys.stdin.read()
    print(
        f"mostegollm {args.command}: no input (pass a string, -f FILE, or pipe stdin)",
        file=sys.stderr,
    )
    sys.exit(1)


def _cmd_models() -> None:
    """Print a formatted table of recommended models."""
    models = StegoCodec.list_models()
    name_w = max(len(m.name) for m in models)
    param_w = max(len(m.parameters) for m in models)

    header = f"  {'Model':<{name_w}}  {'Params':<{param_w}}  Description"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for m in models:
        gated = " [gated]" if m.gated else ""
        print(f"  {m.name:<{name_w}}  {m.parameters:<{param_w}}  {m.description}{gated}")


def _cmd_topics() -> None:
    """Print available cover-story topics with an example opener."""
    from .seeds import TOPICS

    name_w = max(len(name) for name in TOPICS)
    print(f"  {'Topic':<{name_w}}  Example opener")
    print("  " + "-" * (name_w + 16))
    for name, phrases in TOPICS.items():
        example = phrases[0]
        if len(example) > 60:
            example = example[:57] + "..."
        print(f"  {name:<{name_w}}  {example}")


def _cmd_encode(codec: StegoCodec, args: argparse.Namespace, verbose: bool, quiet: bool) -> None:
    raw = _read_input(args)
    data = raw if isinstance(raw, bytes) else raw.encode("utf-8")

    if verbose:
        _log(f"Payload size: {len(data)} bytes")

    chunk_size = getattr(args, "chunk_size", None)
    show_stats = getattr(args, "stats", False) or verbose

    t0 = time.perf_counter()

    if chunk_size is not None:
        result = codec.encode(data, chunk_size=chunk_size)
        assert isinstance(result, list)
        cover_text = CHUNK_SEPARATOR.join(result)
        elapsed = time.perf_counter() - t0
        if show_stats:
            _log(f"Chunks:           {len(result)}")
            _log(f"Encoding time:    {elapsed:.2f}s")
    else:
        if show_stats:
            stats = codec.encode_with_stats(data)
            elapsed = time.perf_counter() - t0
            cover_text = stats.cover_text
            # Stats are printed even if quiet
            _log(f"Tokens generated: {stats.total_tokens}")
            _log(f"Bits per token:   {stats.bits_per_token:.2f}")
            _log(f"Encoding time:    {elapsed:.2f}s")
        else:
            result_str = codec.encode(data)
            assert isinstance(result_str, str)
            cover_text = result_str
            elapsed = time.perf_counter() - t0

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(cover_text)
        if verbose:
            _log(f"Cover text written to {args.output}")
    else:
        sys.stdout.write(cover_text)
        if sys.stdout.isatty():
            sys.stdout.write("\n")


def _cmd_decode(codec: StegoCodec, args: argparse.Namespace, verbose: bool, quiet: bool) -> None:
    raw = _read_input(args)
    cover_text = raw if isinstance(raw, str) else raw.decode("utf-8")

    # Chunked cover text (from `encode --chunk-size`) is joined by CHUNK_SEPARATOR.
    # Split it back into the list form that codec.decode expects so the chained
    # per-chunk prompts can be reconstructed.
    cover_input: str | list[str]
    if CHUNK_SEPARATOR in cover_text:
        cover_input = cover_text.split(CHUNK_SEPARATOR)
    else:
        cover_input = cover_text

    t0 = time.perf_counter()
    recovered = codec.decode(cover_input)
    elapsed = time.perf_counter() - t0

    if verbose:
        _log(f"Recovered payload: {len(recovered)} bytes")
        _log(f"Decoding time:     {elapsed:.2f}s")

    text_mode = getattr(args, "text_mode", False)

    if args.output:
        with open(args.output, "wb") as fh:
            fh.write(recovered)
        if verbose:
            _log(f"Decoded bytes written to {args.output}")
    elif text_mode:
        sys.stdout.write(recovered.decode("utf-8"))
        if sys.stdout.isatty():
            sys.stdout.write("\n")
    else:
        sys.stdout.buffer.write(recovered)
        if sys.stdout.isatty():
            sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "models":
        _cmd_models()
        return

    if args.command == "topics":
        _cmd_topics()
        return

    # Global options use argparse.SUPPRESS defaults (so they work before or
    # after the subcommand); resolve real defaults here.
    verbose = getattr(args, "verbose", False)
    quiet = getattr(args, "quiet", False)

    if quiet:
        # transformers writes weight-loading progress bars to stderr; --quiet
        # promises a silent stderr, so turn them off before the model loads.
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    model_name = getattr(args, "model", PRIMARY_MODEL)
    device = getattr(args, "device", "auto")
    top_k = getattr(args, "top_k", TOP_K)
    prompt = getattr(args, "prompt", DEFAULT_PROMPT)

    if verbose:
        import torch

        device_str = device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device_info = device_str
        if device_str.startswith("cuda") and torch.cuda.is_available():
            device_info = f"{device_str} — {torch.cuda.get_device_name()}"
        _log(f"Device: {device_info}")
        _log(f"Loading model: {model_name}")

    t_model = time.perf_counter()
    sentence_boundary = getattr(args, "sentence_boundary", False)
    password = getattr(args, "password", None)
    topic = getattr(args, "topic", None)
    try:
        codec = StegoCodec(
            model_name=model_name,
            device=device,
            prompt=prompt,
            topic=topic,
            top_k=top_k,
            sentence_boundary=sentence_boundary,
            password=password,
        )
    except ValueError as exc:
        print(f"mostegollm: {exc}", file=sys.stderr)
        sys.exit(1)
    # Force model load so we can report timing
    _log("Loading model…", quiet=quiet)
    codec._ensure_model()
    t_model = time.perf_counter() - t_model

    if verbose:
        _log(f"Model loaded in {t_model:.2f}s")
    else:
        _log("Model ready.", quiet=quiet)

    try:
        if args.command == "encode":
            _cmd_encode(codec, args, verbose, quiet)
        else:
            _cmd_decode(codec, args, verbose, quiet)
    except StegoDecodeError:
        print(
            "Error: could not decode -- wrong model or corrupted text.",
            file=sys.stderr,
        )
        if verbose:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    except StegoModelError as exc:
        print(
            f"Error: could not load model '{model_name}'. {exc}",
            file=sys.stderr,
        )
        if verbose:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    except StegoCryptoError:
        print(
            "Error: decryption failed -- wrong password or tampered data.",
            file=sys.stderr,
        )
        if verbose:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    except StegoError as exc:
        print(f"mostegollm: {exc}", file=sys.stderr)
        if verbose:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
