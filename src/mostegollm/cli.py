"""Command-line interface for MoStegoLLM."""

from __future__ import annotations

import argparse
import sys
import time

from .codec import StegoCodec
from .encoder import TOP_K
from .model import DEFAULT_PROMPT, PRIMARY_MODEL
from .utils import StegoError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mostegollm",
        description="Hide secret data inside LLM-generated English prose.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print diagnostics to stderr"
    )
    parser.add_argument("--model", default=PRIMARY_MODEL, help="HuggingFace model name")
    parser.add_argument(
        "--device", default="auto", help="torch device (auto, cpu, cuda, …)"
    )
    parser.add_argument("--top-k", type=int, default=TOP_K, help="top-k filtering width")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="seed prompt for generation")

    sub = parser.add_subparsers(dest="command")

    # -- encode --------------------------------------------------------
    enc = sub.add_parser("encode", help="encode secret data into cover text")
    enc.add_argument("text", nargs="?", default=None, help="string to encode")
    enc.add_argument("-f", "--file", default=None, help="file to encode")
    enc.add_argument("-o", "--output", default=None, help="write cover text to file")
    enc.add_argument(
        "--sentence-boundary",
        action="store_true",
        default=False,
        help="continue generating until cover text ends at a sentence boundary",
    )

    # -- decode --------------------------------------------------------
    dec = sub.add_parser("decode", help="decode cover text back to secret data")
    dec.add_argument("text", nargs="?", default=None, help="cover text to decode")
    dec.add_argument("-f", "--file", default=None, help="file containing cover text")
    dec.add_argument("-o", "--output", default=None, help="write decoded bytes to file")

    return parser


def _log(msg: str) -> None:
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
    print(f"mostegollm {args.command}: no input (pass a string, -f FILE, or pipe stdin)", file=sys.stderr)
    sys.exit(1)


def _cmd_encode(codec: StegoCodec, args: argparse.Namespace, verbose: bool) -> None:
    raw = _read_input(args)
    data = raw if isinstance(raw, bytes) else raw.encode("utf-8")

    if verbose:
        _log(f"Payload size: {len(data)} bytes")

    t0 = time.perf_counter()
    stats = codec.encode_with_stats(data)
    elapsed = time.perf_counter() - t0

    if verbose:
        _log(f"Tokens generated: {stats.total_tokens}")
        _log(f"Bits per token:   {stats.bits_per_token:.2f}")
        _log(f"Encoding time:    {elapsed:.2f}s")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(stats.cover_text)
        if verbose:
            _log(f"Cover text written to {args.output}")
    else:
        sys.stdout.write(stats.cover_text)
        if sys.stdout.isatty():
            sys.stdout.write("\n")


def _cmd_decode(codec: StegoCodec, args: argparse.Namespace, verbose: bool) -> None:
    raw = _read_input(args)
    cover_text = raw if isinstance(raw, str) else raw.decode("utf-8")

    t0 = time.perf_counter()
    recovered = codec.decode(cover_text)
    elapsed = time.perf_counter() - t0

    if verbose:
        _log(f"Recovered payload: {len(recovered)} bytes")
        _log(f"Decoding time:     {elapsed:.2f}s")

    if args.output:
        with open(args.output, "wb") as fh:
            fh.write(recovered)
        if verbose:
            _log(f"Decoded bytes written to {args.output}")
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

    verbose = args.verbose

    if verbose:
        import torch

        device_str = args.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device_info = device_str
        if device_str.startswith("cuda") and torch.cuda.is_available():
            device_info = f"{device_str} — {torch.cuda.get_device_name()}"
        _log(f"Device: {device_info}")
        _log(f"Loading model: {args.model}")

    t_model = time.perf_counter()
    sentence_boundary = getattr(args, "sentence_boundary", False)
    codec = StegoCodec(
        model_name=args.model,
        device=args.device,
        prompt=args.prompt,
        top_k=args.top_k,
        sentence_boundary=sentence_boundary,
    )
    # Force model load so we can report timing
    codec._ensure_model()
    t_model = time.perf_counter() - t_model

    if verbose:
        _log(f"Model loaded in {t_model:.2f}s")

    try:
        if args.command == "encode":
            _cmd_encode(codec, args, verbose)
        else:
            _cmd_decode(codec, args, verbose)
    except StegoError as exc:
        print(f"mostegollm: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
