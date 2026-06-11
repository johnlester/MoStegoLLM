"""CLI integration tests for mostegollm."""

from __future__ import annotations

import subprocess
import sys


def _run(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess:
    """Run mostegollm CLI and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "mostegollm.cli"] + args,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=300,
    )


class TestCLIEncode:
    def test_encode_string(self) -> None:
        result = _run(["encode", "hello"])
        assert result.returncode == 0
        assert len(result.stdout) > 0

    def test_encode_stats(self) -> None:
        result = _run(["encode", "hello", "--stats"])
        assert result.returncode == 0
        assert "token" in result.stderr.lower()

    def test_encode_quiet(self) -> None:
        result = _run(["encode", "hi", "--quiet"])
        assert result.returncode == 0
        assert result.stderr == ""


class TestCLIDecode:
    def test_roundtrip(self) -> None:
        enc = _run(["encode", "roundtrip test"])
        assert enc.returncode == 0
        dec = _run(["decode"], input_text=enc.stdout)
        assert dec.returncode == 0

    def test_text_flag(self) -> None:
        enc = _run(["encode", "text flag test"])
        assert enc.returncode == 0
        dec = _run(["decode", "--text"], input_text=enc.stdout)
        assert dec.returncode == 0
        assert "text flag test" in dec.stdout

    def test_password_roundtrip(self) -> None:
        enc = _run(["encode", "secret", "--password", "mypass"])
        assert enc.returncode == 0
        dec = _run(["decode", "--text", "--password", "mypass"], input_text=enc.stdout)
        assert dec.returncode == 0
        assert "secret" in dec.stdout

    def test_wrong_password_fails(self) -> None:
        enc = _run(["encode", "secret", "--password", "right"])
        assert enc.returncode == 0
        dec = _run(["decode", "--text", "--password", "wrong"], input_text=enc.stdout)
        assert dec.returncode != 0

    def test_decode_garbage(self) -> None:
        dec = _run(["decode", "this is not encoded text at all"])
        assert dec.returncode != 0
        assert "error" in dec.stderr.lower()


class TestCLIChunked:
    def test_chunked_roundtrip(self) -> None:
        """encode --chunk-size output must decode back through the CLI."""
        secret = "chunk round trip test " * 3
        enc = _run(["encode", secret, "--chunk-size", "20"])
        assert enc.returncode == 0
        assert "\n---\n" in enc.stdout, "expected multiple chunks joined by separator"
        dec = _run(["decode", "--text"], input_text=enc.stdout)
        assert dec.returncode == 0
        assert secret in dec.stdout


class TestCLIParser:
    """Fast parser-only checks (no model load) for global-flag placement."""

    def test_global_flag_after_subcommand(self) -> None:
        from mostegollm.cli import _build_parser

        ns = _build_parser().parse_args(["encode", "hi", "--quiet", "--model", "X"])
        assert getattr(ns, "quiet", False) is True
        assert getattr(ns, "model", None) == "X"

    def test_global_flag_before_subcommand(self) -> None:
        from mostegollm.cli import _build_parser

        ns = _build_parser().parse_args(["--quiet", "--model", "X", "encode", "hi"])
        assert getattr(ns, "quiet", False) is True
        assert getattr(ns, "model", None) == "X"


class TestCLIModels:
    def test_models_list(self) -> None:
        result = _run(["models"])
        assert result.returncode == 0
        assert "SmolLM" in result.stdout


class TestCLITopics:
    def test_topics_subcommand_lists_topics(self) -> None:
        result = _run(["topics"])
        assert result.returncode == 0
        assert "cooking" in result.stdout
        assert "science" in result.stdout

    def test_encode_with_topic_roundtrips(self) -> None:
        enc = _run(["encode", "--topic", "travel", "hi there"])
        assert enc.returncode == 0
        cover = enc.stdout
        dec = _run(["decode", "--text"], input_text=cover)
        assert dec.returncode == 0
        assert dec.stdout.strip() == "hi there"

    def test_encode_unknown_topic_errors(self) -> None:
        result = _run(["encode", "--topic", "nonsense", "hi"])
        assert result.returncode == 1
        assert "topic" in result.stderr.lower()
