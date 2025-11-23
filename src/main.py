"""Small CLI and core functions for the starter project."""

from __future__ import annotations

import argparse
from typing import Iterable


def greet(name: str) -> str:
    return f"Hello, {name}!"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simple starter CLI for AgisFlowAI")
    parser.add_argument("--name", "-n", default="World", help="Name to greet")
    args = parser.parse_args(argv)
    print(greet(args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
