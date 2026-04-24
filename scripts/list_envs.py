#!/usr/bin/env python3
"""List MJLab tasks after registering this package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import go2_pendulum_mjlab  # noqa: E402,F401
from mjlab.tasks.registry import list_tasks


def main() -> None:
  for task in list_tasks():
    print(task)


if __name__ == "__main__":
  main()
