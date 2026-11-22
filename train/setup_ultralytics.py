"""Prepare the pinned Ultralytics checkout and this project's custom layers."""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "vendor/ultralytics"
COMMIT = "6e43d1e1e5db72afbf686dee6745669bcb124b0a"  # Ultralytics v8.3.0
PATCH = ROOT / "train/ultralytics.patch"


def git(*args, **kwargs):
    return subprocess.run(["git", "-C", str(SOURCE), *args], **kwargs)


def main():
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", "v8.3.0",
             "https://github.com/ultralytics/ultralytics.git", str(SOURCE)],
            check=True,
        )

    actual = git("rev-parse", "HEAD", capture_output=True, text=True, check=True).stdout.strip()
    if actual != COMMIT:
        raise RuntimeError(f"Expected Ultralytics commit {COMMIT}, found {actual} in {SOURCE}")

    # A reverse check succeeds when our patch has already been applied.
    applied = git("apply", "--reverse", "--check", str(PATCH), capture_output=True).returncode == 0
    if not applied:
        git("apply", "--check", str(PATCH), check=True)
        git("apply", str(PATCH), check=True)

    modules = SOURCE / "ultralytics/nn/modules"
    for path in (ROOT / "train/custom_modules").glob("*.py"):
        shutil.copyfile(path, modules / path.name)
    print(f"Prepared Ultralytics v8.3.0 with custom layers in {SOURCE}")


if __name__ == "__main__":
    main()
