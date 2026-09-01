"""Verify Python, dependencies, CUDA, Ollama API, and the selected model."""

from __future__ import annotations

import platform
import sys

import numpy
import pandas
import requests
import sklearn
import torch


def main() -> None:
    print("CSE445 LOCAL AGENT SETUP CHECK")
    print("=" * 48)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"Platform:     {platform.platform()}")
    print(f"NumPy:        {numpy.__version__}")
    print(f"Pandas:       {pandas.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA ready:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(0)}")

    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
        response.raise_for_status()
        models = [entry["name"] for entry in response.json().get("models", [])]
        print("Ollama API:   reachable")
        print(f"Models:       {', '.join(models) if models else 'none installed'}")
        if not any(name.startswith("llama3.2:3b") for name in models):
            print("WARNING: run 'ollama pull llama3.2:3b'")
    except requests.RequestException as exc:
        print(f"Ollama API:   FAILED ({exc})")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

