#!/bin/bash

[[ -d .venv ]] && rm -rf .venv
[[ -f .python-version ]] && rm -rf pyproject.toml

# プロジェクト初期化 (pyproject.tomlを作成)
uv init --no-readme

# Python 3.14をインストール
uv python install 3.14
uv python pin 3.14

# 仮想環境を明示的に作成
uv venv

# Pytorch + torchvision を CUDA 13.0 対応版でインストール
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 動作確認
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
