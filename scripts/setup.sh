#!/bin/bash
# 环境安装
set -e
cd "$(dirname "$0")/.."
echo ">>> pip install"
pip install -r requirements.txt
echo ">>> npm install"
npm install
echo ">>> .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example - please edit."
else
  echo ".env exists"
fi
echo ">>> mkdirs"
mkdir -p data/raw data/processed data/models
echo "Done."
