#!/usr/bin/env bash
# Install PyTorch 1.10.0 and the remaining Python dependencies.
#
# The script is intentionally environment-agnostic: activate a conda
# environment or a uv virtual environment first, then run
#   bash install.sh
#
# Optional environment variables:
#   PYTHON          Python interpreter to use (default: python)
#   CUDA_TAG        PyTorch build tag: cu113 (default), cu111, cpu
#   TORCH_VERSION   torch version (default: 1.10.0)
#   TVERSION        torchvision version (default: 0.11.0)
set -euo pipefail

PYTHON=${PYTHON:-python}
CUDA_TAG=${CUDA_TAG:-cu113}
TORCH_VERSION=${TORCH_VERSION:-1.10.0}
TVERSION=${TVERSION:-0.11.0}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/${CUDA_TAG}}

echo "****************** Installing PyTorch ******************"
if [ "${CUDA_TAG}" = "cpu" ]; then
    ${PYTHON} -m pip install "torch==${TORCH_VERSION}" "torchvision==${TVERSION}" \
        --extra-index-url "${TORCH_INDEX}"
else
    ${PYTHON} -m pip install "torch==${TORCH_VERSION}+${CUDA_TAG}" "torchvision==${TVERSION}+${CUDA_TAG}" \
        --extra-index-url "${TORCH_INDEX}"
fi

echo ""
echo "****************** Installing requirements ******************"
${PYTHON} -m pip install --upgrade pip
${PYTHON} -m pip install -r requirements.txt

echo ""
echo "****************** Installation complete ******************"
