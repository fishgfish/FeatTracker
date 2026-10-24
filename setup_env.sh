#!/usr/bin/env bash
# Create the FeatTracker environment.
#
# Usage (run from the repository root):
#   bash setup_env.sh conda    # create the conda environment from environment.yml
#   bash setup_env.sh uv       # create a .venv with uv and install dependencies
#   bash setup_env.sh pip      # install into the currently active environment
set -euo pipefail

MODE=${1:-conda}

case "${MODE}" in
    conda)
        if ! command -v conda >/dev/null 2>&1; then
            echo "conda was not found. Install Miniconda first or use 'bash setup_env.sh uv'."
            exit 1
        fi
        if conda env list | awk '{print $1}' | grep -qx feattracker; then
            echo "Updating the existing 'feattracker' environment."
            conda env update -f environment.yml --prune
        else
            conda env create -f environment.yml
        fi
        echo ""
        echo "Done. Activate the environment with:"
        echo "    conda activate feattracker"
        ;;

    uv)
        if ! command -v uv >/dev/null 2>&1; then
            echo "uv was not found. Install it with:"
            echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
        uv venv --python 3.8
        # shellcheck disable=SC1091
        source .venv/bin/activate
        uv sync
        echo ""
        echo "Done. Activate the environment with:"
        echo "    source .venv/bin/activate"
        ;;

    pip)
        bash install.sh
        ;;

    *)
        echo "Unknown mode '${MODE}'. Choose one of: conda, uv, pip."
        exit 1
        ;;
esac
