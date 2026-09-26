#!/bin/sh
set -eu

# Resolve the experiment, framework, and binary recorder source trees.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR/../Perception_Utility/src:$SCRIPT_DIR/../Binary_Data_Log/py_src${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCRIPT_DIR"
# Launch the experiment with the selected Python environment.
exec "${PYTHON_BIN:-python3}" src/train.py "$@"
