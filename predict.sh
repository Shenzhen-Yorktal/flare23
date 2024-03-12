#!/bin/bash
set -e
echo "begin..."
cd /nnUNet/nnunet
python3 run_inference.py -i /workspace/inputs -o /workspace/outputs
echo "done!"
