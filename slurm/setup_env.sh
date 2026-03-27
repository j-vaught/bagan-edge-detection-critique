#!/bin/bash
# Environment setup for bagan-edge-detection-critique on Theia HPC

module load python3/anaconda/3.12
module load cuda/12.1

pip install --user opencv-python-headless 2>/dev/null

python3 -c "
import numpy; print('numpy:', numpy.__version__)
import scipy; print('scipy:', scipy.__version__)
import matplotlib; print('matplotlib:', matplotlib.__version__)
import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
try:
    import cv2; print('opencv:', cv2.__version__)
except ImportError:
    print('opencv: not available (using scipy fallbacks)')
"
