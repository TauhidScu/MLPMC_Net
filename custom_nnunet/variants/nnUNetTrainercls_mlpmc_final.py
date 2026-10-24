# -*- coding: utf-8 -*-
"""
MLPMC Final Trainer registered into nnU-Net variants.
"""

import os
import sys
from pathlib import Path

# Placeholder for installed repository path (injected during installation if needed)
_MLPMC_INSTALLED_ROOT = None

# Discover MLPMC repository root:
# 1. Check explicit environment variable MLPMC_ROOT
# 2. Check injected installation root
# 3. Check current working directory
# 4. Check relative path if running within source tree
candidates = [
    os.environ.get("MLPMC_ROOT"),
    _MLPMC_INSTALLED_ROOT,
    str(Path.cwd()),
    str(Path(__file__).resolve().parent.parent.parent),
]

for candidate in candidates:
    if candidate and os.path.isfile(os.path.join(candidate, "training", "trainer_mlpmc.py")):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        break

from training.trainer_mlpmc import nnUNetTrainercls_mlpmc_final

__all__ = ['nnUNetTrainercls_mlpmc_final']

