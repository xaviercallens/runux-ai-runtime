# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# scikit-runux: Scikit-Learn Biomimetic Extension Package
# ========================================================

import os
import sys

# Dynamic router: check if private implementation is available locally
try:
    from .scikit_runux_ext import RunuxClassifier
    PRIVATE_MODE = True
except ImportError:
    from .scikit_runux_ext_stub import RunuxClassifier
    PRIVATE_MODE = False

__all__ = ["RunuxClassifier", "PRIVATE_MODE"]
