#!/usr/bin/env python3
"""
Legacy checkpoint compatibility shim.

Older checkpoints in this workspace were serialized with ``config.Config`` as
the module path. Keeping this top-level alias lets ``torch.load`` restore them
without custom unpickling hooks.
"""

from src.config import Config, get_default_config

__all__ = ["Config", "get_default_config"]
