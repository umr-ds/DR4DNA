#!/usr/bin/env python3
import os
from setuptools import setup

from pathlib import Path

thisDir = Path(__file__).parent

formatsPath = thisDir / "repair_algorithms"
kaitaiSetuptoolsCfg = {
    "formats": {
        "zip_data.py": {
            "path": "repair_algorithms/zip.ksy",
        }
    },
    "outputDir": thisDir / "repair_algorithms",
    "inputDir": formatsPath
}

setup(use_scm_version=True, kaitai=kaitaiSetuptoolsCfg, packages=["repair_algorithms", "NOREC4DNA", "api", "assets"])
