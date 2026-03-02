#!/usr/bin/env python3
"""
Setup script for DR4DNA package installation.

Handles package installation with automatic dependency resolution and
Kaitai Struct format compilation.

Usage:
    pip install .
    python setup.py install
"""
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.install import install


class CustomInstall(install):
    """
    Custom installation command that ensures dependencies are installed.

    Extends the standard install command to automatically install
    requirements from requirements.txt after the main installation.
    """

    def run(self):
        """
        Run the installation process.

        First runs the standard installation, then installs dependencies
        from requirements.txt using pip.
        """
        install.run(self)  # Run the default installation first
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])


thisDir = Path(__file__).parent

formatsPath = thisDir / "repair_algorithms"
kaitaiSetuptoolsCfg = {
    "formats": {
        "zip_data.py": {
            "path": "repair_algorithms/zip.ksy",
        }
    },
    "outputDir": thisDir / "repair_algorithms",
    "inputDir": formatsPath,
}

setup(
    name="DR4DNA",
    version="1.0.0b",
    use_scm_version=True,
    kaitai=kaitaiSetuptoolsCfg,
    packages=find_packages(),
    setup_requires=[
        "pip",
    ],
    cmdclass={
        "install": CustomInstall,
    },
)
