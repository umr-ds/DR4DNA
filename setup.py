#!/usr/bin/env python3
import os
from setuptools import setup, find_packages
from pathlib import Path
from setuptools.command.install import install
import subprocess


class CustomInstall(install):
    def run(self):
        install.run(self)  # Run the default installation first
        subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])


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

setup(name='DR4DNA', version='1.0.0b', use_scm_version=True, kaitai=kaitaiSetuptoolsCfg,
      packages=find_packages(),
      setup_requires=[
          'pip',
      ],
      cmdclass={
          'install': CustomInstall,
      },
      )
