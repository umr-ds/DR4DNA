"""Repair algorithms package for DR4DNA.

This package contains various file repair plugins for DNA-encoded data storage.
Each plugin handles specific file types and provides methods to identify and
correct errors in the encoded data.

Available repair plugins:
- BMPFileRepair: BMP image file repair
- LangaugeToolTextRepair: Text file repair using LanguageTool
- MetadataRepair: Metadata sequence repair
- MissingRowRepair: Missing row/packet repair
- RandomShuffleRepair: Shuffle-based error detection
- UploadRepair: Manual upload-based repair
- ZipFileRepair: ZIP archive repair
"""

import glob
from os.path import basename, dirname, isfile, join

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")]
