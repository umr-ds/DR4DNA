#!/usr/bin/env python
"""Debug wrapper to catch SIGSEGV and print stack trace."""
import faulthandler
import sys
import os

# Enable fault handler BEFORE any imports
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=False, exit=True)

# Now import and run
from app import _main_entry

if __name__ == "__main__":
    sys.argv = ['debug_wrapper.py'] + sys.argv[1:]
    try:
        _main_entry()
    except Exception as e:
        print(f"Python exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        faulthandler.dump_traceback()
