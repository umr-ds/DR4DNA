# DR4DNA Plugin Development Guide

## Overview

DR4DNA uses a plugin system to allow users to define custom repair methods for different file types. Plugins extend the core functionality by providing file-type-specific repair algorithms and user interface elements.

This guide covers:
- Plugin architecture
- Creating custom plugins
- Plugin lifecycle
- Best practices
- Examples

## Plugin Architecture

### Base Class

All plugins must inherit from `FileSpecificRepair` class located in `repair_algorithms/FileSpecificRepair.py`.

```python
from repair_algorithms.FileSpecificRepair import FileSpecificRepair

class MyCustomRepair(FileSpecificRepair):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize plugin-specific attributes

    def is_compatible(self, meta_info):
        # Check if this plugin should be active for the given file type
        return meta_info == "MyFileType"

    def get_ui_elements(self):
        # Define UI elements for this plugin
        return {
            "btn-repair": {
                "type": "button",
                "text": "Repair",
                "callback": self.repair,
                "updates_b": False
            }
        }

    def repair(self, *args, **kwargs):
        # Implement repair logic
        return {"update_b": True, "refresh_view": True}
```

### Plugin Lifecycle

1. **Discovery**: Plugins are automatically discovered from the `repair_algorithms/` directory
2. **Instantiation**: Plugin instances are created with the semi-automatic solver
3. **Compatibility Check**: `is_compatible()` is called to determine if plugin should be active
4. **Activation**: `on_load()` is called when plugin is activated
5. **Operation**: Plugin callbacks are executed based on user interactions
6. **Updates**: `update_gepp()` and `update_chunk_tag()` are called when state changes

## Creating a Custom Plugin

### Step 1: Create Plugin File

Create a new Python file in `repair_algorithms/`:

```python
# repair_algorithms/MyCustomRepair.py
import typing
from repair_algorithms.FileSpecificRepair import FileSpecificRepair
from repair_algorithms.PluginManager import PluginManager


class MyCustomRepair(FileSpecificRepair):
    """
    Custom repair plugin for [file type].

    This plugin provides repair functionality for [description].
    """

    def __init__(self, *args, **kwargs):
        """Initialize the plugin."""
        super().__init__(*args, **kwargs)
        # Plugin-specific initialization
        self.custom_attribute = None

    def is_compatible(self, meta_info: str) -> bool:
        """
        Check if this plugin is compatible with the given file type.

        Args:
            meta_info: File type information from magic detection

        Returns:
            True if plugin should be active for this file type
        """
        # Example: activate for PDF files
        return meta_info == "PDF" or meta_info.lower().endswith(".pdf")

    def on_load(self, *args, **kwargs):
        """Called when plugin is loaded/activated."""
        super().on_load(*args, **kwargs)
        # Additional initialization when plugin becomes active

    def get_ui_elements(self) -> typing.Dict[str, typing.Dict]:
        """
        Define UI elements for this plugin.

        Returns:
            Dictionary of UI element definitions
        """
        return {
            "btn-analyze": {
                "type": "button",
                "text": "Analyze Errors",
                "callback": self.analyze,
                "updates_b": False
            },
            "btn-repair": {
                "type": "button",
                "text": "Automatic Repair",
                "callback": self.repair,
                "updates_b": True
            },
            "txt-threshold": {
                "type": "int",
                "text": "Error Threshold",
                "default": 5,
                "callback": self.update_threshold,
                "updates_b": False
            }
        }

    def analyze(self, *args, **kwargs):
        """Analyze errors in the current data."""
        try:
            # Access chunk tag
            chunk_tag = kwargs.get("chunk_tag", self.chunk_tag)

            # Access GEPP matrix
            gepp = self.gepp
            b_matrix = gepp.b  # Data matrix
            A_matrix = gepp.A  # Coefficient matrix

            # Perform analysis
            # ...

            return {
                "info": "Analysis complete",
                "chunk_tag": updated_chunk_tag,
                "refresh_view": True,
                "update_b": False
            }
        except Exception as e:
            return {"info": f"Analysis failed: {str(e)}", "refresh_view": True}

    def repair(self, *args, **kwargs):
        """Perform automatic repair."""
        try:
            # Access state
            chunk_tag = self.chunk_tag

            # Find corrupt chunks
            # ...

            # Perform repair
            # ...

            return {
                "update_b": True,
                "refresh_view": True,
                "chunk_tag": updated_chunk_tag
            }
        except Exception as e:
            return {"info": f"Repair failed: {str(e)}", "refresh_view": True}

    def update_threshold(self, *args, **kwargs):
        """Update error threshold from UI."""
        c_ctx = kwargs.get("c_ctx")
        if c_ctx:
            self.custom_attribute = c_ctx.triggered[0]["value"]
        return {"refresh_view": False, "update_b": False}

    def update_gepp(self, gepp):
        """
        Called when GEPP is updated.

        Args:
            gepp: New GEPP instance
        """
        super().update_gepp(gepp)
        self.gepp = gepp
        # Invalidate cached data
        self.custom_attribute = None

    def update_chunk_tag(self, chunk_tag):
        """
        Called when chunk tag is updated.

        Args:
            chunk_tag: New chunk tag list
        """
        super().update_chunk_tag(chunk_tag)
        self.chunk_tag = chunk_tag


# Register plugin
mgr = PluginManager()
mgr.register_plugin(MyCustomRepair)
```

### Step 2: Implement Required Methods

#### `is_compatible(meta_info)`

Determines if the plugin should be active for a given file type.

```python
def is_compatible(self, meta_info: str) -> bool:
    # Check file type
    if meta_info == "Bitmap":
        return True

    # Check file extension
    if meta_info.lower().endswith(".bmp"):
        return True

    # Always active (for generic plugins)
    return True
```

#### `get_ui_elements()`

Defines the plugin's user interface.

**Supported UI Element Types:**

| Type | Description | Properties |
|------|-------------|------------|
| `button` | Clickable button | `text`, `callback`, `updates_b` |
| `int` | Integer input | `text`, `default`, `callback` |
| `text` | Text input | `text`, `callback` |
| `upload` | File upload | `text`, `callback`, `updates_canvas` |
| `download` | File download | `text`, `callback` |
| `toggle` | Toggle switch | `text`, `on_label`, `off_label`, `callback` |
| `canvas` | Canvas for drawing | `width`, `height` |
| `kaitai_view` | Kaitai Struct viewer | `text`, `callback` |

#### `repair(*args, **kwargs)`

Main repair method. Should return a dictionary with:

- `update_b`: Boolean - whether to update GEPP matrix
- `refresh_view`: Boolean - whether to refresh UI
- `chunk_tag`: Optional - updated chunk tag list
- `column_tag`: Optional - updated column tag list
- `info`: Optional - status message
- `repair`: Optional - repair details for manual repair

### Step 3: Access Core Functionality

#### Accessing Data

```python
# Access chunk tag (status of each chunk)
chunk_tag = self.chunk_tag  # 0=unknown, 1=corrupt, 2=correct, 3=missing

# Access column tag
column_tag = self.column_tag

# Access GEPP matrix
b_matrix = self.gepp.b  # Data matrix (chunks × columns)
A_matrix = self.gepp.A  # Coefficient matrix

# Access semi-automatic solver
solver = self.semi_automatic_solver
```

#### Modifying State

```python
# Update chunk tag
self.chunk_tag[i] = 1  # Mark chunk as corrupt

# Return updated state
return {
    "chunk_tag": self.chunk_tag,
    "update_b": True,
    "refresh_view": True
}
```

#### Performing Manual Repair

```python
# In your repair method
return {
    "update_b": True,
    "repair": {
        "corrected_row": chunk_id,
        "corrected_value": corrected_bytes
    },
    "refresh_view": True
}
```

## Plugin Examples

### Example 1: Simple Text Repair Plugin

```python
# repair_algorithms/SimpleTextRepair.py
import typing
from repair_algorithms.FileSpecificRepair import FileSpecificRepair
from repair_algorithms.PluginManager import PluginManager


class SimpleTextRepair(FileSpecificRepair):
    """Simple text repair plugin."""

    def is_compatible(self, meta_info):
        return "text" in meta_info.lower()

    def get_ui_elements(self):
        return {
            "btn-find-errors": {
                "type": "button",
                "text": "Find Non-Printable Characters",
                "callback": self.find_errors,
                "updates_b": False
            }
        }

    def find_errors(self, *args, **kwargs):
        """Find chunks with non-printable characters."""
        chunk_tag = kwargs.get("chunk_tag", self.chunk_tag)

        for i, chunk in enumerate(self.gepp.b):
            # Check for non-printable characters
            has_non_printable = any(b < 32 or b > 126 for b in chunk)
            if has_non_printable:
                chunk_tag[i] = 1  # Mark as corrupt

        return {
            "chunk_tag": chunk_tag,
            "info": f"Found {sum(chunk_tag)} corrupt chunks",
            "refresh_view": True,
            "update_b": False
        }


mgr = PluginManager()
mgr.register_plugin(SimpleTextRepair)
```

### Example 2: Image Repair Plugin

```python
# repair_algorithms/ImageRepair.py
import numpy as np
import typing
from repair_algorithms.FileSpecificRepair import FileSpecificRepair
from repair_algorithms.PluginManager import PluginManager


class ImageRepair(FileSpecificRepair):
    """Image file repair plugin."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_width = None
        self.image_height = None

    def is_compatible(self, meta_info):
        return meta_info in ["PNG", "JPEG", "Bitmap"]

    def get_ui_elements(self):
        return {
            "btn-reload": {
                "type": "button",
                "text": "Reload Image",
                "callback": self.reload_image,
                "updates_canvas": True
            },
            "btn-repair": {
                "type": "button",
                "text": "Automatic Repair",
                "callback": self.repair,
                "updates_b": True
            }
        }

    def reload_image(self, *args, **kwargs):
        """Reload and analyze image."""
        # Parse image header
        # Extract dimensions
        # Load into canvas

        return {
            "update_b": False,
            "refresh_view": True,
            "updates_canvas": True,
            "canvas_data": image_matrix,
            "width": self.image_width,
            "height": self.image_height
        }

    def repair(self, *args, **kwargs):
        """Perform image-specific repair."""
        # Analyze image structure
        # Find corrupted regions
        # Attempt repair

        return {
            "update_b": True,
            "refresh_view": True,
            "chunk_tag": self.chunk_tag
        }


mgr = PluginManager()
mgr.register_plugin(ImageRepair)
```

## Best Practices

### 1. Error Handling

Always wrap plugin operations in try-except blocks:

```python
def repair(self, *args, **kwargs):
    try:
        # Repair logic
        pass
    except Exception as e:
        return {"info": f"Repair failed: {str(e)}", "refresh_view": True}
```

### 2. State Management

Always return updated state:

```python
return {
    "chunk_tag": self.chunk_tag,
    "column_tag": self.column_tag,
    "update_b": True,
    "refresh_view": True
}
```

### 3. Performance

- Cache expensive computations
- Invalidate cache when `update_gepp()` is called
- Use efficient data structures

### 4. User Feedback

Provide clear status messages:

```python
return {
    "info": f"Successfully repaired {repaired_count} chunks",
    "refresh_view": True
}
```

### 5. Compatibility

Be specific about file type compatibility:

```python
def is_compatible(self, meta_info):
    # Specific is better than generic
    return meta_info == "PNG"  # Better than: return True
```

## Testing Plugins

### Unit Tests

Create tests in `repair_algorithms/tests/`:

```python
# repair_algorithms/tests/test_my_custom_repair.py
import unittest
from unittest.mock import Mock
from repair_algorithms.MyCustomRepair import MyCustomRepair


class TestMyCustomRepair(unittest.TestCase):

    def setUp(self):
        self.mock_solver = Mock()
        self.plugin = MyCustomRepair(semi_automatic_solver=self.mock_solver)

    def test_is_compatible(self):
        self.assertTrue(self.plugin.is_compatible("PDF"))
        self.assertFalse(self.plugin.is_compatible("PNG"))

    def test_get_ui_elements(self):
        ui = self.plugin.get_ui_elements()
        self.assertIn("btn-repair", ui)


if __name__ == '__main__':
    unittest.main()
```

## Debugging

Enable debug logging:

```python
from logging_config import get_logger

logger = get_logger(__name__)

class MyCustomRepair(FileSpecificRepair):
    def repair(self, *args, **kwargs):
        logger.debug("Starting repair")
        try:
            # Logic
            logger.info("Repair successful")
        except Exception as e:
            logger.error(f"Repair failed: {e}")
            raise
```

## Migration from Old Plugin System

Existing plugins continue to work without modification. The new plugin service provides additional features but maintains backward compatibility.

## Support

For questions or issues:
- Check existing plugins for examples
- Review the `FileSpecificRepair` base class
- Open an issue on GitHub
