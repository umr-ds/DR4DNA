# DR4DNA Architecture Documentation

## Overview

DR4DNA (Data Recovery for DNA Storage) is a web-based application for recovering corrupted data encoded using fountain codes, specifically designed for DNA storage systems.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                    (Dash/Plotly Web UI)                          │
├─────────────────────────────────────────────────────────────────┤
│                        Callback Handlers                         │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────┐ │
│  │ Plugin Handler   │ │ Button Handler   │ │ Repair Handler  │ │
│  └──────────────────┘ └──────────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                         Service Layer                            │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────┐ │
│  │ Decoder Service  │ │ Repair Service   │ │ Plugin Service  │ │
│  └──────────────────┘ └──────────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      Core Components                             │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────┐ │
│  │   State Mgmt     │ │    Logging       │ │    Config       │ │
│  └──────────────────┘ └──────────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      External Dependencies                       │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────┐ │
│  │  NOREC4DNA       │ │  Plugin System   │ │  KaitaiStruct   │ │
│  └──────────────────┘ └──────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. User Interface Layer

**File**: `layout.py`, `app.py`

The UI is built using Dash/Plotly and provides:
- Hex/ASCII view of decoded data
- Interactive chunk tagging
- Plugin UI elements
- Canvas for image-based repair
- Kaitai Struct viewer

**Key Components**:
- `gen_app_layout()`: Generates the main application layout
- Interactive rows for each chunk
- Plugin container for dynamic UI elements
- Canvas for image manipulation

### 2. Callback Handler Layer

**File**: `callback_handlers.py`

Handles all user interactions through specialized handlers:

#### PluginCallbackHandler
- Manages plugin I/O operations
- Processes plugin callback responses
- Handles plugin errors gracefully

#### ButtonCallbackHandler
- Handles main UI button clicks
- Manages chunk tagging operations
- Coordinates view updates

#### RepairCallbackHandler
- Manages repair-related inputs
- Handles hex/text repair editing
- Coordinates repair operations

### 3. Service Layer (New)

**Directory**: `services/`

Provides clean separation between UI and business logic:

#### DecoderService (`decoder_service.py`)
- Encapsulates all decoder operations
- Manages GEPP (Gaussian Elimination with Partial Pivoting)
- Provides file type detection
- Handles save/load operations

#### RepairService (`repair_service.py`)
- Manages repair operations
- Provides error analysis
- Validates repair operations
- Tracks repair statistics

#### PluginService (`plugin_service.py`)
- Manages plugin lifecycle
- Handles plugin discovery and loading
- Provides error isolation
- Manages plugin state updates

### 4. State Management (New)

**File**: `state.py`

Centralized state management replacing global variables:

```python
class AppState:
    - chunk_tag: List[int]       # Status of each chunk
    - column_tag: List[int]      # Column correctness
    - common_packets: List[bool] # Potentially corrupt packets
    - content_updated: bool      # Content change flag
    - show_canvas: bool          # Canvas visibility
    - multi_error_packets_mode: bool
    - colorblind_mode: bool
```

**Features**:
- Thread-safe access with RLock
- Change notification callbacks
- Validation on state changes
- Backward compatible with global variables

### 5. Configuration Management (New)

**File**: `config_manager.py`

Centralized configuration with environment variable support:

```python
class AppConfig:
    - server: ServerConfig
    - logging: LoggingConfig
    - plugins: PluginConfig
    - performance: PerformanceConfig
```

**Features**:
- Environment variable support (DR4DNA_*)
- Configuration validation
- Type-safe access
- Default values

### 6. Logging Infrastructure (New)

**File**: `logging_config.py`

Structured logging replacing print statements:

**Features**:
- Console and file logging
- Log rotation
- Color-coded console output
- JSON format support for production
- Function call logging decorator

**Usage**:
```python
from logging_config import get_logger

logger = get_logger(__name__)
logger.info("Operation completed")
logger.error(f"Error: {e}")
```

### 7. Exception Handling (New)

**File**: `exceptions.py`

Comprehensive exception hierarchy:

```
DR4DNAException
├── ConfigurationException
├── PluginException
│   ├── PluginLoadError
│   ├── PluginExecutionError
│   └── PluginCompatibilityError
├── DecoderException
│   └── DecodeError
├── RepairException
│   └── RepairValidationError
├── StateException
│   └── StateValidationError
├── FileIOException
└── DataIntegrityException
```

## Data Flow

### 1. Application Initialization

```
1. Parse command line arguments
2. Load configuration (.ini file)
3. Initialize decoder from NOREC4DNA
4. Create AppState
5. Initialize services (DecoderService, RepairService, PluginService)
6. Discover and load plugins
7. Generate UI layout
8. Start Dash server
```

### 2. Chunk Tagging Flow

```
1. User clicks on chunk row
2. change_button_style() callback triggered
3. Update chunk_tag state
4. propagate_gepp_update() called
5. Plugins notified of state change
6. Recalculate view with new colors
```

### 3. Plugin Repair Flow

```
1. User clicks plugin repair button
2. PluginCallbackHandler.handle_plugin_io() called
3. Plugin callback executed with error isolation
4. Plugin returns result dict:
   - chunk_tag updates
   - column_tag updates
   - update_b flag
   - refresh_view flag
5. If update_b: propagate_gepp_update()
6. If refresh_view: recalculate_view()
```

### 4. Manual Repair Flow

```
1. User opens repair window
2. repair_callback() populates hex/text fields
3. User edits content
4. repair_chunks() validates and applies repair
5. propagate_gepp_update() updates all plugins
6. View recalculated
```

## Plugin Architecture

### Plugin Lifecycle

```
1. Discovery
   └─> Scan repair_algorithms/ for .py files
   
2. Registration
   └─> PluginManager.register_plugin(PluginClass)
   
3. Instantiation
   └─> PluginClass(semi_automatic_solver, chunk_tag)
   
4. Compatibility Check
   └─> plugin.is_compatible(file_type)
   
5. Activation
   └─> plugin.on_load()
   
6. Operation
   └─> plugin callbacks executed
   
7. State Updates
   └─> plugin.update_gepp()
   └─> plugin.update_chunk_tag()
```

### Plugin Interface

```python
class FileSpecificRepair:
    def __init__(self, semi_automatic_solver, chunk_tag)
    def is_compatible(self, meta_info) -> bool
    def on_load(self)
    def get_ui_elements(self) -> Dict
    def repair(self) -> Dict
    def update_gepp(self, gepp)
    def update_chunk_tag(self, chunk_tag)
    def update_canvas(self, canvas_data)
```

## Key Algorithms

### 1. Corrupt Packet Detection

Based on fountain code properties:
- Each chunk is a linear combination of original packets
- By comparing different solutions (permutations), corrupt packets can be identified
- Common packets between invalid rows are potential corrupt packets

### 2. Automatic Repair

When rank(A) ≠ rank([A|b]):
- Error is linearly dependent
- Can be detected and corrected
- Uses Gaussian elimination with different permutations

### 3. Manual Repair

When automatic repair not possible:
- User tags chunks as corrupt/correct
- System narrows down possible corrupt packets
- User can manually correct chunk content

### 4. Plugin-Based Repair

File-type-specific repair:
- Parse file structure (KaitaiStruct)
- Identify structural inconsistencies
- Apply domain-specific repair logic

## Security Considerations

1. **Input Validation**: All user inputs validated before processing
2. **Plugin Isolation**: Plugin errors caught and isolated
3. **File Upload**: Uploaded files scanned and validated
4. **Path Traversal**: File paths sanitized

## Performance Considerations

1. **Large Files**: Streaming processing where possible
2. **Matrix Operations**: Optimized numpy operations
3. **Caching**: State caching for expensive operations
4. **Lazy Loading**: Plugins loaded on demand

## Testing Strategy

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **Plugin Tests**: Test plugin interface compliance
4. **Performance Tests**: Benchmark critical operations

## Directory Structure

```
DR4DNA/
├── app.py                  # Main application entry point
├── app_callbacks.py        # Dash callback definitions
├── callback_handlers.py    # Callback handler classes
├── layout.py               # UI layout generation
├── config.py               # Legacy configuration
├── config_manager.py       # New configuration system
├── state.py                # State management
├── exceptions.py           # Exception hierarchy
├── logging_config.py       # Logging configuration
├── CustomExceptions.py     # Legacy exceptions
├── MultiVersionDecoder.py  # Version-aware decoder
├── services/
│   ├── __init__.py
│   ├── decoder_service.py
│   ├── repair_service.py
│   └── plugin_service.py
├── repair_algorithms/      # Plugin directory
│   ├── FileSpecificRepair.py
│   ├── PluginManager.py
│   ├── UploadRepair.py
│   ├── BMPFileRepair.py
│   ├── RandomShuffleRepair.py
│   └── ...
├── docs/                   # Documentation
│   ├── PLUGIN_DEVELOPMENT_GUIDE.md
│   └── ARCHITECTURE.md
└── tests/                  # Test suite
```

## Dependencies

### Core Dependencies
- **Dash/Plotly**: Web framework
- **NOREC4DNA**: Fountain code decoder
- **NumPy**: Matrix operations
- **KaitaiStruct**: File format parsing

### Plugin Dependencies
- **Pillow**: Image processing
- **language-tool-python**: Text repair
- **crcmod**: CRC calculations

## Future Improvements

1. **Microservices Architecture**: Separate decoder into standalone service
2. **Database Backend**: Persist session state
3. **REST API**: Enable programmatic access
4. **WebSocket Support**: Real-time updates
5. **Plugin Marketplace**: Centralized plugin repository
