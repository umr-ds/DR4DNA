# -*- coding: utf-8 -*-
"""Base class for file-specific repair plugins."""

import typing
from typing import Any, Dict, List, Optional

from NOREC4DNA.norec4dna.GEPP import GEPP
from NOREC4DNA.semi_automatic_reconstruction_toolkit import SemiAutomaticReconstructionToolkit


class FileSpecificRepair:
    """
    Base class for file-specific repair plugins.

    All repair plugins must inherit from this class and implement
    the repair() and is_compatible() methods.

    Attributes:
        semi_automatic_solver: The semi-automatic reconstruction toolkit instance
        gepp: GEPP matrix instance from the solver
        use_header_chunk: Whether to use header chunk in decoding
        chunk_tag: List of chunk tags (0=unknown, 1=invalid, 2=valid, 3=undecoded)
        active: Whether the plugin is currently active
    """

    def __init__(
        self,
        semi_automatic_solver: Optional[SemiAutomaticReconstructionToolkit] = None,
        chunk_tag: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize FileSpecificRepair plugin.

        Args:
            semi_automatic_solver: Semi-automatic reconstruction toolkit instance.
                Must be provided by subclasses.
            chunk_tag: Chunk tag list. If None, will be initialized by subclass.
        """
        if semi_automatic_solver is None:
            raise ValueError("semi_automatic_solver must be provided")

        self.semi_automatic_solver: SemiAutomaticReconstructionToolkit = semi_automatic_solver
        self.gepp: GEPP = self.semi_automatic_solver.decoder.GEPP
        self.use_header_chunk: bool = self.semi_automatic_solver.decoder.use_headerchunk
        self.chunk_tag: Optional[List[int]] = chunk_tag
        self.active: bool = False

    def on_load(self) -> None:
        """
        Activate the plugin when loaded.

        Subclasses can override this method to perform additional initialization.
        """
        self.active = True

    def repair(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Perform repair operation. Must be implemented by subclasses.

        Args:
            **kwargs: Subclass-specific keyword arguments. Common arguments include:
                - c_ctx: Dash callback context
                - chunk_tag: Current chunk tag list

        Returns:
            Dictionary with repair results. Common keys:
                - 'info': Status message string
                - 'update_b': bool, whether to update GEPP b matrix
                - 'refresh_view': bool, whether to refresh the view
                - 'repair_variations': dict, repair variations data

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("Please use a valid subclass of FileSpecificRepair")

    def get_ui_elements(self) -> Dict[str, Any]:
        """
        Get UI elements for the plugin.

        Returns:
            Dictionary mapping element IDs to UI element configurations.
            Each configuration should have:
                - 'type': str, element type ('button', 'input', 'select', etc.)
                - 'text' or 'default': str, element label or default value
                - 'callback': callable, function to call on interaction (optional)
                - 'updates_b': bool, whether callback updates GEPP b matrix (optional)
        """
        return {}

    def is_compatible(self, meta_info: Optional[str] = None) -> bool:
        """
        Check if plugin is compatible with current file type.

        Must be implemented by subclasses.

        Args:
            meta_info: Optional metadata information to check compatibility against.
                If None, checks against the current solver's file type.

        Returns:
            True if compatible, False otherwise

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("Please use a valid subclass of FileSpecificRepair")

    def update_chunk_tag(self, chunk_tag: List[int]) -> None:
        """
        Update chunk tags from external source.

        Args:
            chunk_tag: New chunk tag list (0=unknown, 1=invalid, 2=valid, 3=undecoded)
        """
        self.chunk_tag = chunk_tag

    def update_gepp(self, gepp: GEPP) -> None:
        """
        Update GEPP matrix from external source.

        Args:
            gepp: New GEPP instance
        """
        # update gepp.b
        pass

    def update_canvas(self, canvas_data: Any) -> None:
        """
        Handle canvas data updates. Base implementation for image-based plugins.

        Args:
            canvas_data: Canvas data from Dash canvas component.
                Typically a dict with 'image_content', 'json_data', etc.
        """
        # base for all image-based plugins on canvas updates such as reload image or uploading of a repaired image
        pass
