"""Base class for file-specific repair plugins."""

from NOREC4DNA.semi_automatic_reconstruction_toolkit import SemiAutomaticReconstructionToolkit


class FileSpecificRepair:
    """
    Base class for file-specific repair plugins.

    All repair plugins must inherit from this class and implement
    the repair() and is_compatible() methods.
    """

    def __init__(
        self,
        semi_automatic_solver: SemiAutomaticReconstructionToolkit = None,
        chunk_tag=None,
        *args,
        **kwargs,
    ):
        """
        Initialize FileSpecificRepair plugin.

        Args:
            semi_automatic_solver: Semi-automatic reconstruction toolkit instance
            chunk_tag: Chunk tag list
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.semi_automatic_solver = semi_automatic_solver
        self.gepp = self.semi_automatic_solver.decoder.GEPP
        self.use_header_chunk = self.semi_automatic_solver.decoder.use_headerchunk
        self.chunk_tag = chunk_tag
        self.active = False

    def on_load(self, *args, **kwargs):
        """
        Activate the plugin when loaded.

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.active = True

    def repair(self, *args, **kwargs):
        """
        Perform repair operation. Must be implemented by subclasses.

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("Please use a valid subclass of FileSpecificRepair")

    def get_ui_elements(self):
        """
        Get UI elements for the plugin.

        Returns:
            Empty list by default, subclasses should override
        """
        return []

    def is_compatible(self, *args, **kwargs):
        """
        Check if plugin is compatible with current file type.

        Must be implemented by subclasses.

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("Please use a valid subclass of FileSpecificRepair")

    def update_chunk_tag(self, chunk_tag):
        """
        Update chunk tags from external source.

        Args:
            chunk_tag: New chunk tag list
        """
        self.chunk_tag = chunk_tag

    def update_gepp(self, gepp):
        """
        Update GEPP matrix from external source.

        Args:
            gepp: New GEPP instance
        """
        # update gepp.b
        pass

    def update_canvas(self, canvas_data):
        """
        Handle canvas data updates. Base implementation for image-based plugins.

        Args:
            canvas_data: Canvas data from Dash canvas component
        """
        # base for all image-based plugins on canvas updates such as reload image or uploading of a repaired image
        pass
