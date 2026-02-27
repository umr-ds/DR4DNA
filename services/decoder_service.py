# -*- coding: utf-8 -*-
"""
Decoder service for DR4DNA.

This service encapsulates all decoder-related operations and provides
a clean API for the UI layer.
"""

import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from exceptions import DecodeError, DecoderException, FileIOException
from logging_config import get_logger
from NOREC4DNA.norec4dna.GEPP import GEPP
from NOREC4DNA.norec4dna.HeaderChunk import HeaderChunk
from NOREC4DNA.norec4dna.LTDecoder import LTDecoder
from NOREC4DNA.norec4dna.OnlineDecoder import OnlineDecoder
from NOREC4DNA.norec4dna.RU10Decoder import RU10Decoder
from NOREC4DNA.semi_automatic_reconstruction_toolkit import SemiAutomaticReconstructionToolkit
from state import AppState

logger = get_logger(__name__)


class DecoderService:
    """
    Service class for decoder operations.

    This class encapsulates all decoder-related functionality and provides
    a clean API for the UI layer. It manages the decoder state and provides
    methods for common operations.

    Attributes:
        semi_automatic_solver: The semi-automatic reconstruction toolkit
        state: Application state instance
    """

    def __init__(
        self,
        semi_automatic_solver: SemiAutomaticReconstructionToolkit,
        state: Optional[AppState] = None,
    ):
        """
        Initialize the decoder service.

        Args:
            semi_automatic_solver: Initialized semi-automatic solver
            state: Application state instance (uses global if None)
        """
        from state import get_app_state

        self.solver = semi_automatic_solver
        self.state = state or get_app_state()
        self._decoder = semi_automatic_solver.decoder
        self._gepp = self._decoder.GEPP

        logger.info(f"DecoderService initialized with {self._decoder.number_of_chunks} chunks")

    @property
    def decoder(self) -> typing.Union[RU10Decoder, LTDecoder, OnlineDecoder]:
        """Get the underlying decoder instance."""
        return self._decoder

    @property
    def gepp(self) -> GEPP:
        """Get the GEPP instance."""
        return self._gepp

    @property
    def number_of_chunks(self) -> int:
        """Get the number of chunks."""
        return self._decoder.number_of_chunks

    @property
    def header_chunk(self) -> Optional[HeaderChunk]:
        """Get the header chunk if available."""
        return self.solver.headerChunk

    def calculate_rank_a(self) -> int:
        """
        Calculate the rank of matrix A.

        Returns:
            Rank of matrix A
        """
        try:
            rank = self.solver.calculate_rank_A()
            logger.debug(f"Calculated rank(A) = {rank}")
            return rank
        except Exception as e:
            logger.error(f"Error calculating rank(A): {e}")
            raise DecoderException(f"Failed to calculate rank(A): {e}")

    def calculate_rank_augmented_matrix(self) -> int:
        """
        Calculate the rank of the augmented matrix [A|b].

        Returns:
            Rank of augmented matrix
        """
        try:
            rank = self.solver.calculate_rank_augmented_matrix()
            logger.debug(f"Calculated rank(A|b) = {rank}")
            return rank
        except Exception as e:
            logger.error(f"Error calculating rank(A|b): {e}")
            raise DecoderException(f"Failed to calculate rank(A|b): {e}")

    def analyze_system_solvability(self) -> Dict[str, Any]:
        """
        Analyze the linear equation system solvability.

        Returns:
            Dictionary with analysis results:
            - rank_a: Rank of matrix A
            - rank_augmented: Rank of augmented matrix
            - num_chunks: Number of chunks
            - solvable: Whether system is solvable
            - error_detectable: Whether errors are detectable
            - message: Human-readable status message
        """
        rank_a = self.calculate_rank_a()
        rank_augmented = self.calculate_rank_augmented_matrix()
        num_chunks = self.number_of_chunks

        solvable = rank_augmented >= num_chunks
        error_detectable = rank_a != rank_augmented

        if rank_augmented < num_chunks:
            message = (
                f"Augmented rank ({rank_augmented}) < number of chunks ({num_chunks}), "
                f"but partial recovery might be possible."
            )
        elif error_detectable:
            message = "Erroneous packet detectable!"
        else:
            message = (
                f"LES seems solvable. Either all packets are correct or the corrupt "
                f"packet is not linear dependent in the LES."
            )

        result = {
            "rank_a": rank_a,
            "rank_augmented": rank_augmented,
            "num_chunks": num_chunks,
            "solvable": solvable,
            "error_detectable": error_detectable,
            "message": message,
        }

        logger.info(f"System analysis: {message}")
        return result

    def get_common_packets(
        self, invalid_rows: List[int], valid_rows: List[int], multi_error_mode: bool = False
    ) -> List[bool]:
        """
        Get packets common to invalid rows.

        Args:
            invalid_rows: List of invalid row indices
            valid_rows: List of valid row indices
            multi_error_mode: Whether to use multi-error mode

        Returns:
            List of booleans indicating potentially corrupt packets
        """
        try:
            packets = self._gepp.get_common_packets(invalid_rows, valid_rows, multi_error_mode)
            logger.debug(f"Found {sum(packets)} potentially corrupt packets")
            return packets
        except Exception as e:
            logger.error(f"Error getting common packets: {e}")
            raise DecoderException(f"Failed to get common packets: {e}")

    def calculate_unused_packets(self) -> List[bool]:
        """
        Calculate which packets were not used.

        Returns:
            List of booleans indicating unused packets
        """
        try:
            return self.solver.calculate_unused_packets()
        except Exception as e:
            logger.error(f"Error calculating unused packets: {e}")
            raise DecoderException(f"Failed to calculate unused packets: {e}")

    def get_possible_invalid_chunks(self, common_packets: List[bool]) -> List[bool]:
        """
        Get chunks that could be invalid based on common packets.

        Args:
            common_packets: List of potentially corrupt packets

        Returns:
            List of booleans indicating possibly invalid chunks
        """
        try:
            return self.solver.get_possible_invalid_chunks_from_common_packets(common_packets)
        except Exception as e:
            logger.error(f"Error getting possible invalid chunks: {e}")
            raise DecoderException(f"Failed to get possible invalid chunks: {e}")

    def manual_repair(self, chunk_id: int, packet_id: int, hex_value: bytes):
        """
        Perform manual repair on a chunk.

        Args:
            chunk_id: ID of chunk to repair
            packet_id: ID of packet to use for repair
            hex_value: Corrected content as bytes
        """
        try:
            if not 0 <= chunk_id < self.number_of_chunks:
                raise DecodeError(f"Invalid chunk ID: {chunk_id}", chunk_id=chunk_id)

            self.solver.manual_repair(chunk_id, packet_id, hex_value)
            self.state.mark_content_updated()
            logger.info(f"Manual repair performed on chunk {chunk_id}")
        except Exception as e:
            logger.error(f"Error during manual repair: {e}")
            raise

    def save_decoded_file(
        self,
        output_path: Optional[Path] = None,
        return_file_name: bool = True,
        print_to_output: bool = False,
    ) -> str:
        """
        Save the decoded file.

        Args:
            output_path: Optional output path
            return_file_name: Whether to return the filename
            print_to_output: Whether to print to output

        Returns:
            Path to saved file

        Raises:
            FileIOException: If saving fails
        """
        try:
            filename = self._decoder.saveDecodedFile(
                return_file_name=return_file_name, print_to_output=print_to_output
            )

            if output_path:
                Path(filename).rename(output_path)
                filename = str(output_path)

            logger.info(f"Saved decoded file to {filename}")
            return filename
        except ValueError as ve:
            # Handle ValueError that returns filename in args
            if len(ve.args) > 1:
                filename = ve.args[1]
                logger.info(f"Saved decoded file to {filename}")
                return filename
            raise FileIOException(f"Failed to save decoded file: {ve}")
        except Exception as e:
            logger.error(f"Error saving decoded file: {e}")
            raise FileIOException(f"Failed to save decoded file: {e}", filepath=str(output_path))

    def view_file_with_chunkborders(
        self,
        show_hex: bool = False,
        show_ascii: bool = True,
        last_chunk_len_format: str = "I",
        checksum_len_format: Optional[str] = None,
    ) -> List[Any]:
        """
        View file content with chunk borders.

        Args:
            show_hex: Show hex representation
            show_ascii: Show ASCII representation
            last_chunk_len_format: Format for last chunk length
            checksum_len_format: Format for checksum length

        Returns:
            List of view elements
        """
        try:
            return self.solver.view_file_with_chunkborders(
                show_hex, show_ascii, last_chunk_len_format, checksum_len_format=checksum_len_format
            )
        except Exception as e:
            logger.error(f"Error viewing file: {e}")
            raise DecoderException(f"Failed to view file: {e}")

    def predict_file_type(self) -> str:
        """
        Predict the file type of decoded data.

        Returns:
            Predicted file type string
        """
        try:
            return self.solver.predict_file_type()
        except Exception as e:
            logger.error(f"Error predicting file type: {e}")
            return "Unknown"

    def parse_header(self, last_chunk_len_format: str = "I") -> bool:
        """
        Parse the file header.

        Args:
            last_chunk_len_format: Format for last chunk length

        Returns:
            True if header was parsed successfully
        """
        try:
            self.solver.parse_header(last_chunk_len_format)
            return self.header_chunk is not None
        except Exception as e:
            logger.error(f"Error parsing header: {e}")
            return False

    def is_checksum_correct(self) -> bool:
        """
        Check if the file checksum is correct.

        Returns:
            True if checksum is correct
        """
        try:
            return self.solver.is_checksum_correct()
        except Exception as e:
            logger.error(f"Error checking checksum: {e}")
            return False

    def repair_by_exclusion(self, common_packets: List[bool]) -> Tuple[bool, Optional[GEPP]]:
        """
        Attempt repair by excluding corrupt packets.

        Args:
            common_packets: List of potentially corrupt packets

        Returns:
            Tuple of (success, GEPP instance or None)
        """
        try:
            result, gepp = self.solver.repair_by_exclusion(common_packets)
            if result:
                logger.info("Repair by exclusion successful")
            else:
                logger.warning("Repair by exclusion failed")
            return result, gepp
        except Exception as e:
            logger.error(f"Error during repair by exclusion: {e}")
            return False, None

    def all_solutions_by_reordering(
        self, common_packets: List[bool], only_possible_invalid: bool = False
    ) -> Dict[int, GEPP]:
        """
        Find all solutions by reordering packets.

        Args:
            common_packets: List of potentially corrupt packets
            only_possible_invalid: Only consider possibly invalid packets

        Returns:
            Dictionary mapping permutation IDs to GEPP instances
        """
        try:
            return self.solver.all_solutions_by_reordering(common_packets, only_possible_invalid)
        except Exception as e:
            logger.error(f"Error finding solutions by reordering: {e}")
            raise DecoderException(f"Failed to find solutions: {e}")

    def get_corrupt_chunks_by_packets(
        self, packet_ids: List[int], current_chunk_tag: Optional[List[int]] = None, tag_num: int = 1
    ) -> List[int]:
        """
        Get chunks affected by specific packets.

        Args:
            packet_ids: List of packet IDs
            current_chunk_tag: Current chunk tag list
            tag_num: Tag number to use (1=invalid, 2=valid)

        Returns:
            Updated chunk tag list
        """
        try:
            return self.solver.get_corrupt_chunks_by_packets(packet_ids, current_chunk_tag, tag_num)
        except Exception as e:
            logger.error(f"Error getting corrupt chunks: {e}")
            raise DecoderException(f"Failed to get corrupt chunks: {e}")
