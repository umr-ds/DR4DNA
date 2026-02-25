# -*- coding: utf-8 -*-
"""
Repair service for DR4DNA.

This service encapsulates all repair-related operations and provides
a clean API for the UI layer.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from exceptions import RepairException, RepairValidationError, DataIntegrityException
from logging_config import get_logger
from state import AppState
from services.decoder_service import DecoderService


logger = get_logger(__name__)


class RepairService:
    """
    Service class for repair operations.
    
    This class encapsulates all repair-related functionality and provides
    methods for chunk repair, validation, and error analysis.
    
    Attributes:
        decoder_service: Decoder service instance
        state: Application state instance
    """
    
    def __init__(
        self,
        decoder_service: DecoderService,
        state: Optional[AppState] = None
    ):
        """
        Initialize the repair service.
        
        Args:
            decoder_service: Decoder service instance
            state: Application state instance (uses global if None)
        """
        from state import get_app_state
        
        self.decoder_service = decoder_service
        self.state = state or get_app_state()
        self._solver = decoder_service.solver
        self._gepp = decoder_service.gepp
        
        logger.info("RepairService initialized")
    
    def repair_chunk(
        self,
        chunk_id: int,
        packet_id: int,
        corrected_data: bytes
    ) -> bool:
        """
        Repair a specific chunk with corrected data.
        
        Args:
            chunk_id: ID of chunk to repair
            packet_id: ID of packet responsible for the chunk
            corrected_data: Corrected chunk data as bytes
        
        Returns:
            True if repair was successful
        
        Raises:
            RepairValidationError: If validation fails
        """
        try:
            # Validate inputs
            if not 0 <= chunk_id < self.decoder_service.number_of_chunks:
                raise RepairValidationError(
                    f"Invalid chunk ID: {chunk_id}",
                    chunk_id=chunk_id
                )
            
            if len(corrected_data) != self._gepp.b.shape[1]:
                raise RepairValidationError(
                    f"Invalid data length: expected {self._gepp.b.shape[1]}, "
                    f"got {len(corrected_data)}",
                    chunk_id=chunk_id,
                    invalid_data=corrected_data
                )
            
            # Perform repair
            self.decoder_service.manual_repair(chunk_id, packet_id, corrected_data)
            
            logger.info(f"Successfully repaired chunk {chunk_id}")
            return True
            
        except RepairValidationError:
            raise
        except Exception as e:
            logger.error(f"Error repairing chunk {chunk_id}: {e}")
            raise RepairException(f"Failed to repair chunk: {e}")
    
    def repair_chunk_from_hex(
        self,
        chunk_id: int,
        packet_id: int,
        hex_value: str
    ) -> bool:
        """
        Repair a chunk from hex string.
        
        Args:
            chunk_id: ID of chunk to repair
            packet_id: ID of packet responsible for the chunk
            hex_value: Corrected data as hex string (with or without spaces)
        
        Returns:
            True if repair was successful
        """
        try:
            # Clean and convert hex value
            cleaned_hex = hex_value.replace(" ", "")
            corrected_data = bytearray.fromhex(cleaned_hex)
            
            return self.repair_chunk(chunk_id, packet_id, corrected_data)
            
        except ValueError as e:
            raise RepairValidationError(
                f"Invalid hex value: {e}",
                chunk_id=chunk_id,
                invalid_data=hex_value
            )
    
    def validate_repair(
        self,
        chunk_id: int,
        expected_checksum: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Validate a repair operation.
        
        Args:
            chunk_id: ID of repaired chunk
            expected_checksum: Optional expected checksum
        
        Returns:
            Dictionary with validation results
        """
        try:
            result = {
                'valid': True,
                'chunk_id': chunk_id,
                'errors': []
            }
            
            # Check chunk bounds
            if not 0 <= chunk_id < self.decoder_service.number_of_chunks:
                result['valid'] = False
                result['errors'].append(f"Chunk ID {chunk_id} out of bounds")
            
            # Check data integrity if checksum provided
            if expected_checksum is not None:
                chunk_data = self._gepp.b[chunk_id]
                # Add checksum validation logic here
                
            logger.debug(f"Validation result for chunk {chunk_id}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error validating repair: {e}")
            return {
                'valid': False,
                'chunk_id': chunk_id,
                'errors': [str(e)]
            }
    
    def find_error_regions(
        self,
        original_data: bytes,
        reconstructed_data: bytes
    ) -> np.ndarray:
        """
        Find error regions by comparing original and reconstructed data.
        
        Args:
            original_data: Original file data
            reconstructed_data: Reconstructed file data
        
        Returns:
            Error matrix showing differences
        """
        try:
            start_pos = (1 if self.decoder_service.header_chunk is not None else 0) * self._gepp.b.shape[1]
            
            # Ensure same length
            min_len = min(len(original_data), len(reconstructed_data))
            
            # Calculate differences
            error_matrix = np.zeros(self._gepp.b.shape[0] * self._gepp.b.shape[1], dtype=np.float32)
            
            for i in range(min_len - start_pos):
                diff = original_data[i] ^ reconstructed_data[i]
                if diff != 0:
                    error_matrix[i + start_pos] = diff
            
            return error_matrix.reshape(-1, self._gepp.b.shape[1])
            
        except Exception as e:
            logger.error(f"Error finding error regions: {e}")
            raise RepairException(f"Failed to find error regions: {e}")
    
    def find_incorrect_columns(
        self,
        error_matrix: np.ndarray
    ) -> List[Tuple[int, float, int, Any]]:
        """
        Find columns with errors.
        
        Args:
            error_matrix: Matrix of errors
        
        Returns:
            List of (column_index, diff, count, counter) tuples
        """
        from collections import Counter
        
        try:
            results = []
            row_counters = []
            
            # Count errors per column
            for i in range(error_matrix.shape[1]):
                ctr = Counter(error_matrix[:, i])
                row_counters.append(ctr)
            
            # Find columns with errors
            for i, counter in enumerate(row_counters):
                exists_gr_zero = False
                for diff, count in counter.most_common(4):
                    if diff < 1.0:
                        continue
                    exists_gr_zero = True
                    results.append((i, diff, count, counter))
                    break
                if not exists_gr_zero:
                    results.append((i, 0.0, 0, counter))
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding incorrect columns: {e}")
            return []
    
    def analyze_error_patterns(
        self,
        error_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze patterns in error matrix.
        
        Args:
            error_matrix: Matrix of errors
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Count affected rows and columns
            affected_rows = np.any(error_matrix != 0, axis=1)
            affected_cols = np.any(error_matrix != 0, axis=0)
            
            # Calculate error density
            total_errors = np.count_nonzero(error_matrix != 0)
            error_density = total_errors / error_matrix.size
            
            # Find error clusters
            row_error_counts = np.sum(error_matrix != 0, axis=1)
            col_error_counts = np.sum(error_matrix != 0, axis=0)
            
            result = {
                'total_errors': int(total_errors),
                'affected_rows': int(np.sum(affected_rows)),
                'affected_cols': int(np.sum(affected_cols)),
                'error_density': float(error_density),
                'max_row_errors': int(np.max(row_error_counts)),
                'max_col_errors': int(np.max(col_error_counts)),
                'avg_row_errors': float(np.mean(row_error_counts)),
                'avg_col_errors': float(np.mean(col_error_counts)),
            }
            
            logger.debug(f"Error pattern analysis: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing error patterns: {e}")
            return {}
    
    def get_repair_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about repair operations.
        
        Returns:
            Dictionary with repair statistics
        """
        chunk_tag = self.state.get_chunk_tag()
        
        stats = {
            'total_chunks': len(chunk_tag),
            'unknown_chunks': sum(1 for t in chunk_tag if t == 0),
            'corrupt_chunks': sum(1 for t in chunk_tag if t == 1),
            'correct_chunks': sum(1 for t in chunk_tag if t == 2),
            'missing_chunks': sum(1 for t in chunk_tag if t == 3),
        }
        
        stats['known_percentage'] = (
            (stats['correct_chunks'] + stats['corrupt_chunks']) / 
            max(stats['total_chunks'], 1) * 100
        )
        
        logger.debug(f"Repair statistics: {stats}")
        return stats
    
    def reset_repair_state(self):
        """Reset all repair state to initial values."""
        self.state.reset_chunk_tag()
        self.state.reset_column_tag()
        logger.info("Repair state reset")
