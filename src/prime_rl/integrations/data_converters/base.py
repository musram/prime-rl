"""
Base data converter interface.

This module defines the abstract base class for converting external data sources
into the Interaction Trace JSONL format.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional
from pathlib import Path

from prime_rl.core.interaction_trace import InteractionTrace


class DataConverter(ABC):
    """
    Abstract base class for data converters.
    
    Data converters read data from external sources and convert them into
    the Interaction Trace JSONL format for offline RL training.
    
    Subclasses should implement:
    - `convert()`: Convert source data to InteractionTrace objects
    - `convert_to_jsonl()`: Convert and write directly to JSONL file
    """
    
    @abstractmethod
    def convert(self, source_data: Any) -> List[InteractionTrace]:
        """
        Convert source data to InteractionTrace objects.
        
        Args:
            source_data: Source data in the format specific to this converter
            
        Returns:
            List of InteractionTrace objects
        """
        pass
    
    @abstractmethod
    def convert_to_jsonl(
        self,
        source_path: Path,
        output_path: Path,
        **kwargs,
    ) -> None:
        """
        Convert source data file to Interaction Trace JSONL format.
        
        Args:
            source_path: Path to source data file
            output_path: Path to output JSONL file
            **kwargs: Additional converter-specific options
        """
        pass
    
    def validate_trace(self, trace: InteractionTrace) -> bool:
        """
        Validate an InteractionTrace object.
        
        Args:
            trace: InteractionTrace to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation
        if not trace.trace_id:
            return False
        if not trace.environment_id:
            return False
        if not trace.steps or len(trace.steps) == 0:
            return False
        return True

