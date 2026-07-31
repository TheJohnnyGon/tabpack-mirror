"""Checkpoint storage for ensemble models.

This module provides a centralized store for model checkpoints used in ensembles.
Each checkpoint is uniquely identified by (model_id, step), allowing the same model
to be included multiple times in an ensemble with different training steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Mapping


class EnsembleCheckpointStore:
    """Centralized storage for ensemble model checkpoints.
    
    Each checkpoint is uniquely identified by (model_id, step), which allows:
    - The same model to appear multiple times in an ensemble with different steps
    - No overwriting when ensemble contains duplicate model IDs
    - Clear semantics for saving, retrieving, and checking checkpoints
    
    Example:
        >>> store = EnsembleCheckpointStore()
        >>> store.save_checkpoint(model_id=32, step=123060, state_dict=model.state_dict())
        >>> store.save_checkpoint(model_id=32, step=228540, state_dict=model.state_dict())
        >>> store.has_checkpoint(32, 123060)
        True
        >>> checkpoint = store.get_checkpoint(32, 123060)
    """
    
    def __init__(self) -> None:
        self._checkpoints: dict[tuple[int, int], dict[str, Tensor]] = {}
        self._metadata: dict[tuple[int, int], dict] = {}
    
    def save_checkpoint(
        self,
        model_id: int,
        step: int,
        state_dict: Mapping[str, Tensor],
        *,
        metadata: dict | None = None,
    ) -> None:
        """Save a model checkpoint.
        
        Args:
            model_id: Unique identifier of the model.
            step: Training step at which the checkpoint was taken.
            state_dict: Model state dictionary (will be cloned and moved to CPU).
            metadata: Optional metadata to store with the checkpoint.
        """
        key = (model_id, step)
        self._checkpoints[key] = {
            name: value.detach().cpu().clone()
            for name, value in state_dict.items()
        }
        if metadata is not None:
            self._metadata[key] = metadata
    
    def get_checkpoint(self, model_id: int, step: int) -> dict[str, Tensor]:
        """Retrieve a checkpoint by model_id and step.
        
        Args:
            model_id: Unique identifier of the model.
            step: Training step at which the checkpoint was taken.
            
        Returns:
            The state dictionary for the requested checkpoint.
            
        Raises:
            KeyError: If the checkpoint does not exist.
        """
        key = (model_id, step)
        if key not in self._checkpoints:
            raise KeyError(f"Checkpoint not found: model_id={model_id}, step={step}")
        return self._checkpoints[key]
    
    def has_checkpoint(self, model_id: int, step: int) -> bool:
        """Check if a checkpoint exists.
        
        Args:
            model_id: Unique identifier of the model.
            step: Training step at which the checkpoint was taken.
            
        Returns:
            True if the checkpoint exists, False otherwise.
        """
        return (model_id, step) in self._checkpoints
    
    def get_all_checkpoints(self) -> dict[tuple[int, int], dict[str, Tensor]]:
        """Get all stored checkpoints.
        
        Returns:
            Dictionary mapping (model_id, step) to state dictionaries.
        """
        return self._checkpoints
    
    def get_unique_model_ids(self) -> set[int]:
        """Get all unique model IDs that have checkpoints.
        
        Returns:
            Set of unique model IDs.
        """
        return {model_id for model_id, _ in self._checkpoints.keys()}
    
    def get_checkpoints_for_model(self, model_id: int) -> dict[int, dict[str, Tensor]]:
        """Get all checkpoints for a specific model.
        
        Args:
            model_id: Unique identifier of the model.
            
        Returns:
            Dictionary mapping step to state dictionary for the given model.
        """
        return {
            step: checkpoint
            for (mid, step), checkpoint in self._checkpoints.items()
            if mid == model_id
        }
    
    def remove_checkpoint(self, model_id: int, step: int) -> None:
        """Remove a checkpoint.
        
        Args:
            model_id: Unique identifier of the model.
            step: Training step at which the checkpoint was taken.
        """
        key = (model_id, step)
        self._checkpoints.pop(key, None)
        self._metadata.pop(key, None)
    
    def clear(self) -> None:
        """Remove all checkpoints."""
        self._checkpoints.clear()
        self._metadata.clear()
    
    def __len__(self) -> int:
        return len(self._checkpoints)
    
    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._checkpoints
