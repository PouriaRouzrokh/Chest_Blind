"""Checkpoint management for resume capability."""

import json
import os
import csv
from typing import Optional, Dict
import config


class CheckpointManager:
    """Manages checkpoint state for resumable processing."""

    def __init__(self, output_dir: str):
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory for output files and checkpoint
        """
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(output_dir, config.CHECKPOINT_FILE)
        self.output_path = os.path.join(output_dir, config.PRODUCTION_OUTPUT_FILE)

    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if it exists.

        Returns:
            Checkpoint data dict or None if no checkpoint exists
        """
        if not os.path.exists(self.checkpoint_path):
            return None

        try:
            with open(self.checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def save_checkpoint(self, last_processed_row: int, total_rows: int):
        """Save checkpoint state.

        Args:
            last_processed_row: Index of last successfully processed row (0-indexed)
            total_rows: Total number of rows in dataset
        """
        checkpoint = {
            'last_processed_row': last_processed_row,
            'total_rows': total_rows,
            'output_file': self.output_path
        }

        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def get_processed_count(self) -> int:
        """Get number of already processed rows by reading output file.

        Returns:
            Number of data rows already processed (0 if file doesn't exist)
        """
        if not os.path.exists(self.output_path):
            return 0

        try:
            with open(self.output_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Count rows excluding header
                return sum(1 for _ in reader) - 1
        except Exception:
            return 0

    def clear_checkpoint(self):
        """Clear checkpoint file (for starting fresh)."""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    def output_exists(self) -> bool:
        """Check if output file exists.

        Returns:
            True if output file exists
        """
        return os.path.exists(self.output_path)
