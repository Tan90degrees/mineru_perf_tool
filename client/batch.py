# Copyright (c) Opendatalab. All rights reserved.
"""
File batching and processing utilities.
"""

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from loguru import logger


class BatchProcessor:
    """Process files in batches for benchmarking."""
    
    def __init__(
        self,
        input_dir: str,
        batch_size: int = 1,
        file_extensions: Optional[List[str]] = None,
    ):
        """
        Initialize batch processor.
        
        Args:
            input_dir: Input directory containing files
            batch_size: Number of files per batch
            file_extensions: File extensions to process (e.g., ['.pdf', '.jpg'])
        """
        self.input_dir = Path(input_dir)
        self.batch_size = batch_size
        self.file_extensions = file_extensions or ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        self.files: List[Path] = []
        self.current_index = 0
        
        self._scan_files()
        
    def _scan_files(self) -> None:
        """Scan input directory for files."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        logger.info(f"Scanning files in {self.input_dir}")
        
        for ext in self.file_extensions:
            pattern = f"*{ext}"
            self.files.extend(self.input_dir.glob(pattern))
            # Also search uppercase extension
            self.files.extend(self.input_dir.glob(pattern.upper()))
        
        # Remove duplicates and sort
        self.files = sorted(list(set(self.files)))
        
        logger.info(f"Found {len(self.files)} files to process")
    
    def get_total_batches(self) -> int:
        """Get total number of batches."""
        return (len(self.files) + self.batch_size - 1) // self.batch_size
    
    def get_batch(self, batch_index: int) -> List[Path]:
        """Get a specific batch of files."""
        start_idx = batch_index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.files))
        return self.files[start_idx:end_idx]
    
    def get_next_batch(self) -> Optional[List[Path]]:
        """Get next batch of files."""
        if self.current_index >= len(self.files):
            return None
        
        batch = self.get_batch(self.current_index // self.batch_size)
        self.current_index += self.batch_size
        return batch
    
    async def generate_batches(self) -> AsyncGenerator[List[Path], None]:
        """Generate batches asynchronously."""
        for i in range(0, len(self.files), self.batch_size):
            batch = self.files[i:i + self.batch_size]
            yield batch
            await asyncio.sleep(0)  # Yield control
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get file information."""
        total_size = sum(f.stat().st_size for f in self.files if f.exists())
        
        extensions = {}
        for f in self.files:
            ext = f.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        return {
            "total_files": len(self.files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "extensions": extensions,
            "batch_size": self.batch_size,
            "total_batches": self.get_total_batches(),
        }
    
    def reset(self) -> None:
        """Reset batch processor to start from beginning."""
        self.current_index = 0
