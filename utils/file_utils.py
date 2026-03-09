# Copyright (c) Opendatalab. All rights reserved.
"""
File utilities for handling various document types.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
from loguru import logger


class FileType(Enum):
    """Supported file types."""
    PDF = "pdf"
    IMAGE = "image"
    UNKNOWN = "unknown"


class FileUtils:
    """Utilities for file operations."""
    
    # Supported file extensions
    PDF_EXTENSIONS = {'.pdf'}
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif', '.jp2'}
    
    @staticmethod
    def get_file_type(file_path: Path) -> FileType:
        """
        Determine file type from extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            FileType enum value
        """
        ext = file_path.suffix.lower()
        
        if ext in FileUtils.PDF_EXTENSIONS:
            return FileType.PDF
        elif ext in FileUtils.IMAGE_EXTENSIONS:
            return FileType.IMAGE
        else:
            return FileType.UNKNOWN
    
    @staticmethod
    def scan_directory(
        directory: Path,
        extensions: Optional[List[str]] = None,
        recursive: bool = False,
    ) -> List[Path]:
        """
        Scan directory for files with specific extensions.
        
        Args:
            directory: Directory to scan
            extensions: File extensions to include (None = all supported)
            recursive: Whether to scan recursively
            
        Returns:
            List of file paths
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if extensions is None:
            extensions = list(FileUtils.PDF_EXTENSIONS | FileUtils.IMAGE_EXTENSIONS)
        
        files = []
        
        if recursive:
            for ext in extensions:
                pattern = f"**/*{ext}"
                files.extend(directory.glob(pattern))
                # Also search uppercase extension
                files.extend(directory.glob(pattern.upper()))
        else:
            for ext in extensions:
                pattern = f"*{ext}"
                files.extend(directory.glob(pattern))
                files.extend(directory.glob(pattern.upper()))
        
        # Remove duplicates and sort
        files = sorted(list(set(files)))
        
        logger.info(f"Scanned {len(files)} files in {directory}")
        return files
    
    @staticmethod
    def get_file_info(file_path: Path) -> Dict[str, Any]:
        """
        Get file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        stat = file_path.stat()
        
        return {
            "path": str(file_path),
            "name": file_path.name,
            "stem": file_path.stem,
            "extension": file_path.suffix.lower(),
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "file_type": FileUtils.get_file_type(file_path).value,
            "modified_time": stat.st_mtime,
        }
    
    @staticmethod
    def calculate_file_hash(file_path: Path, algorithm: str = "md5") -> str:
        """
        Calculate file hash.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256)
            
        Returns:
            Hexadecimal hash string
        """
        if algorithm == "md5":
            hasher = hashlib.md5()
        elif algorithm == "sha1":
            hasher = hashlib.sha1()
        elif algorithm == "sha256":
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    @staticmethod
    def ensure_directory(path: Path) -> Path:
        """
        Ensure directory exists, create if necessary.
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def cleanup_old_files(directory: Path, max_age_hours: int = 24) -> int:
        """
        Clean up files older than specified age.
        
        Args:
            directory: Directory to clean
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of files removed
        """
        import time
        
        if not directory.exists():
            return 0
        
        removed_count = 0
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
        
        logger.info(f"Cleaned up {removed_count} old files from {directory}")
        return removed_count
    
    @staticmethod
    def split_files_by_size(
        files: List[Path],
        target_size_mb: float = 100.0,
    ) -> List[List[Path]]:
        """
        Split files into groups by total size.
        
        Args:
            files: List of file paths
            target_size_mb: Target size per group in MB
            
        Returns:
            List of file groups
        """
        groups = []
        current_group = []
        current_size = 0.0
        
        for file_path in files:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if current_size + file_size_mb > target_size_mb and current_group:
                groups.append(current_group)
                current_group = []
                current_size = 0.0
            
            current_group.append(file_path)
            current_size += file_size_mb
        
        if current_group:
            groups.append(current_group)
        
        logger.info(f"Split {len(files)} files into {len(groups)} groups")
        return groups
    
    @staticmethod
    def validate_file(file_path: Path) -> bool:
        """
        Validate if file is accessible and supported.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if valid, False otherwise
        """
        if not file_path.exists():
            return False
        
        if not file_path.is_file():
            return False
        
        if FileUtils.get_file_type(file_path) == FileType.UNKNOWN:
            return False
        
        return True
