"""File storage utilities for uploaded files."""

import shutil
from pathlib import Path
from uuid import uuid4


class FileStorage:
    """File storage handler for uploaded documents."""
    
    def __init__(self, base_path: str = "/tmp/pipeline"):
        """Initialize file storage.
        
        Args:
            base_path: Base directory for file storage
        """
        self.base_path = Path(base_path)
        self.uploads_dir = self.base_path / "uploads"
        self.staging_dir = self.base_path / "staging"
        
        # Ensure directories exist
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
    
    def save_upload(self, file_content: bytes, original_filename: str) -> Path:
        """Save uploaded file to storage.
        
        Args:
            file_content: File content as bytes
            original_filename: Original filename
            
        Returns:
            Path to saved file
        """
        file_id = str(uuid4())
        file_ext = Path(original_filename).suffix
        file_path = self.uploads_dir / f"{file_id}{file_ext}"
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return file_path
    
    def get_file_path(self, file_id: str) -> Path | None:
        """Get path to stored file by ID.
        
        Args:
            file_id: File ID (UUID)
            
        Returns:
            Path if file exists, None otherwise
        """
        # Search in uploads directory
        for ext in [".pdf", ".docx", ".txt", ".json", ".csv", ".html", ""]:
            path = self.uploads_dir / f"{file_id}{ext}"
            if path.exists():
                return path
        return None
    
    def delete_file(self, file_path: Path | str) -> bool:
        """Delete a stored file.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if deleted, False otherwise
        """
        path = Path(file_path)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Clean up files older than specified hours.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of files deleted
        """
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0
        
        for directory in [self.uploads_dir, self.staging_dir]:
            for file_path in directory.iterdir():
                if file_path.is_file():
                    file_stat = file_path.stat()
                    file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
                    if file_mtime < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
        
        return deleted_count
    
    def get_file_size(self, file_path: Path | str) -> int:
        """Get file size in bytes.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
        """
        return Path(file_path).stat().st_size
