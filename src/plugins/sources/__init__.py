"""Source plugins for the Agentic Data Pipeline Ingestor.

This package contains source plugins for various data sources:
- S3SourcePlugin: Amazon S3
- AzureBlobSourcePlugin: Azure Blob Storage
- SharePointSourcePlugin: SharePoint Online
"""

from src.plugins.sources.s3_source import S3SourcePlugin
from src.plugins.sources.azure_blob_source import AzureBlobSourcePlugin
from src.plugins.sources.sharepoint_source import SharePointSourcePlugin

__all__ = [
    "S3SourcePlugin",
    "AzureBlobSourcePlugin",
    "SharePointSourcePlugin",
]
