"""Parser plugins package.

This package contains parser plugins for extracting content from documents.
"""

from src.plugins.parsers.azure_ocr_parser import AzureOCRParser
from src.plugins.parsers.csv_parser import CSVParser
from src.plugins.parsers.docling_parser import DoclingParser
from src.plugins.parsers.email_parser import EmailParser
from src.plugins.parsers.json_parser import JSONParser
from src.plugins.parsers.xml_parser import XMLParser

__all__ = [
    "AzureOCRParser",
    "CSVParser",
    "DoclingParser",
    "EmailParser",
    "JSONParser",
    "XMLParser",
]
