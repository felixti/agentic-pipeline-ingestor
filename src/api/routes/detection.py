"""API routes for content detection endpoints."""

import asyncio
import hashlib
import tempfile
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from src.api.detection_models import (
    BatchDetectionItem,
    BatchDetectionRequest,
    BatchDetectionResponse,
    BatchDetectionResponseData,
    DetectionRequest,
    DetectionResponse,
    DetectionUrlRequest,
    convert_analysis_result_to_response,
    create_meta,
)
from src.core.content_detection.analyzer import PDFContentAnalyzer

router = APIRouter(prefix="/detect", tags=["Content Detection"])

# Rate limiting storage (in production, use Redis)
_rate_limit_store = {}
RATE_LIMIT_REQUESTS = 60  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds


def _get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting.
    
    Args:
        request: FastAPI request
        
    Returns:
        Client identifier (API key or IP address)
    """
    # Try to get API key from header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"
    
    # Fall back to IP address
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def _check_rate_limit(identifier: str) -> tuple[bool, int]:
    """Check if request is within rate limit.
    
    Args:
        identifier: Client identifier
        
    Returns:
        Tuple of (allowed, retry_after_seconds)
    """
    import time
    
    now = int(time.time())
    window_start = now - RATE_LIMIT_WINDOW
    
    # Get or create request history
    if identifier not in _rate_limit_store:
        _rate_limit_store[identifier] = []
    
    # Clean old requests
    _rate_limit_store[identifier] = [
        ts for ts in _rate_limit_store[identifier] if ts > window_start
    ]
    
    # Check limit
    if len(_rate_limit_store[identifier]) >= RATE_LIMIT_REQUESTS:
        oldest_request = min(_rate_limit_store[identifier])
        retry_after = RATE_LIMIT_WINDOW - (now - oldest_request)
        return False, max(1, retry_after)
    
    # Record this request
    _rate_limit_store[identifier].append(now)
    return True, 0


async def _download_file_from_url(url: str, headers: dict = None) -> bytes:
    """Download file from URL.
    
    Args:
        url: URL to download from
        headers: Optional HTTP headers
        
    Returns:
        File content as bytes
        
    Raises:
        HTTPException: If download fails
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=headers,
                timeout=30.0,
                follow_redirects=True
            )
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download file: HTTP {e.response.status_code}"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download file: {e!s}"
        )


def _validate_pdf_file(file_content: bytes) -> None:
    """Validate that file content is a valid PDF.
    
    Args:
        file_content: File content as bytes
        
    Raises:
        HTTPException: If file is not a valid PDF
    """
    if len(file_content) < 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_FILE_FORMAT",
                    "message": "File is too small to be a valid PDF"
                }
            }
        )
    
    # Check PDF magic number
    if not file_content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_FILE_FORMAT",
                    "message": "File is not a valid PDF (missing PDF header)"
                }
            }
        )


@router.post(
    "",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect content type of uploaded PDF",
    description="Upload a PDF file and detect whether it's text-based, scanned, or mixed content."
)
async def detect_content(
    request: Request,
    file: UploadFile = File(..., description="PDF file to analyze"),
    detailed: bool = Form(default=False, description="Include per-page details"),
):
    """Detect content type of uploaded PDF file.
    
    Args:
        request: FastAPI request
        file: Uploaded PDF file
        detailed: Whether to include detailed per-page analysis
        
    Returns:
        Detection result
    """
    # Rate limiting
    client_id = _get_client_identifier(request)
    allowed, retry_after = _check_rate_limit(client_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(retry_after)},
            detail={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Try again in {retry_after} seconds."
                }
            }
        )
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_FILE_FORMAT",
                    "message": "Only PDF files are supported"
                }
            }
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Validate PDF
        _validate_pdf_file(content)
        
        # Save to temporary file for analysis
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            # Analyze
            analyzer = PDFContentAnalyzer()
            result = analyzer.analyze(tmp_path)
            
            # Build response
            request_id = str(uuid4())
            response_data = convert_analysis_result_to_response(result, detailed=detailed)
            
            return DetectionResponse(
                data=response_data,
                meta=create_meta(request_id)
            )
        finally:
            # Cleanup
            tmp_path.unlink(missing_ok=True)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "ANALYSIS_ERROR",
                    "message": f"Failed to analyze PDF: {e!s}"
                }
            }
        )


@router.post(
    "/url",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect content type from URL",
    description="Download a PDF from URL and detect its content type."
)
async def detect_content_from_url(
    request: Request,
    body: DetectionUrlRequest,
):
    """Detect content type of PDF from URL.
    
    Args:
        request: FastAPI request
        body: URL request body
        
    Returns:
        Detection result
    """
    # Rate limiting
    client_id = _get_client_identifier(request)
    allowed, retry_after = _check_rate_limit(client_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(retry_after)},
            detail={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Try again in {retry_after} seconds."
                }
            }
        )
    
    try:
        # Download file
        url = str(body.url)
        content = await _download_file_from_url(url, body.headers)
        
        # Validate PDF
        _validate_pdf_file(content)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            # Analyze
            analyzer = PDFContentAnalyzer()
            result = analyzer.analyze(tmp_path)
            
            # Build response
            request_id = str(uuid4())
            response_data = convert_analysis_result_to_response(result, detailed=body.detailed)
            
            return DetectionResponse(
                data=response_data,
                meta=create_meta(request_id)
            )
        finally:
            # Cleanup
            tmp_path.unlink(missing_ok=True)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "ANALYSIS_ERROR",
                    "message": f"Failed to analyze PDF: {e!s}"
                }
            }
        )


@router.post(
    "/batch",
    response_model=BatchDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch content detection",
    description="Detect content type for multiple PDF files (max 10)."
)
async def detect_content_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="PDF files to analyze (max 10)"),
    detailed: bool = Form(default=False),
):
    """Detect content type for multiple PDF files.
    
    Args:
        request: FastAPI request
        files: List of uploaded PDF files
        detailed: Whether to include detailed per-page analysis
        
    Returns:
        Batch detection results
    """
    # Rate limiting (batch counts as 1 request)
    client_id = _get_client_identifier(request)
    allowed, retry_after = _check_rate_limit(client_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(retry_after)},
            detail={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Try again in {retry_after} seconds."
                }
            }
        )
    
    # Validate file count
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "TOO_MANY_FILES",
                    "message": "Maximum 10 files allowed per batch request"
                }
            }
        )
    
    if len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "NO_FILES",
                    "message": "No files provided"
                }
            }
        )
    
    async def analyze_single_file(file: UploadFile) -> BatchDetectionItem:
        """Analyze a single file for batch processing."""
        filename = file.filename or "unknown"
        
        try:
            # Validate file type
            if not filename.lower().endswith(".pdf"):
                return BatchDetectionItem(
                    filename=filename,
                    content_type="unknown",
                    confidence=0.0,
                    recommended_parser="",
                    processing_time_ms=0,
                    error="Only PDF files are supported"
                )
            
            # Read and validate
            content = await file.read()
            try:
                _validate_pdf_file(content)
            except HTTPException as e:
                error_detail = e.detail.get("error", {}) if isinstance(e.detail, dict) else {}
                return BatchDetectionItem(
                    filename=filename,
                    content_type="unknown",
                    confidence=0.0,
                    recommended_parser="",
                    processing_time_ms=0,
                    error=error_detail.get("message", "Invalid PDF file")
                )
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)
            
            try:
                # Analyze
                analyzer = PDFContentAnalyzer()
                result = analyzer.analyze(tmp_path)
                
                return BatchDetectionItem(
                    filename=filename,
                    content_type=result.content_type,
                    confidence=result.confidence,
                    recommended_parser=result.recommended_parser,
                    processing_time_ms=result.processing_time_ms
                )
            finally:
                tmp_path.unlink(missing_ok=True)
                
        except Exception as e:
            return BatchDetectionItem(
                filename=filename,
                content_type="unknown",
                confidence=0.0,
                recommended_parser="",
                processing_time_ms=0,
                error=str(e)
            )
    
    # Process all files in parallel
    results = await asyncio.gather(*[analyze_single_file(f) for f in files])
    
    # Calculate summary
    total = len(results)
    successful = sum(1 for r in results if r.error is None)
    text_based = sum(1 for r in results if r.content_type == "text_based" and r.error is None)
    scanned = sum(1 for r in results if r.content_type == "scanned" and r.error is None)
    mixed = sum(1 for r in results if r.content_type == "mixed" and r.error is None)
    errors = total - successful
    
    request_id = str(uuid4())
    
    return BatchDetectionResponse(
        data=BatchDetectionResponseData(
            results=results,
            summary={
                "total": total,
                "successful": successful,
                "errors": errors,
                "text_based": text_based,
                "scanned": scanned,
                "mixed": mixed
            }
        ),
        meta=create_meta(request_id)
    )
