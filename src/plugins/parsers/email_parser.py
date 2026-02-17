"""Email parser plugin for EML and MSG files.

This module provides email parsing capabilities with support for:
- Header extraction (From, To, Subject, Date, etc.)
- Body extraction (plain text and HTML)
- Attachment handling
- Thread reconstruction
- MIME structure parsing
"""

import email
import email.message
import email.policy
import logging
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from src.plugins.base import (
    HealthStatus,
    ParserPlugin,
    ParsingResult,
    PluginMetadata,
    PluginType,
    SupportResult,
)

logger = logging.getLogger(__name__)


@dataclass
class EmailAttachment:
    """Represents an email attachment."""
    filename: str
    content_type: str
    size: int
    content_id: str | None = None
    content: bytes | None = None
    is_inline: bool = False


@dataclass
class EmailAddress:
    """Represents an email address with display name."""
    address: str
    display_name: str | None = None

    def __str__(self) -> str:
        if self.display_name:
            return f"{self.display_name} <{self.address}>"
        return self.address


@dataclass
class EmailHeaders:
    """Email header information."""
    message_id: str | None = None
    subject: str | None = None
    from_addr: EmailAddress | None = None
    to_addrs: list[EmailAddress] = field(default_factory=list)
    cc_addrs: list[EmailAddress] = field(default_factory=list)
    bcc_addrs: list[EmailAddress] = field(default_factory=list)
    reply_to: EmailAddress | None = None
    date: datetime | None = None
    in_reply_to: str | None = None
    references: list[str] = field(default_factory=list)
    thread_topic: str | None = None
    thread_index: str | None = None
    priority: str | None = None
    sensitivity: str | None = None


@dataclass
class EmailBody:
    """Email body content."""
    plain_text: str | None = None
    html: str | None = None
    content_type: str = "text/plain"
    charset: str = "utf-8"


@dataclass
class ParsedEmail:
    """Complete parsed email structure."""
    headers: EmailHeaders = field(default_factory=EmailHeaders)
    body: EmailBody = field(default_factory=EmailBody)
    attachments: list[EmailAttachment] = field(default_factory=list)
    embedded_images: list[EmailAttachment] = field(default_factory=list)
    raw_size: int = 0


class EmailParser(ParserPlugin):
    """Email parser plugin for EML and MSG files.
    
    This parser handles email files with comprehensive extraction of:
    - Headers (From, To, Subject, Date, threading info)
    - Body content (plain text and HTML)
    - Attachments with metadata
    - Embedded images
    - MIME structure
    
    Example:
        >>> parser = EmailParser()
        >>> await parser.initialize({})
        >>> result = await parser.parse("/path/to/email.eml")
        >>> print(result.text)
    """

    SUPPORTED_FORMATS = [".eml", ".msg"]

    MIME_TYPE_MAP = {
        "message/rfc822": 1.0,
        "application/vnd.ms-outlook": 0.95,  # .msg files
    }

    def __init__(self) -> None:
        """Initialize the email parser."""
        self._config: dict[str, Any] = {}

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            id="email",
            name="Email Parser",
            version="1.0.0",
            type=PluginType.PARSER,
            description="Parser for EML and MSG email files with attachment extraction",
            author="Pipeline Team",
            supported_formats=self.SUPPORTED_FORMATS,
            requires_auth=False,
        )

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the parser with configuration.
        
        Args:
            config: Parser configuration options including:
                - encoding: Default encoding (default: utf-8)
                - extract_attachments: Whether to extract attachment content (default: True)
                - max_attachment_size: Maximum attachment size in bytes (default: 50MB)
                - save_attachments: Whether to save attachments to disk (default: False)
                - attachment_path: Directory to save attachments
                - html_to_text: Convert HTML to plain text (default: True)
                - include_headers: Include all headers in output (default: True)
        """
        self._config = {
            "encoding": config.get("encoding", "utf-8"),
            "extract_attachments": config.get("extract_attachments", True),
            "max_attachment_size": config.get("max_attachment_size", 50 * 1024 * 1024),
            "save_attachments": config.get("save_attachments", False),
            "attachment_path": config.get("attachment_path", "./attachments"),
            "html_to_text": config.get("html_to_text", True),
            "include_headers": config.get("include_headers", True),
        }
        logger.info("Email parser initialized")

    async def supports(
        self,
        file_path: str,
        mime_type: str | None = None,
    ) -> SupportResult:
        """Check if this parser supports the given file.
        
        Args:
            file_path: Path to the file
            mime_type: Optional MIME type of the file
            
        Returns:
            SupportResult indicating support status
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        # Check extension
        if extension == ".eml":
            return SupportResult(
                supported=True,
                confidence=0.95,
                reason="EML format",
            )

        if extension == ".msg":
            return SupportResult(
                supported=True,
                confidence=0.90,
                reason="MSG format (requires extract-msg library)",
            )

        # Check MIME type if provided
        if mime_type and mime_type in self.MIME_TYPE_MAP:
            return SupportResult(
                supported=True,
                confidence=self.MIME_TYPE_MAP[mime_type],
                reason=f"Supported MIME type: {mime_type}",
            )

        return SupportResult(
            supported=False,
            confidence=1.0,
            reason=f"Unsupported file format: {extension}",
        )

    async def parse(
        self,
        file_path: str,
        options: dict[str, Any] | None = None,
    ) -> ParsingResult:
        """Parse an email file and extract content.
        
        Args:
            file_path: Path to the file to parse
            options: Parser-specific options that override config
            
        Returns:
            ParsingResult containing extracted content
        """
        import time

        opts = {**self._config, **(options or {})}
        start_time = time.time()

        path = Path(file_path)
        if not path.exists():
            return ParsingResult(
                success=False,
                error=f"File not found: {file_path}",
            )

        # Check support first
        support = await self.supports(file_path)
        if not support.supported:
            return ParsingResult(
                success=False,
                error=support.reason,
            )

        try:
            extension = path.suffix.lower()

            if extension == ".msg":
                parsed_email = await self._parse_msg(file_path, opts)
            else:
                parsed_email = await self._parse_eml(file_path, opts)

            # Generate result
            result = self._create_result(parsed_email, path, opts)
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            return result

        except Exception as e:
            logger.error(f"Email parsing failed: {e}", exc_info=True)
            return ParsingResult(
                success=False,
                error=f"Parsing failed: {e!s}",
            )

    async def _parse_eml(
        self,
        file_path: str,
        options: dict[str, Any],
    ) -> ParsedEmail:
        """Parse an EML file.
        
        Args:
            file_path: Path to the EML file
            options: Parsing options
            
        Returns:
            ParsedEmail structure
        """
        path = Path(file_path)

        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(
                f,
                policy=email.policy.default
            )

        headers = self._extract_headers(msg)
        body = self._extract_body(msg, options)
        attachments, embedded = self._extract_attachments(msg, options)

        return ParsedEmail(
            headers=headers,
            body=body,
            attachments=attachments,
            embedded_images=embedded,
            raw_size=path.stat().st_size,
        )

    async def _parse_msg(
        self,
        file_path: str,
        options: dict[str, Any],
    ) -> ParsedEmail:
        """Parse an MSG file.
        
        Args:
            file_path: Path to the MSG file
            options: Parsing options
            
        Returns:
            ParsedEmail structure
        """
        try:
            import extract_msg

            msg = extract_msg.Message(file_path)

            # Extract headers
            headers = EmailHeaders(
                message_id=msg.messageId,
                subject=msg.subject,
                date=msg.date,
            )

            # Parse From address
            if msg.sender:
                headers.from_addr = self._parse_address(msg.sender)

            # Parse To addresses
            if msg.to:
                headers.to_addrs = [self._parse_address(addr) for addr in msg.to.split(";")]

            # Parse CC addresses
            if msg.cc:
                headers.cc_addrs = [self._parse_address(addr) for addr in msg.cc.split(";")]

            # Extract body
            body = EmailBody(
                plain_text=msg.body,
                html=msg.htmlBody,
                content_type="multipart/alternative" if msg.htmlBody else "text/plain",
            )

            # Extract attachments
            attachments = []
            for attachment in msg.attachments:
                att = EmailAttachment(
                    filename=attachment.longFilename or attachment.shortFilename or "unnamed",
                    content_type=attachment.mimetype or "application/octet-stream",
                    size=len(attachment.data) if attachment.data else 0,
                    content=attachment.data,
                    is_inline=attachment.isInline,
                )
                attachments.append(att)

            return ParsedEmail(
                headers=headers,
                body=body,
                attachments=attachments,
                raw_size=Path(file_path).stat().st_size,
            )

        except ImportError:
            raise ImportError(
                "extract-msg library is required for MSG parsing. "
                "Install with: pip install extract-msg"
            )

    def _extract_headers(self, msg: email.message.EmailMessage) -> EmailHeaders:
        """Extract headers from an email message.
        
        Args:
            msg: Email message object
            
        Returns:
            EmailHeaders structure
        """
        headers = EmailHeaders()

        # Extract basic headers
        headers.message_id = msg.get("Message-ID")
        headers.subject = msg.get("Subject")
        headers.in_reply_to = msg.get("In-Reply-To")
        headers.thread_topic = msg.get("Thread-Topic")
        headers.thread_index = msg.get("Thread-Index")
        headers.priority = msg.get("X-Priority") or msg.get("Importance")
        headers.sensitivity = msg.get("Sensitivity")

        # Parse date
        date_str = msg.get("Date")
        if date_str:
            try:
                headers.date = parsedate_to_datetime(date_str)
            except Exception:
                pass

        # Parse addresses
        from_str = msg.get("From")
        if from_str:
            headers.from_addr = self._parse_address(from_str)

        to_str = msg.get("To")
        if to_str:
            headers.to_addrs = self._parse_address_list(to_str)

        cc_str = msg.get("Cc")
        if cc_str:
            headers.cc_addrs = self._parse_address_list(cc_str)

        bcc_str = msg.get("Bcc")
        if bcc_str:
            headers.bcc_addrs = self._parse_address_list(bcc_str)

        reply_to_str = msg.get("Reply-To")
        if reply_to_str:
            headers.reply_to = self._parse_address(reply_to_str)

        # Parse references
        refs_str = msg.get("References")
        if refs_str:
            headers.references = refs_str.split()

        return headers

    def _parse_address(self, addr_str: str) -> EmailAddress:
        """Parse a single email address string.
        
        Args:
            addr_str: Address string (e.g., "Name <email@example.com>")
            
        Returns:
            EmailAddress structure
        """
        addr_str = addr_str.strip()

        if "<" in addr_str and ">" in addr_str:
            # Format: "Display Name <email@example.com>"
            display_name = addr_str[:addr_str.index("<")].strip()
            address = addr_str[addr_str.index("<") + 1:addr_str.index(">")].strip()
            # Remove quotes from display name
            if display_name.startswith('"') and display_name.endswith('"'):
                display_name = display_name[1:-1]
            return EmailAddress(address=address, display_name=display_name or None)
        else:
            # Just the email address
            return EmailAddress(address=addr_str)

    def _parse_address_list(self, addrs_str: str) -> list[EmailAddress]:
        """Parse a comma-separated list of email addresses.
        
        Args:
            addrs_str: Comma-separated address string
            
        Returns:
            List of EmailAddress structures
        """
        addresses = []
        # Simple split - handles most common cases
        for addr in addrs_str.split(","):
            addr = addr.strip()
            if addr:
                addresses.append(self._parse_address(addr))
        return addresses

    def _extract_body(
        self,
        msg: email.message.EmailMessage,
        options: dict[str, Any],
    ) -> EmailBody:
        """Extract body content from an email message.
        
        Args:
            msg: Email message object
            options: Parsing options
            
        Returns:
            EmailBody structure
        """
        body = EmailBody()

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = part.get("Content-Disposition", "")

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                try:
                    if content_type == "text/plain" and not body.plain_text:
                        body.plain_text = self._get_text_content(part)
                        body.content_type = "text/plain"
                        body.charset = part.get_content_charset("utf-8")

                    elif content_type == "text/html" and not body.html:
                        body.html = self._get_text_content(part)
                        body.content_type = "text/html"
                        body.charset = part.get_content_charset("utf-8")

                        # Convert HTML to plain text if requested
                        if options.get("html_to_text", True) and not body.plain_text:
                            body.plain_text = self._html_to_text(body.html)

                except Exception as e:
                    logger.warning(f"Failed to extract body part: {e}")
        else:
            # Single part message
            content_type = msg.get_content_type()
            try:
                content = self._get_text_content(msg)
                body.charset = msg.get_content_charset("utf-8")

                if content_type == "text/html":
                    body.html = content
                    body.content_type = "text/html"
                    if options.get("html_to_text", True):
                        body.plain_text = self._html_to_text(content)
                else:
                    body.plain_text = content
                    body.content_type = "text/plain"
            except Exception as e:
                logger.warning(f"Failed to extract body: {e}")

        return body

    def _get_text_content(self, part: email.message.EmailMessage) -> str:
        """Get text content from a message part.
        
        Args:
            part: Email message part
            
        Returns:
            Decoded text content
        """
        charset = part.get_content_charset("utf-8")
        payload = part.get_payload(decode=True)

        if payload is None:
            return ""

        try:
            return payload.decode(charset, errors="replace")
        except Exception:
            return payload.decode("utf-8", errors="replace")

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text.
        
        Args:
            html: HTML content
            
        Returns:
            Plain text representation
        """
        try:
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self.skip_tags = {"script", "style"}
                    self.current_tag = None

                def handle_starttag(self, tag, attrs):
                    self.current_tag = tag
                    if tag == "br":
                        self.text.append("\n")
                    elif tag == "p":
                        self.text.append("\n\n")

                def handle_endtag(self, tag):
                    self.current_tag = None

                def handle_data(self, data):
                    if self.current_tag not in self.skip_tags:
                        self.text.append(data)

                def get_text(self):
                    return " ".join("".join(self.text).split())

            extractor = TextExtractor()
            extractor.feed(html)
            return extractor.get_text()

        except Exception as e:
            logger.warning(f"HTML to text conversion failed: {e}")
            # Fallback: strip tags
            import re
            text = re.sub(r"<[^>]+>", " ", html)
            return " ".join(text.split())

    def _extract_attachments(
        self,
        msg: email.message.EmailMessage,
        options: dict[str, Any],
    ) -> Tuple[list[EmailAttachment], list[EmailAttachment]]:
        """Extract attachments from an email message.
        
        Args:
            msg: Email message object
            options: Parsing options
            
        Returns:
            Tuple of (attachments, embedded_images)
        """
        attachments = []
        embedded = []

        if not msg.is_multipart():
            return attachments, embedded

        max_size = options.get("max_attachment_size", 50 * 1024 * 1024)
        extract_content = options.get("extract_attachments", True)

        for part in msg.walk():
            content_disposition = part.get("Content-Disposition", "")
            content_type = part.get_content_type()

            # Check if this is an attachment or embedded image
            is_attachment = "attachment" in content_disposition
            is_inline = "inline" in content_disposition

            if not is_attachment and not is_inline:
                # Check for common attachment content types
                if content_type not in ["text/plain", "text/html", "multipart/alternative",
                                        "multipart/mixed", "multipart/related"]:
                    is_attachment = True
                else:
                    continue

            # Get filename
            filename = part.get_filename()
            if not filename:
                # Generate filename from content type
                ext = mimetypes.guess_extension(content_type) or ".bin"
                filename = f"attachment_{len(attachments) + len(embedded)}{ext}"

            # Get content
            content = None
            size = 0

            if extract_content:
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        size = len(payload)
                        if size <= max_size:
                            content = payload
                        else:
                            logger.warning(f"Attachment {filename} exceeds max size ({size} bytes)")
                except Exception as e:
                    logger.warning(f"Failed to extract attachment {filename}: {e}")

            # Get content ID for embedded images
            content_id = part.get("Content-ID", "").strip("<>")

            att = EmailAttachment(
                filename=filename,
                content_type=content_type,
                size=size,
                content_id=content_id or None,
                content=content,
                is_inline=is_inline,
            )

            if is_inline and content_type.startswith("image/"):
                embedded.append(att)
            else:
                attachments.append(att)

        return attachments, embedded

    def _create_result(
        self,
        parsed: ParsedEmail,
        path: Path,
        options: dict[str, Any],
    ) -> ParsingResult:
        """Create a ParsingResult from parsed email.
        
        Args:
            parsed: ParsedEmail structure
            path: File path
            options: Parsing options
            
        Returns:
            ParsingResult
        """
        headers = parsed.headers
        body = parsed.body

        # Generate full text
        lines = []

        if options.get("include_headers", True):
            lines.append("=== Email Headers ===")
            lines.append(f"From: {headers.from_addr}")
            lines.append(f"To: {', '.join(str(a) for a in headers.to_addrs)}")
            if headers.cc_addrs:
                lines.append(f"Cc: {', '.join(str(a) for a in headers.cc_addrs)}")
            lines.append(f"Subject: {headers.subject}")
            if headers.date:
                lines.append(f"Date: {headers.date.isoformat()}")
            if headers.message_id:
                lines.append(f"Message-ID: {headers.message_id}")
            if headers.in_reply_to:
                lines.append(f"In-Reply-To: {headers.in_reply_to}")
            if headers.references:
                lines.append(f"References: {' '.join(headers.references)}")
            lines.append("")

        lines.append("=== Email Body ===")
        lines.append("")

        # Use plain text if available, otherwise HTML
        body_text = body.plain_text or body.html or ""
        lines.append(body_text)

        if parsed.attachments:
            lines.append("")
            lines.append("=== Attachments ===")
            for att in parsed.attachments:
                lines.append(f"- {att.filename} ({att.content_type}, {att.size} bytes)")

        full_text = "\n".join(lines)

        # Create chunks
        chunks = []
        header_text = "\n".join(lines[:lines.index("=== Email Body ===")]) if "=== Email Body ===" in lines else ""
        if header_text:
            chunks.append(header_text)

        # Split body into chunks if large
        if body_text:
            chunk_size = 5000
            for i in range(0, len(body_text), chunk_size):
                chunks.append(body_text[i:i + chunk_size])

        # Metadata
        metadata = {
            "message_id": headers.message_id,
            "subject": headers.subject,
            "from": str(headers.from_addr) if headers.from_addr else None,
            "to": [str(a) for a in headers.to_addrs],
            "cc": [str(a) for a in headers.cc_addrs],
            "date": headers.date.isoformat() if headers.date else None,
            "in_reply_to": headers.in_reply_to,
            "references": headers.references,
            "thread_topic": headers.thread_topic,
            "attachment_count": len(parsed.attachments),
            "has_html": body.html is not None,
            "raw_size": parsed.raw_size,
        }

        # Attachment metadata
        if parsed.attachments:
            metadata["attachments"] = [
                {
                    "filename": att.filename,
                    "content_type": att.content_type,
                    "size": att.size,
                    "content_id": att.content_id,
                }
                for att in parsed.attachments
            ]

        # Calculate confidence
        confidence = 0.95
        if not body_text:
            confidence = 0.6
        if parsed.attachments and not any(att.content for att in parsed.attachments):
            confidence -= 0.1

        return ParsingResult(
            success=True,
            text=full_text,
            pages=chunks or [full_text],
            metadata=metadata,
            format=path.suffix.lower(),
            parser_used="email",
            confidence=max(0.5, confidence),
            attachments=[
                {
                    "filename": att.filename,
                    "content_type": att.content_type,
                    "size": att.size,
                }
                for att in parsed.attachments
            ],
        )

    async def health_check(self, config: dict[str, Any] | None = None) -> HealthStatus:
        """Check the health of the parser.
        
        Args:
            config: Optional configuration for health check
            
        Returns:
            HealthStatus indicating parser health
        """
        # EML parsing uses standard library
        try:
            import extract_msg
            return HealthStatus.HEALTHY
        except ImportError:
            # MSG support unavailable but EML still works
            return HealthStatus.DEGRADED
