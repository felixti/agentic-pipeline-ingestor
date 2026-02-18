"""Unit tests for email parser plugin."""

import email
import email.policy
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.plugins.base import HealthStatus, ParsingResult, SupportResult
from src.plugins.parsers.email_parser import (
    EmailAddress,
    EmailAttachment,
    EmailBody,
    EmailHeaders,
    EmailParser,
    ParsedEmail,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def parser():
    """Create an initialized email parser."""
    p = EmailParser()
    await p.initialize({})
    return p


@pytest.fixture
def sample_eml_file(tmp_path):
    """Create a sample EML file."""
    eml_file = tmp_path / "test.eml"
    eml_content = """From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 15 Jan 2024 10:30:00 +0000
Message-ID: <test123@example.com>
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

This is a test email body.
"""
    eml_file.write_text(eml_content)
    return eml_file


# ============================================================================
# EmailParser Class Tests
# ============================================================================

@pytest.mark.unit
class TestEmailParser:
    """Tests for EmailParser class."""

    def test_init(self):
        """Test parser initialization."""
        parser = EmailParser()
        assert parser._config == {}

    def test_metadata(self):
        """Test parser metadata."""
        parser = EmailParser()
        metadata = parser.metadata

        assert metadata.id == "email"
        assert metadata.name == "Email Parser"
        assert metadata.version == "1.0.0"
        assert ".eml" in metadata.supported_formats
        assert ".msg" in metadata.supported_formats

    @pytest.mark.asyncio
    async def test_initialize_default_config(self):
        """Test initialization with default config."""
        parser = EmailParser()
        await parser.initialize({})

        assert parser._config["encoding"] == "utf-8"
        assert parser._config["extract_attachments"] is True
        assert parser._config["max_attachment_size"] == 50 * 1024 * 1024
        assert parser._config["save_attachments"] is False
        assert parser._config["attachment_path"] == "./attachments"
        assert parser._config["html_to_text"] is True
        assert parser._config["include_headers"] is True

    @pytest.mark.asyncio
    async def test_initialize_custom_config(self):
        """Test initialization with custom config."""
        parser = EmailParser()
        await parser.initialize({
            "encoding": "latin-1",
            "extract_attachments": False,
            "max_attachment_size": 10 * 1024 * 1024,
            "save_attachments": True,
            "attachment_path": "/custom/attachments",
            "html_to_text": False,
            "include_headers": False,
        })

        assert parser._config["encoding"] == "latin-1"
        assert parser._config["extract_attachments"] is False
        assert parser._config["max_attachment_size"] == 10 * 1024 * 1024
        assert parser._config["save_attachments"] is True
        assert parser._config["attachment_path"] == "/custom/attachments"
        assert parser._config["html_to_text"] is False
        assert parser._config["include_headers"] is False


# ============================================================================
# Supports Method Tests
# ============================================================================

@pytest.mark.unit
class TestEmailParserSupports:
    """Tests for email parser supports method."""

    @pytest.mark.asyncio
    async def test_supports_eml_extension(self):
        """Test support for .eml files."""
        parser = EmailParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file.eml")

        assert isinstance(result, SupportResult)
        assert result.supported is True
        assert result.confidence == 0.95
        assert "EML" in result.reason

    @pytest.mark.asyncio
    async def test_supports_msg_extension(self):
        """Test support for .msg files."""
        parser = EmailParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file.msg")

        assert result.supported is True
        assert result.confidence == 0.90
        assert "MSG" in result.reason

    @pytest.mark.asyncio
    async def test_supports_mime_type_rfc822(self):
        """Test support with message/rfc822 MIME type."""
        parser = EmailParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file", mime_type="message/rfc822")

        assert result.supported is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_supports_mime_type_outlook(self):
        """Test support with Outlook MIME type."""
        parser = EmailParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file", mime_type="application/vnd.ms-outlook")

        assert result.supported is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_supports_unsupported_extension(self):
        """Test rejection of unsupported file extensions."""
        parser = EmailParser()
        await parser.initialize({})

        result = await parser.supports("/path/to/file.pdf")

        assert result.supported is False
        assert result.confidence == 1.0


# ============================================================================
# Parse Simple Email Tests
# ============================================================================

@pytest.mark.unit
class TestEmailParserParseSimple:
    """Tests for parsing simple emails."""

    @pytest.mark.asyncio
    async def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        parser = EmailParser()
        await parser.initialize({})

        result = await parser.parse("/nonexistent/file.eml")

        assert isinstance(result, ParsingResult)
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_parse_unsupported_format(self, tmp_path):
        """Test parsing unsupported file format."""
        parser = EmailParser()
        await parser.initialize({})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("not a pdf")

        result = await parser.parse(str(pdf_file))

        assert result.success is False

    @pytest.mark.asyncio
    async def test_parse_simple_eml(self, tmp_path):
        """Test parsing simple EML file."""
        parser = EmailParser()
        await parser.initialize({})

        eml_file = tmp_path / "test.eml"
        eml_content = """From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 15 Jan 2024 10:30:00 +0000
Message-ID: <test123@example.com>
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

This is a test email body.
"""
        eml_file.write_text(eml_content)

        result = await parser.parse(str(eml_file))

        assert result.success is True
        assert "Test Email" in result.text
        assert "sender@example.com" in result.text
        assert "recipient@example.com" in result.text
        assert "This is a test email body" in result.text
        assert result.metadata["subject"] == "Test Email"
        assert result.metadata["message_id"] == "<test123@example.com>"

    @pytest.mark.asyncio
    async def test_parse_eml_with_multiple_recipients(self, tmp_path):
        """Test parsing EML with multiple recipients."""
        parser = EmailParser()
        await parser.initialize({})

        eml_file = tmp_path / "test.eml"
        eml_content = """From: sender@example.com
To: recipient1@example.com, recipient2@example.com
Cc: cc@example.com
Subject: Multi-recipient Test
Date: Mon, 15 Jan 2024 10:30:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

Test body.
"""
        eml_file.write_text(eml_content)

        result = await parser.parse(str(eml_file))

        assert result.success is True
        assert "recipient1@example.com" in result.text
        assert "recipient2@example.com" in result.text
        assert "cc@example.com" in result.text


# ============================================================================
# Parse Email with Attachments Tests
# ============================================================================

@pytest.mark.unit
class TestEmailParserParseAttachments:
    """Tests for parsing emails with attachments."""

    @pytest.mark.asyncio
    async def test_parse_eml_with_attachment(self, tmp_path):
        """Test parsing EML with attachment."""
        parser = EmailParser()
        await parser.initialize({})

        eml_file = tmp_path / "test.eml"
        # Create a multipart email with attachment
        msg = email.message.EmailMessage(policy=email.policy.default)
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "Email with Attachment"
        msg["Date"] = "Mon, 15 Jan 2024 10:30:00 +0000"
        msg["Message-ID"] = "<test456@example.com>"

        # Add body
        msg.set_content("This is the email body.")

        # Add attachment
        msg.add_attachment(
            b"PDF content here",
            maintype="application",
            subtype="pdf",
            filename="document.pdf"
        )

        eml_file.write_bytes(msg.as_bytes())

        result = await parser.parse(str(eml_file))

        assert result.success is True
        assert result.metadata["attachment_count"] == 1
        assert len(result.metadata["attachments"]) == 1
        assert result.metadata["attachments"][0]["filename"] == "document.pdf"
        assert result.metadata["attachments"][0]["content_type"] == "application/pdf"

    @pytest.mark.asyncio
    async def test_parse_eml_with_multiple_attachments(self, tmp_path):
        """Test parsing EML with multiple attachments."""
        parser = EmailParser()
        await parser.initialize({})

        eml_file = tmp_path / "test.eml"
        msg = email.message.EmailMessage(policy=email.policy.default)
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "Multiple Attachments"
        msg["Date"] = "Mon, 15 Jan 2024 10:30:00 +0000"

        msg.set_content("Body with multiple attachments.")
        msg.add_attachment(b"Content 1", maintype="text", subtype="plain", filename="file1.txt")
        msg.add_attachment(b"Content 2", maintype="text", subtype="plain", filename="file2.txt")

        eml_file.write_bytes(msg.as_bytes())

        result = await parser.parse(str(eml_file))

        assert result.success is True
        assert result.metadata["attachment_count"] == 2
        assert len(result.metadata["attachments"]) == 2

    @pytest.mark.asyncio
    async def test_parse_eml_with_embedded_image(self, tmp_path):
        """Test parsing EML with embedded image."""
        parser = EmailParser()
        await parser.initialize({})

        eml_file = tmp_path / "test.eml"
        msg = email.message.EmailMessage(policy=email.policy.default)
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "Email with Image"
        msg["Date"] = "Mon, 15 Jan 2024 10:30:00 +0000"

        msg.set_content("See the attached image.")
        msg.add_attachment(
            b"fake image data",
            maintype="image",
            subtype="png",
            filename="image.png",
            cid="<image001>"
        )

        eml_file.write_bytes(msg.as_bytes())

        result = await parser.parse(str(eml_file))

        assert result.success is True
        # Image should be in attachments list
        assert any(att.get("content_type", "").startswith("image/") 
                   for att in result.metadata.get("attachments", []))


# ============================================================================
# Parse Email with HTML Body Tests
# ============================================================================

@pytest.mark.unit
class TestEmailParserParseHTML:
    """Tests for parsing emails with HTML bodies."""

    @pytest.mark.asyncio
    async def test_parse_eml_with_html_body(self, tmp_path):
        """Test parsing EML with HTML body."""
        parser = EmailParser()
        await parser.initialize({"html_to_text": True})

        eml_file = tmp_path / "test.eml"
        msg = email.message.EmailMessage(policy=email.policy.default)
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "HTML Email"
        msg["Date"] = "Mon, 15 Jan 2024 10:30:00 +0000"

        html_content = """<html>
<body>
    <h1>Hello World</h1>
    <p>This is a <b>bold</b> paragraph.</p>
</body>
</html>"""
        msg.set_content(html_content, subtype="html")

        eml_file.write_bytes(msg.as_bytes())

        result = await parser.parse(str(eml_file))

        assert result.success is True
        assert result.metadata["has_html"] is True
        assert "Hello World" in result.text

    @pytest.mark.asyncio
    async def test_parse_eml_with_multipart_alternative(self, tmp_path):
        """Test parsing EML with both text and HTML parts."""
        parser = EmailParser()
        await parser.initialize({"html_to_text": True})

        eml_file = tmp_path / "test.eml"
        msg = email.message.EmailMessage(policy=email.policy.default)
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "Multipart Email"
        msg["Date"] = "Mon, 15 Jan 2024 10:30:00 +0000"

        # Add plain text version
        msg.set_content("Plain text version")
        # Add HTML version
        msg.add_alternative(
            "<html><body><h1>HTML Version</h1></body></html>",
            subtype="html"
        )

        eml_file.write_bytes(msg.as_bytes())

        result = await parser.parse(str(eml_file))

        assert result.success is True
        assert result.metadata["has_html"] is True

    @pytest.mark.asyncio
    async def test_html_to_text_conversion(self):
        """Test HTML to text conversion."""
        parser = EmailParser()
        html = "<html><body><h1>Title</h1><p>Paragraph with <b>bold</b> text.</p></body></html>"

        text = parser._html_to_text(html)

        assert "Title" in text
        assert "Paragraph" in text
        assert "bold" in text

    def test_html_to_text_with_br(self):
        """Test HTML to text with line breaks."""
        parser = EmailParser()
        html = "Line 1<br/>Line 2<br>Line 3"

        text = parser._html_to_text(html)

        assert "Line 1" in text
        assert "Line 2" in text
        assert "Line 3" in text

    def test_html_to_text_with_paragraphs(self):
        """Test HTML to text with paragraphs."""
        parser = EmailParser()
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"

        text = parser._html_to_text(html)

        assert "First paragraph" in text
        assert "Second paragraph" in text

    def test_html_to_text_strips_script(self):
        """Test HTML to text strips script tags."""
        parser = EmailParser()
        html = "<p>Text</p><script>alert('xss')</script><p>More text</p>"

        text = parser._html_to_text(html)

        assert "Text" in text
        assert "More text" in text
        assert "alert" not in text


# ============================================================================
# Parse Email with Text Body Tests
# ============================================================================

@pytest.mark.unit
class TestEmailParserParseText:
    """Tests for parsing emails with text bodies."""

    @pytest.mark.asyncio
    async def test_parse_eml_with_text_body(self, tmp_path):
        """Test parsing EML with plain text body."""
        parser = EmailParser()
        await parser.initialize({})

        eml_file = tmp_path / "test.eml"
        msg = email.message.EmailMessage(policy=email.policy.default)
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "Text Email"
        msg["Date"] = "Mon, 15 Jan 2024 10:30:00 +0000"

        msg.set_content("This is plain text content.")

        eml_file.write_bytes(msg.as_bytes())

        result = await parser.parse(str(eml_file))

        assert result.success is True
        assert result.metadata["has_html"] is False
        assert "This is plain text content" in result.text

    @pytest.mark.asyncio
    async def test_parse_eml_without_headers(self, tmp_path):
        """Test parsing EML without including headers."""
        parser = EmailParser()
        await parser.initialize({"include_headers": False})

        eml_file = tmp_path / "test.eml"
        msg = email.message.EmailMessage(policy=email.policy.default)
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "No Headers"
        msg["Date"] = "Mon, 15 Jan 2024 10:30:00 +0000"

        msg.set_content("Just the body.")

        eml_file.write_bytes(msg.as_bytes())

        result = await parser.parse(str(eml_file))

        assert result.success is True
        # Should not contain header section
        assert "=== Email Headers ===" not in result.text


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestEmailParserErrorHandling:
    """Tests for email parser error handling."""

    @pytest.mark.asyncio
    async def test_parse_invalid_email(self, tmp_path):
        """Test parsing invalid email content."""
        parser = EmailParser()
        await parser.initialize({})

        eml_file = tmp_path / "test.eml"
        # Write invalid email content
        eml_file.write_text("This is not a valid email format")

        # The parser should handle this gracefully
        result = await parser.parse(str(eml_file))

        # The parser might still succeed (email library is forgiving) or fail gracefully
        if result.success:
            # If successful, the result should contain the raw content
            assert "This is not a valid email format" in result.text

    @pytest.mark.asyncio
    async def test_parse_empty_file(self, tmp_path):
        """Test parsing empty file."""
        parser = EmailParser()
        await parser.initialize({})

        eml_file = tmp_path / "test.eml"
        eml_file.write_text("")

        result = await parser.parse(str(eml_file))

        # Empty email might parse successfully but have no content
        assert isinstance(result, ParsingResult)

    @pytest.mark.asyncio
    async def test_parse_msg_without_extract_msg_library(self, tmp_path):
        """Test parsing MSG file without extract-msg library."""
        parser = EmailParser()
        await parser.initialize({})

        msg_file = tmp_path / "test.msg"
        msg_file.write_text("fake msg content")

        with patch.dict("sys.modules", {"extract_msg": None}):
            result = await parser.parse(str(msg_file))

        # Should fail because extract-msg is not available
        assert result.success is False
        assert "extract-msg" in result.error.lower() or "parsing failed" in result.error.lower()


# ============================================================================
# Address Parsing Tests
# ============================================================================

@pytest.mark.unit
class TestEmailParserAddressParsing:
    """Tests for email address parsing."""

    def test_parse_address_with_display_name(self):
        """Test parsing address with display name."""
        parser = EmailParser()
        addr = parser._parse_address("John Doe <john@example.com>")

        assert addr.address == "john@example.com"
        assert addr.display_name == "John Doe"

    def test_parse_address_with_quotes(self):
        """Test parsing address with quoted display name."""
        parser = EmailParser()
        addr = parser._parse_address('"John Doe" <john@example.com>')

        assert addr.address == "john@example.com"
        assert addr.display_name == "John Doe"

    def test_parse_address_simple(self):
        """Test parsing simple email address."""
        parser = EmailParser()
        addr = parser._parse_address("john@example.com")

        assert addr.address == "john@example.com"
        assert addr.display_name is None

    def test_parse_address_list(self):
        """Test parsing list of email addresses."""
        parser = EmailParser()
        addrs = parser._parse_address_list("a@example.com, b@example.com, c@example.com")

        assert len(addrs) == 3
        assert addrs[0].address == "a@example.com"
        assert addrs[1].address == "b@example.com"
        assert addrs[2].address == "c@example.com"

    def test_parse_address_list_with_display_names(self):
        """Test parsing address list with display names."""
        parser = EmailParser()
        addrs = parser._parse_address_list(
            "User A <a@example.com>, User B <b@example.com>"
        )

        assert len(addrs) == 2
        assert addrs[0].address == "a@example.com"
        assert addrs[0].display_name == "User A"
        assert addrs[1].address == "b@example.com"
        assert addrs[1].display_name == "User B"

    def test_email_address_str(self):
        """Test EmailAddress string representation."""
        addr_with_name = EmailAddress("test@example.com", "Test User")
        addr_without_name = EmailAddress("test@example.com")

        assert str(addr_with_name) == "Test User <test@example.com>"
        assert str(addr_without_name) == "test@example.com"


# ============================================================================
# MSG Parsing Tests (Mocked)
# ============================================================================

@pytest.mark.unit
class TestEmailParserMSGMocked:
    """Tests for MSG parsing with mocked extract_msg."""

    @pytest.mark.asyncio
    async def test_parse_msg_with_mock(self, tmp_path):
        """Test parsing MSG file with mocked extract_msg library."""
        parser = EmailParser()
        await parser.initialize({})

        msg_file = tmp_path / "test.msg"
        msg_file.write_text("fake msg content")

        # Mock the extract_msg module
        mock_msg = MagicMock()
        mock_msg.messageId = "<msg123@example.com>"
        mock_msg.subject = "Test MSG"
        mock_msg.date = datetime(2024, 1, 15, 10, 30, 0)
        mock_msg.sender = "sender@example.com"
        mock_msg.to = "recipient@example.com"
        mock_msg.cc = "cc@example.com"
        mock_msg.body = "MSG body content"
        mock_msg.htmlBody = None

        mock_attachment = MagicMock()
        mock_attachment.longFilename = "attachment.txt"
        mock_attachment.shortFilename = "attach.txt"
        mock_attachment.mimetype = "text/plain"
        mock_attachment.data = b"attachment content"
        mock_attachment.isInline = False
        mock_msg.attachments = [mock_attachment]

        mock_extract_msg = MagicMock()
        mock_extract_msg.Message.return_value = mock_msg

        with patch.dict("sys.modules", {"extract_msg": mock_extract_msg}):
            result = await parser.parse(str(msg_file))

        assert result.success is True
        assert "Test MSG" in result.text
        assert "MSG body content" in result.text


# ============================================================================
# Header Extraction Tests
# ============================================================================

@pytest.mark.unit
class TestEmailParserHeaderExtraction:
    """Tests for email header extraction."""

    def test_extract_headers_basic(self):
        """Test extracting basic headers."""
        parser = EmailParser()
        msg = email.message.EmailMessage(policy=email.policy.default)
        msg["Message-ID"] = "<test@example.com>"
        msg["Subject"] = "Test Subject"
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Date"] = "Mon, 15 Jan 2024 10:30:00 +0000"

        headers = parser._extract_headers(msg)

        assert headers.message_id == "<test@example.com>"
        assert headers.subject == "Test Subject"
        assert headers.from_addr.address == "sender@example.com"
        assert len(headers.to_addrs) == 1
        assert headers.to_addrs[0].address == "recipient@example.com"
        assert headers.date is not None

    def test_extract_headers_threading(self):
        """Test extracting threading headers."""
        parser = EmailParser()
        msg = email.message.EmailMessage(policy=email.policy.default)
        msg["In-Reply-To"] = "<parent@example.com>"
        msg["References"] = "<ref1@example.com> <ref2@example.com>"
        msg["Thread-Topic"] = "Discussion"
        msg["Thread-Index"] = "abc123"

        headers = parser._extract_headers(msg)

        assert headers.in_reply_to == "<parent@example.com>"
        assert headers.references == ["<ref1@example.com>", "<ref2@example.com>"]
        assert headers.thread_topic == "Discussion"
        assert headers.thread_index == "abc123"


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.unit
class TestEmailParserHealthCheck:
    """Tests for email parser health check."""

    @pytest.mark.asyncio
    async def test_health_check_with_extract_msg(self):
        """Test health check when extract_msg is available."""
        parser = EmailParser()
        await parser.initialize({})

        with patch.dict("sys.modules", {"extract_msg": MagicMock()}):
            health = await parser.health_check()

        assert health == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_without_extract_msg(self):
        """Test health check when extract_msg is not available."""
        parser = EmailParser()
        await parser.initialize({})

        with patch.dict("sys.modules", {"extract_msg": None}):
            health = await parser.health_check()

        assert health == HealthStatus.DEGRADED


# ============================================================================
# Options Override Tests
# ============================================================================

@pytest.mark.unit
class TestEmailParserOptionsOverride:
    """Tests for parse options override."""

    @pytest.mark.asyncio
    async def test_parse_with_options_override(self, tmp_path):
        """Test parsing with options override."""
        parser = EmailParser()
        await parser.initialize({"include_headers": True})

        eml_file = tmp_path / "test.eml"
        msg = email.message.EmailMessage(policy=email.policy.default)
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "Override Test"
        msg["Date"] = "Mon, 15 Jan 2024 10:30:00 +0000"
        msg.set_content("Body content.")

        eml_file.write_bytes(msg.as_bytes())

        # Override include_headers option
        result = await parser.parse(str(eml_file), options={"include_headers": False})

        assert result.success is True
