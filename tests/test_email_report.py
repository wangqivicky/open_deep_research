"""Tests for final report email delivery."""

import asyncio

from open_deep_research import deep_researcher
from open_deep_research.configuration import Configuration
from langchain_core.messages import AIMessage

from open_deep_research.email_report import (
    _markdown_to_responsive_html,
    normalize_report_text,
    send_report_email,
)


class FakeSMTP:
    """Capture SMTP interactions without sending network traffic."""

    instance = None

    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.logged_in = None
        self.message = None
        self.to_addrs = None
        self.started_tls = False
        FakeSMTP.instance = self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def ehlo(self):
        """Record a no-op SMTP greeting."""

    def starttls(self):
        """Record that STARTTLS was requested."""
        self.started_tls = True

    def login(self, username, password):
        """Record SMTP authentication."""
        self.logged_in = (username, password)

    def send_message(self, message, to_addrs):
        """Capture the outgoing message."""
        self.message = message
        self.to_addrs = to_addrs


def test_send_report_email_uses_starttls(monkeypatch):
    """Send the report body and support multiple recipients."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "sender@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.setenv("SMTP_SECURITY", "starttls")
    monkeypatch.setattr("open_deep_research.email_report.smtplib.SMTP", FakeSMTP)

    send_report_email(
        "# Final report\n\nResearch findings.",
        "one@example.com, two@example.com",
        "Research complete",
    )

    smtp = FakeSMTP.instance
    assert smtp.started_tls is True
    assert smtp.logged_in == ("sender@example.com", "secret")
    assert smtp.to_addrs == ["one@example.com", "two@example.com"]
    assert smtp.message["Subject"] == "Final report"
    assert smtp.message.is_multipart()
    plain_part, html_part = smtp.message.get_payload()
    assert plain_part.get_content_type() == "text/plain"
    assert plain_part.get_content().strip() == "# Final report\n\nResearch findings."
    assert html_part.get_content_type() == "text/html"
    html_body = html_part.get_content()
    assert "<h1>Final report</h1>" in html_body
    assert "@media only screen and (max-width: 600px)" in html_body
    assert ".email-shell { padding: 0 !important; }" in html_body
    assert "padding: 14px 12px !important" in html_body


def test_normalize_report_text_removes_message_metadata():
    """Extract content without serializing AIMessage IDs and type fields."""
    message = AIMessage(
        content="# 字节跳动企业业务与企业文化研究报告\n\n正文",
        id="message-id-that-must-not-be-sent",
    )

    report = normalize_report_text(message)

    assert report == "# 字节跳动企业业务与企业文化研究报告\n\n正文"
    assert "message-id-that-must-not-be-sent" not in report
    assert "type=" not in report


def test_normalize_report_text_extracts_responses_api_content_blocks():
    """Flatten Responses API text blocks into clean Markdown."""
    report = normalize_report_text(
        [
            {"type": "text", "text": "# 研究报告"},
            {"type": "text", "text": "## 核心结论\n\n正文。"},
        ]
    )

    assert report == "# 研究报告\n\n## 核心结论\n\n正文。"
    assert "'type': 'text'" not in report


def test_html_email_renders_markdown_pipe_tables():
    """Render GFM-style pipe tables as rows instead of one collapsed paragraph."""
    report = """# 研究报告

| 层次 | 代表业务 | 主要价值 |
|---|---|---|
| 内容平台 | 抖音 | 聚集用户 |
| 企业服务 | 飞书 | 企业协作 |
"""

    rendered = _markdown_to_responsive_html(report, "研究报告")

    assert "<table>" in rendered
    assert "<th>层次</th>" in rendered
    assert "<td>内容平台</td>" in rendered
    assert rendered.count("<tr>") == 3


def test_email_configuration_can_be_loaded_from_environment(monkeypatch):
    """Load optional report delivery settings from the environment."""
    monkeypatch.setenv("EMAIL_REPORT_ENABLED", "true")
    monkeypatch.setenv("EMAIL_REPORT_TO", "recipient@example.com")

    configuration = Configuration.from_runnable_config()

    assert configuration.email_report_enabled is True
    assert configuration.email_report_to == "recipient@example.com"


def test_email_node_reports_failure_without_raising(monkeypatch):
    """Return a delivery status instead of failing the research graph."""
    async def fail_to_send(**kwargs):
        raise OSError("SMTP unavailable")

    monkeypatch.setattr(deep_researcher, "send_report_email_async", fail_to_send)

    result = asyncio.run(
        deep_researcher.send_report_email(
            {"final_report": "# Completed report"},
            {
                "configurable": {
                    "email_report_enabled": True,
                    "email_report_to": "recipient@example.com",
                }
            },
        )
    )

    assert result["email_delivery_status"] == "failed: SMTP unavailable"


def test_email_node_passes_only_message_content_and_uses_report_title(monkeypatch):
    """Send only report content and derive the subject from its H1 heading."""
    captured = {}

    async def capture_email(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(deep_researcher, "send_report_email_async", capture_email)
    report_message = AIMessage(
        content="# 字节跳动企业业务与企业文化研究报告\n\n报告正文。",
        id="hidden-message-id",
    )

    result = asyncio.run(
        deep_researcher.send_report_email(
            {"final_report": report_message},
            {
                "configurable": {
                    "email_report_enabled": True,
                    "email_report_to": "recipient@example.com",
                }
            },
        )
    )

    assert result["email_delivery_status"] == "sent"
    assert captured["report"] == "# 字节跳动企业业务与企业文化研究报告\n\n报告正文。"
    assert "hidden-message-id" not in captured["report"]


def test_email_delivery_is_a_graph_node():
    """Expose email delivery as a distinct LangGraph node."""
    assert "send_report_email" in deep_researcher.deep_researcher.get_graph().nodes
