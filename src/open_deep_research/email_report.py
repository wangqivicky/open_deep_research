"""Email delivery for generated research reports."""

import asyncio
import html
import os
import re
import smtplib
from email.message import EmailMessage
from email.utils import getaddresses

from markdown_it import MarkdownIt


def normalize_report_text(report) -> str:
    """Extract only human-readable report text from strings or message objects."""
    if isinstance(report, str):
        return report.strip()

    content = getattr(report, "content", None)
    if content is not None:
        return normalize_report_text(content)

    if isinstance(report, dict):
        for key in ("final_report", "content", "text", "output_text"):
            if key in report:
                text = normalize_report_text(report[key])
                if text:
                    return text
        return ""

    if isinstance(report, (list, tuple)):
        parts = [normalize_report_text(item) for item in report]
        return "\n\n".join(part for part in parts if part)

    return ""


def extract_report_title(report: str) -> str | None:
    """Use the first Markdown H1 as a safe email subject."""
    match = re.search(r"^\s*#\s+(.+?)\s*$", report, flags=re.MULTILINE)
    if not match:
        return None

    title = match.group(1)
    title = re.sub(r"!?(?:\[([^]]+)\])\([^)]*\)", r"\1", title)
    title = re.sub(r"[`*_~]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title[:200] or None


def _markdown_to_responsive_html(report: str, title: str) -> str:
    """Render Markdown as a responsive HTML email with minimal mobile margins."""
    # CommonMark does not enable pipe tables by default. Enable the Markdown-It
    # table rule so each `| ... |` source row becomes a real HTML table row.
    markdown = MarkdownIt("commonmark", {"html": False}).enable("table")
    body = markdown.render(report)
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{safe_title}</title>
  <style>
    html, body {{ margin: 0 !important; padding: 0 !important; width: 100% !important; background: #f5f7fa; }}
    .email-shell {{ width: 100%; box-sizing: border-box; padding: 20px 12px; }}
    .report {{ max-width: 900px; margin: 0 auto; box-sizing: border-box; padding: 28px 32px; background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; color: #1f2937; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", Arial, sans-serif; font-size: 16px; line-height: 1.75; overflow-wrap: anywhere; }}
    .report h1 {{ margin: 0 0 24px; font-size: 28px; line-height: 1.35; color: #111827; }}
    .report h2 {{ margin: 28px 0 12px; font-size: 22px; line-height: 1.4; color: #111827; }}
    .report h3 {{ margin: 22px 0 10px; font-size: 18px; line-height: 1.45; color: #111827; }}
    .report p {{ margin: 0 0 14px; }}
    .report ul, .report ol {{ margin: 0 0 16px; padding-left: 24px; }}
    .report li {{ margin: 6px 0; }}
    .report a {{ color: #2563eb; text-decoration: none; word-break: break-all; }}
    .report blockquote {{ margin: 16px 0; padding: 8px 14px; border-left: 4px solid #cbd5e1; background: #f8fafc; color: #475569; }}
    .report table {{ display: block; width: 100%; max-width: 100%; margin: 18px 0; overflow-x: auto; border-collapse: collapse; border-spacing: 0; -webkit-overflow-scrolling: touch; }}
    .report thead {{ background: #f3f4f6; }}
    .report th, .report td {{ min-width: 110px; padding: 9px 10px; border: 1px solid #d1d5db; text-align: left; vertical-align: top; }}
    .report th {{ color: #111827; font-weight: 600; }}
    .report pre {{ max-width: 100%; overflow-x: auto; padding: 12px; background: #f3f4f6; border-radius: 6px; }}
    .report code {{ font-family: Consolas, Monaco, monospace; word-break: break-word; }}
    @media only screen and (max-width: 600px) {{
      .email-shell {{ padding: 0 !important; }}
      .report {{ width: 100% !important; max-width: none !important; margin: 0 !important; padding: 14px 12px !important; border-left: 0 !important; border-right: 0 !important; border-radius: 0 !important; font-size: 15px !important; line-height: 1.7 !important; }}
      .report h1 {{ font-size: 23px !important; margin-bottom: 18px !important; }}
      .report h2 {{ font-size: 19px !important; margin-top: 24px !important; }}
      .report h3 {{ font-size: 17px !important; }}
      .report ul, .report ol {{ padding-left: 21px !important; }}
      .report th, .report td {{ min-width: 96px !important; padding: 7px 8px !important; font-size: 14px !important; }}
    }}
  </style>
</head>
<body>
  <div class="email-shell">
    <article class="report">
      {body}
    </article>
  </div>
</body>
</html>"""


def _get_recipients(value: str) -> list[str]:
    """Parse and validate a comma-separated recipient list."""
    recipients = [
        address
        for _, address in getaddresses([value])
        if "@" in address and not address.startswith("@") and not address.endswith("@")
    ]
    if not recipients:
        raise ValueError("EMAIL_REPORT_TO does not contain a valid email address")
    return recipients


def send_report_email(report: str, recipient: str, subject: str) -> None:
    """Send a report using SMTP settings from environment variables."""
    report_text = normalize_report_text(report)
    if not report_text:
        raise ValueError("The final report does not contain readable text")

    report_title = extract_report_title(report_text)
    email_subject = report_title or subject
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_username = os.environ.get("SMTP_USERNAME")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    smtp_sender = os.environ.get("SMTP_FROM") or smtp_username
    smtp_security = os.environ.get("SMTP_SECURITY", "starttls").lower()

    if not smtp_host:
        raise ValueError("SMTP_HOST is required when email delivery is enabled")
    if not smtp_sender:
        raise ValueError("SMTP_FROM or SMTP_USERNAME is required")
    if bool(smtp_username) != bool(smtp_password):
        raise ValueError("SMTP_USERNAME and SMTP_PASSWORD must be provided together")
    if smtp_security not in {"starttls", "ssl", "none"}:
        raise ValueError("SMTP_SECURITY must be one of: starttls, ssl, none")

    default_ports = {"ssl": 465, "starttls": 587, "none": 25}
    default_port = default_ports[smtp_security]
    smtp_port = int(os.environ.get("SMTP_PORT", default_port))
    smtp_timeout = float(os.environ.get("SMTP_TIMEOUT", "30"))
    recipients = _get_recipients(recipient)

    message = EmailMessage()
    message["From"] = smtp_sender
    message["To"] = ", ".join(recipients)
    message["Subject"] = email_subject
    # Keep a complete plain-text fallback, then provide responsive HTML as the
    # preferred body for clients that support formatted email.
    message.set_content(report_text, subtype="plain", charset="utf-8")
    message.add_alternative(
        _markdown_to_responsive_html(report_text, email_subject),
        subtype="html",
        charset="utf-8",
    )

    smtp_class = smtplib.SMTP_SSL if smtp_security == "ssl" else smtplib.SMTP
    with smtp_class(smtp_host, smtp_port, timeout=smtp_timeout) as smtp:
        if smtp_security == "starttls":
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
        if smtp_username and smtp_password:
            smtp.login(smtp_username, smtp_password)
        smtp.send_message(message, to_addrs=recipients)


async def send_report_email_async(report: str, recipient: str, subject: str) -> None:
    """Send a report without blocking the async research graph."""
    await asyncio.to_thread(send_report_email, report, recipient, subject)
