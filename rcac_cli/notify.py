"""Notification system for SLURM job completion/failure."""

from __future__ import annotations

import json
import smtplib
import subprocess
import sys
from email.mime.text import MIMEText
from pathlib import Path
from urllib.request import Request, urlopen

import click
import yaml

from .config import get_rcac_config


def _load_notify_config() -> dict:
    """Load notification config from configs/notify.yaml or environment."""
    config_path = Path(__file__).parent.parent / "configs" / "notify.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def send_slack(webhook_url: str, message: str, color: str = "#36a64f") -> bool:
    """Send a message to Slack via webhook."""
    payload = {
        "attachments": [{
            "color": color,
            "text": message,
            "mrkdwn_in": ["text"],
        }]
    }

    data = json.dumps(payload).encode("utf-8")
    req = Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req) as resp:
            return resp.status == 200
    except Exception as e:
        click.echo(f"Slack notification failed: {e}", err=True)
        return False


def send_email(
    smtp_host: str,
    smtp_port: int,
    from_addr: str,
    to_addr: str,
    subject: str,
    body: str,
    password: str | None = None,
) -> bool:
    """Send an email notification."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    try:
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port)
            server.starttls()

        if password:
            server.login(from_addr, password)
        server.sendmail(from_addr, [to_addr], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        click.echo(f"Email notification failed: {e}", err=True)
        return False


def notify_job_result(
    status: str,
    job_name: str,
    exp_id: str,
    metric: str | None = None,
    method: str = "slack",
) -> None:
    """Send a notification about job completion."""
    config = _load_notify_config()

    icon = "✅" if status == "0" else "❌"
    status_text = "COMPLETED" if status == "0" else f"FAILED (exit code: {status})"
    metric_text = f"\nMetric: {metric}" if metric else ""

    message = (
        f"{icon} *RCAC Job {status_text}*\n"
        f"Job: `{job_name}`\n"
        f"Experiment: `{exp_id}`"
        f"{metric_text}"
    )

    if method == "slack":
        webhook = config.get("slack", {}).get("webhook_url", "")
        if not webhook:
            click.echo("Error: Slack webhook URL not configured in configs/notify.yaml")
            return
        color = "#36a64f" if status == "0" else "#ff0000"
        if send_slack(webhook, message, color):
            click.echo("Slack notification sent.")
    elif method == "email":
        email_cfg = config.get("email", {})
        if not email_cfg.get("to"):
            click.echo("Error: Email not configured in configs/notify.yaml")
            return
        subject = f"[RCAC] {job_name} - {status_text}"
        if send_email(
            smtp_host=email_cfg.get("smtp_host", "smtp.gmail.com"),
            smtp_port=email_cfg.get("smtp_port", 587),
            from_addr=email_cfg.get("from", ""),
            to_addr=email_cfg["to"],
            subject=subject,
            body=message.replace("*", "").replace("`", ""),
            password=email_cfg.get("password"),
        ):
            click.echo("Email notification sent.")


@click.group()
def cli() -> None:
    """Notification management for RCAC HPC jobs."""


@cli.command()
@click.option("--method", "-m", type=click.Choice(["slack", "email"]), default="slack")
@click.option("--message", default="Test notification from rcac-cli")
def test(method: str, message: str) -> None:
    """Send a test notification to verify setup."""
    config = _load_notify_config()

    if method == "slack":
        webhook = config.get("slack", {}).get("webhook_url", "")
        if not webhook:
            click.echo("Error: Set slack.webhook_url in configs/notify.yaml")
            return
        if send_slack(webhook, f"🔔 {message}"):
            click.echo("Test notification sent to Slack.")
        else:
            click.echo("Failed to send Slack notification.")
    elif method == "email":
        email_cfg = config.get("email", {})
        if not email_cfg.get("to"):
            click.echo("Error: Set email.to in configs/notify.yaml")
            return
        if send_email(
            smtp_host=email_cfg.get("smtp_host", "smtp.gmail.com"),
            smtp_port=email_cfg.get("smtp_port", 587),
            from_addr=email_cfg.get("from", ""),
            to_addr=email_cfg["to"],
            subject="[RCAC] Test Notification",
            body=message,
            password=email_cfg.get("password"),
        ):
            click.echo("Test email sent.")
        else:
            click.echo("Failed to send email.")


@cli.command()
def setup() -> None:
    """Generate a template notification config file."""
    config_path = Path(__file__).parent.parent / "configs" / "notify.yaml"

    if config_path.exists():
        click.echo(f"Config already exists: {config_path}")
        click.echo("Edit it directly to update settings.")
        return

    template = {
        "slack": {
            "webhook_url": "",  # https://hooks.slack.com/services/...
        },
        "email": {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "from": "",
            "to": "",
            "password": "",  # Use app password for Gmail
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False)

    click.echo(f"Config template created: {config_path}")
    click.echo("Edit it to add your Slack webhook URL or email settings.")


@cli.command("send")
@click.option("--status", required=True, help="Exit code (0=success)")
@click.option("--job-name", required=True)
@click.option("--exp-id", required=True)
@click.option("--metric", default=None)
@click.option("--method", "-m", type=click.Choice(["slack", "email"]), default="slack")
def send(status: str, job_name: str, exp_id: str, metric: str | None, method: str) -> None:
    """Send a job result notification (called from SLURM scripts)."""
    notify_job_result(status, job_name, exp_id, metric, method)
