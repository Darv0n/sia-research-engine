"""Multi-format report export via pandoc (V9, G8).

Converts Markdown reports to DOCX and PDF formats using pandoc.
Falls back gracefully if pandoc is not installed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def pandoc_available() -> bool:
    """Check if pandoc is installed and accessible."""
    return shutil.which("pandoc") is not None


def convert_report(markdown_text: str, output_format: str, output_path: str) -> bool:
    """Convert a Markdown report to the specified format.

    Args:
        markdown_text: The Markdown report content.
        output_format: Target format ('docx' or 'pdf').
        output_path: Path to write the output file.

    Returns:
        True if conversion succeeded, False otherwise.
    """
    if output_format not in ("docx", "pdf"):
        print(f"ERROR: Unsupported format '{output_format}'. Use 'docx' or 'pdf'.", file=sys.stderr)
        return False

    if not pandoc_available():
        print(
            "ERROR: pandoc is not installed. Install it from https://pandoc.org/installing.html\n"
            "  On Windows: winget install pandoc\n"
            "  On macOS: brew install pandoc\n"
            "  On Ubuntu: apt install pandoc",
            file=sys.stderr,
        )
        return False

    # Write markdown to temp file (pandoc reads from file, not stdin on Windows)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(markdown_text)
        temp_path = f.name

    try:
        cmd = [
            "pandoc",
            temp_path,
            "-o",
            output_path,
            "--from",
            "markdown",
            "--standalone",
        ]

        # PDF needs a PDF engine
        if output_format == "pdf":
            # Try common PDF engines in order of preference
            for engine in ["xelatex", "pdflatex", "wkhtmltopdf"]:
                if shutil.which(engine):
                    cmd.extend(["--pdf-engine", engine])
                    break
            else:
                print(
                    "WARNING: No PDF engine found. Trying default pandoc PDF generation.",
                    file=sys.stderr,
                )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print(f"ERROR: pandoc conversion failed:\n{result.stderr}", file=sys.stderr)
            return False

        return True

    except subprocess.TimeoutExpired:
        print("ERROR: pandoc conversion timed out after 60 seconds.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: pandoc conversion failed: {e}", file=sys.stderr)
        return False
    finally:
        Path(temp_path).unlink(missing_ok=True)
