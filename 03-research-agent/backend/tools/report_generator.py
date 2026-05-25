"""PDF report generator (ReportLab).

Slice 1 deliverable: validate the report payload, generate a deterministic
output path, and (when ReportLab is installed) actually emit a PDF. The
ReportLab import is lazy so tests can run without it — the test suite
checks payload validation + path generation instead.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ReportSection(BaseModel):
    heading: str
    body: str
    citations: list[str] = Field(default_factory=list)


class ReportSource(BaseModel):
    title: str
    url: str
    credibility: float = Field(0.5, ge=0.0, le=1.0)


class ReportInput(BaseModel):
    title: str
    executive_summary: str
    methodology: str
    sections: list[ReportSection] = Field(default_factory=list)
    sources: list[ReportSource] = Field(default_factory=list)
    confidence_assessment: str = ""


class ReportGenerator:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def output_path(self, session_id: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self.output_dir / f"report-{safe}-{timestamp}.pdf"

    def generate(self, payload: ReportInput, *, session_id: str) -> Path:
        path = self.output_path(session_id)
        try:
            self._emit_pdf(payload, path)
        except ImportError:
            # ReportLab not installed (dev / test). Write a stub so the
            # caller still has a real file to reference.
            path.write_text(
                f"# {payload.title}\n\n{payload.executive_summary}\n\n"
                f"({len(payload.sections)} sections, {len(payload.sources)} sources)\n",
                encoding="utf-8",
            )
        return path

    def _emit_pdf(self, payload: ReportInput, path: Path) -> None:
        from reportlab.lib.pagesizes import LETTER  # type: ignore
        from reportlab.lib.styles import getSampleStyleSheet  # type: ignore
        from reportlab.platypus import (  # type: ignore
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            PageBreak,
        )

        doc = SimpleDocTemplate(str(path), pagesize=LETTER)
        styles = getSampleStyleSheet()
        story: list[Any] = []
        story.append(Paragraph(payload.title, styles["Title"]))
        story.append(Spacer(1, 18))
        story.append(Paragraph("Executive Summary", styles["Heading2"]))
        story.append(Paragraph(payload.executive_summary, styles["BodyText"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Methodology", styles["Heading2"]))
        story.append(Paragraph(payload.methodology, styles["BodyText"]))
        story.append(PageBreak())
        for section in payload.sections:
            story.append(Paragraph(section.heading, styles["Heading2"]))
            story.append(Paragraph(section.body, styles["BodyText"]))
            for c in section.citations:
                story.append(Paragraph(f"[{c}]", styles["Italic"]))
            story.append(Spacer(1, 12))
        if payload.sources:
            story.append(PageBreak())
            story.append(Paragraph("Sources", styles["Heading2"]))
            for s in payload.sources:
                story.append(Paragraph(
                    f"{s.title} — {s.url} (credibility {s.credibility:.2f})",
                    styles["BodyText"],
                ))
        if payload.confidence_assessment:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Confidence Assessment", styles["Heading2"]))
            story.append(Paragraph(payload.confidence_assessment, styles["BodyText"]))
        doc.build(story)
