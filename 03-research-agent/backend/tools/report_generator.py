from pathlib import Path
from pydantic import BaseModel


class ReportInput(BaseModel):
    title: str
    executive_summary: str
    methodology: str
    sections: list[dict]
    sources: list[dict]
    confidence_assessment: str


class ReportGenerator:
    """
    Generates structured PDF reports with ReportLab.
    Sections:
      - Title page
      - Executive summary
      - Methodology
      - Findings by section
      - Sources (with credibility score)
      - Confidence assessment
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, payload: ReportInput, *, session_id: str) -> Path:
        raise NotImplementedError("Build with ReportLab platypus (SimpleDocTemplate, Paragraph, Table)")
