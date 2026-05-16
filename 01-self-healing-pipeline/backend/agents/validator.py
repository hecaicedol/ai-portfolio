from typing import Any

REQUIRED_FIELDS: dict[str, list[str]] = {
    "invoice": ["invoice_number", "vendor", "total", "issue_date"],
    "contract": ["parties", "effective_date", "term", "governing_law"],
    "purchase_order": ["po_number", "vendor", "line_items", "total"],
    "receipt": ["merchant", "total", "date"],
    "generic": [],
}


class ValidatorAgent:
    """
    Deterministic, cheap validation pass — no LLM call.
    Checks structural completeness against per-document-type schemas.
    """

    def run(self, *, document_type: str, extracted: dict[str, Any]) -> dict[str, Any]:
        required = REQUIRED_FIELDS.get(document_type, [])
        missing = [f for f in required if extracted.get(f) in (None, "", [], {})]
        return {
            "structural_pass": len(missing) == 0,
            "missing_fields": missing,
            "expected_fields": required,
        }
