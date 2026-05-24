"""Validator is deterministic and LLM-free — easiest place to start."""
from agents.validator import ValidatorAgent


def test_invoice_all_required_present():
    v = ValidatorAgent()
    out = v.run(
        document_type="invoice",
        extracted={
            "invoice_number": "INV-001",
            "vendor": "Acme",
            "total": 100.0,
            "issue_date": "2026-01-15",
        },
    )
    assert out["structural_pass"] is True
    assert out["missing_fields"] == []


def test_invoice_missing_fields_detected():
    v = ValidatorAgent()
    out = v.run(
        document_type="invoice",
        extracted={"invoice_number": "INV-001", "vendor": None, "total": "", "issue_date": []},
    )
    assert out["structural_pass"] is False
    assert set(out["missing_fields"]) == {"vendor", "total", "issue_date"}


def test_unknown_doctype_treated_as_no_requirements():
    v = ValidatorAgent()
    out = v.run(document_type="something_weird", extracted={})
    assert out["structural_pass"] is True
    assert out["missing_fields"] == []


def test_generic_doctype_has_no_required_fields():
    v = ValidatorAgent()
    out = v.run(document_type="generic", extracted={"anything": "value"})
    assert out["structural_pass"] is True
