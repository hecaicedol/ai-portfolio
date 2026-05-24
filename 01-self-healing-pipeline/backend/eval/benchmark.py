"""Benchmark harness for the self-healing pipeline.

Loads every document under ./corpus/, runs each through the pipeline with
reflection enabled (max=3), grades the extracted output against ground_truth,
and writes both raw per-doc results and aggregated summary stats.

Two modes:
  --mode dry-run  uses OracleStub (deterministic, no API key, $0 cost)
                  every doc passes with accuracy=1.0 by construction — used
                  ONLY to verify the harness end-to-end before paying for tokens.
  --mode real     uses real Claude via langchain-anthropic; requires
                  ANTHROPIC_API_KEY in the environment.

Usage:
    python -m eval.benchmark --mode dry-run
    python -m eval.benchmark --mode real --label baseline-2026-05
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.orchestrator import build_graph, run_pipeline  # noqa: E402
from config import Settings  # noqa: E402
from tests.conftest import FakeMemory  # noqa: E402

CORPUS_DIR = Path(__file__).parent / "corpus"
RESULTS_DIR = Path(__file__).parent / "results"


# ── corpus + grader ──────────────────────────────────────────────────────────

def load_corpus() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for f in sorted(CORPUS_DIR.glob("*.json")):
        items.append(json.loads(f.read_text(encoding="utf-8")))
    return items


def field_match(actual: Any, expected: Any) -> bool:
    """Lenient equality used by the grader:
      - numbers: tolerated up to 0.01
      - strings: case-insensitive after strip
      - lists/dicts: element-wise via field_match
    """
    if actual is None:
        return expected is None
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) < 0.01
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.strip().lower() == expected.strip().lower()
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(field_match(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(field_match(actual.get(k), v) for k, v in expected.items())
    return actual == expected


def grade(extracted: dict[str, Any] | None, ground_truth: dict[str, Any]) -> dict[str, Any]:
    extracted = extracted or {}
    field_results: dict[str, Any] = {}
    matches = 0
    for field, expected in ground_truth.items():
        actual = extracted.get(field)
        is_match = field_match(actual, expected)
        field_results[field] = {"expected": expected, "actual": actual, "match": is_match}
        if is_match:
            matches += 1
    accuracy = matches / len(ground_truth) if ground_truth else 0.0
    return {"accuracy": accuracy, "fields": field_results}


# ── stub LLM for dry-run ─────────────────────────────────────────────────────

class OracleStub:
    """Returns the doc's ground_truth verbatim for extractor calls and a
    perfect critic report. Use only to verify the harness — not for metrics."""

    def __init__(self, item: dict[str, Any]) -> None:
        self.item = item

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
        sys_content = messages[0].content
        if "Critic agent" in sys_content:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "overall_score": 1.0,
                        "principles": [
                            {"principle": p, "score": 1.0, "feedback": "ok"}
                            for p in ("completeness", "accuracy", "consistency", "format")
                        ],
                    }
                )
            )
        return SimpleNamespace(content=json.dumps(self.item["ground_truth"]))


# ── runner ───────────────────────────────────────────────────────────────────

async def run_benchmark(
    model_factory: Callable[[dict[str, Any]], Any],
    *,
    label: str,
) -> list[dict[str, Any]]:
    items = load_corpus()
    settings = Settings()  # type: ignore[call-arg]
    results: list[dict[str, Any]] = []

    print(f"\n=== Benchmark [{label}] — {len(items)} docs ===")
    for item in items:
        memory = FakeMemory()
        model = model_factory(item)
        graph = build_graph(memory=memory, settings=settings, model=model)

        t0 = time.perf_counter()
        try:
            result = await run_pipeline(
                graph=graph,
                memory=memory,
                document_type=item["document_type"],
                content=item["content"],
                metadata={"id": item["id"]},
            )
            latency = time.perf_counter() - t0
            grading = grade(result.extracted_data, item["ground_truth"])
            row = {
                "id": item["id"],
                "document_type": item["document_type"],
                "difficulty": item["difficulty"],
                "success": result.success,
                "iterations": result.iterations,
                "final_score": result.final_score,
                "accuracy": grading["accuracy"],
                "fields": grading["fields"],
                "latency_s": round(latency, 2),
                "error": None,
            }
            print(
                f"  {item['id']:<38} "
                f"pass={'Y' if result.success else 'N'} "
                f"iter={result.iterations} "
                f"score={result.final_score:.2f} "
                f"acc={grading['accuracy']:.0%} "
                f"t={latency:.1f}s"
            )
        except Exception as exc:
            latency = time.perf_counter() - t0
            row = {
                "id": item["id"],
                "document_type": item["document_type"],
                "difficulty": item["difficulty"],
                "success": False,
                "iterations": 0,
                "final_score": 0.0,
                "accuracy": 0.0,
                "fields": {},
                "latency_s": round(latency, 2),
                "error": str(exc),
            }
            print(f"  {item['id']:<38} ERROR: {exc}")
        results.append(row)

    return results


# ── aggregator ───────────────────────────────────────────────────────────────

def _group_stats(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        grouped.setdefault(str(r.get(key, "unknown")), []).append(r)
    out: dict[str, dict[str, float]] = {}
    for k, rows in grouped.items():
        successes = [r for r in rows if r["success"]]
        out[k] = {
            "n": len(rows),
            "pass_rate": len(successes) / len(rows),
            "avg_accuracy": sum(r["accuracy"] for r in rows) / len(rows),
            "avg_iterations_on_success": (
                sum(r["iterations"] for r in successes) / len(successes) if successes else 0.0
            ),
        }
    return out


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    successes = [r for r in results if r["success"]]
    pass_without_reflection = sum(1 for r in results if r["success"] and r["iterations"] == 1)
    pass_with_reflection = len(successes)
    healed_by_reflection = pass_with_reflection - pass_without_reflection
    return {
        "total_docs": n,
        "pass_rate_with_reflection": pass_with_reflection / n if n else 0.0,
        "pass_rate_without_reflection": pass_without_reflection / n if n else 0.0,
        "healed_by_reflection": healed_by_reflection,
        "avg_iterations_on_success": (
            sum(r["iterations"] for r in successes) / len(successes) if successes else 0.0
        ),
        "avg_accuracy_overall": sum(r["accuracy"] for r in results) / n if n else 0.0,
        "avg_latency_s": sum(r["latency_s"] for r in results) / n if n else 0.0,
        "by_doc_type": _group_stats(results, "document_type"),
        "by_difficulty": _group_stats(results, "difficulty"),
    }


def render_markdown(summary: dict[str, Any], label: str, timestamp: str) -> str:
    p_with = summary["pass_rate_with_reflection"]
    p_without = summary["pass_rate_without_reflection"]
    delta_pp = (p_with - p_without) * 100

    lines = [
        f"# P1 self-healing pipeline — benchmark results",
        "",
        f"**Run label:** `{label}`  ",
        f"**Run timestamp (UTC):** {timestamp}  ",
        f"**Corpus size:** {summary['total_docs']} synthetic documents",
        "",
        "## Headline metrics",
        "",
        "| Metric | Without reflection | With reflection | Δ |",
        "|---|---|---|---|",
        f"| Pass rate (overall) | {p_without:.0%} | {p_with:.0%} | +{delta_pp:.0f} pp |",
        f"| Avg accuracy across required fields | — | {summary['avg_accuracy_overall']:.0%} | — |",
        f"| Avg iterations on successful runs | 1 | {summary['avg_iterations_on_success']:.2f} | — |",
        f"| Docs that needed reflection to pass | n/a | {summary['healed_by_reflection']} | — |",
        f"| Avg latency per doc (s) | — | {summary['avg_latency_s']:.1f} | — |",
        "",
        "## By document type",
        "",
        "| Type | N | Pass rate | Avg accuracy | Avg iters (on success) |",
        "|---|---|---|---|---|",
    ]
    for t, stats in sorted(summary["by_doc_type"].items()):
        lines.append(
            f"| {t} | {int(stats['n'])} | {stats['pass_rate']:.0%} | "
            f"{stats['avg_accuracy']:.0%} | {stats['avg_iterations_on_success']:.2f} |"
        )

    lines += [
        "",
        "## By difficulty",
        "",
        "| Difficulty | N | Pass rate | Avg accuracy | Avg iters (on success) |",
        "|---|---|---|---|---|",
    ]
    diff_order = ["easy", "medium", "hard"]
    diff_items = sorted(
        summary["by_difficulty"].items(),
        key=lambda kv: (diff_order.index(kv[0]) if kv[0] in diff_order else 99, kv[0]),
    )
    for d, stats in diff_items:
        lines.append(
            f"| {d} | {int(stats['n'])} | {stats['pass_rate']:.0%} | "
            f"{stats['avg_accuracy']:.0%} | {stats['avg_iterations_on_success']:.2f} |"
        )
    return "\n".join(lines) + "\n"


# ── factories ────────────────────────────────────────────────────────────────

def make_oracle_factory() -> Callable[[dict[str, Any]], Any]:
    def factory(item: dict[str, Any]) -> Any:
        return OracleStub(item)
    return factory


def make_real_factory() -> Callable[[dict[str, Any]], Any]:
    from langchain_anthropic import ChatAnthropic
    settings = Settings()  # type: ignore[call-arg]

    def factory(_item: dict[str, Any]) -> Any:
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=2048,
        )
    return factory


# ── entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="P1 self-healing pipeline benchmark")
    parser.add_argument(
        "--mode",
        choices=["dry-run", "real"],
        default="dry-run",
        help="dry-run: OracleStub, no API key, $0; real: Claude via ANTHROPIC_API_KEY",
    )
    parser.add_argument("--label", default=None, help="Label used for output filenames")
    args = parser.parse_args()

    label = args.label or args.mode
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.mode == "real":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key or key == "test-key-not-real":
            print("ERROR: --mode real requires a real ANTHROPIC_API_KEY in the env.")
            sys.exit(1)
        factory = make_real_factory()
    else:
        factory = make_oracle_factory()

    timestamp = datetime.now(timezone.utc).isoformat()
    results = asyncio.run(run_benchmark(factory, label=label))
    summary = aggregate(results)

    results_path = RESULTS_DIR / f"{label}_results.json"
    summary_path = RESULTS_DIR / f"{label}_summary.json"
    md_path = RESULTS_DIR / f"{label}_summary.md"

    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(summary, label, timestamp), encoding="utf-8")

    print()
    print(f"=== Summary ({label}) ===")
    print(f"  Pass rate (with reflection):     {summary['pass_rate_with_reflection']:.0%}")
    print(f"  Pass rate (without reflection):  {summary['pass_rate_without_reflection']:.0%}")
    print(f"  Healed by reflection:            {summary['healed_by_reflection']} docs")
    print(f"  Avg accuracy:                    {summary['avg_accuracy_overall']:.0%}")
    print(f"  Avg iterations on success:       {summary['avg_iterations_on_success']:.2f}")
    print(f"  Avg latency:                     {summary['avg_latency_s']:.1f}s/doc")
    print()
    print(f"  Wrote: {results_path.name}, {summary_path.name}, {md_path.name}")


if __name__ == "__main__":
    main()
