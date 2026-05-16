from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    session_id: str
    question: str
    plan: list[dict[str, Any]]
    current_step: int
    findings: list[dict[str, Any]]
    report_path: str
    done: bool


def build_research_graph(*, controller, tools, model):
    """
    LangGraph StateGraph with nodes:

      plan_node       — Claude generates step-by-step plan (search/analyze/synthesize).
                        Reads from controller.get_full_context(question) so previous
                        sessions inform the new plan.

      execute_node    — Runs current step's tool (Tavily / arXiv / synthesizer).

      memory_update   — controller.remember(...) for each result; auto-archives
                        when working memory overflows.

      reflect_node    — At session end, Claude consolidates findings into
                        durable facts → controller.consolidate(...).

      report_node     — Generate structured PDF via tools.report_generator.

    Edges:
      plan → execute → memory_update → (more steps?) execute / reflect
      reflect → report → END
    """
    raise NotImplementedError


async def run(*, question: str, controller, tools, model) -> dict[str, Any]:
    raise NotImplementedError
