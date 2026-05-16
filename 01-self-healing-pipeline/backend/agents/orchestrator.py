import hashlib
from typing import Any, AsyncIterator, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph

from agents.critic import CriticAgent
from agents.extractor import ExtractorAgent
from agents.synthesizer import SynthesizerAgent
from agents.validator import ValidatorAgent
from api.schemas import CriticReport, PipelineEvent, ProcessResponse
from config import Settings
from memory.episodic import EpisodicMemory


class PipelineState(TypedDict, total=False):
    document_type: str
    content: str
    metadata: dict[str, Any]
    extracted: dict[str, Any]
    structural: dict[str, Any]
    critic: CriticReport
    iterations: int
    errors_history: list[dict[str, Any]]
    final: dict[str, Any]
    last_feedback: list[str]


def build_graph(memory: EpisodicMemory, settings: Settings):
    model = ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=2048,
    )

    extractor = ExtractorAgent(model=model)
    validator = ValidatorAgent()
    critic = CriticAgent(model=model, memory=memory, pass_threshold=settings.critic_pass_threshold)
    synthesizer = SynthesizerAgent()

    async def extract_node(state: PipelineState) -> dict[str, Any]:
        extracted = await extractor.run(
            document_type=state["document_type"],
            content=state["content"],
            critic_feedback=state.get("last_feedback"),
        )
        return {"extracted": extracted, "iterations": state.get("iterations", 0) + 1}

    async def validate_node(state: PipelineState) -> dict[str, Any]:
        structural = validator.run(
            document_type=state["document_type"],
            extracted=state["extracted"],
        )
        return {"structural": structural}

    async def critic_node(state: PipelineState) -> dict[str, Any]:
        report = await critic.run(
            document_type=state["document_type"],
            source=state["content"],
            extracted=state["extracted"],
            structural=state["structural"],
        )
        return {"critic": report}

    async def reflection_node(state: PipelineState) -> dict[str, Any]:
        report = state["critic"]
        feedback = [p.feedback for p in report.principles if p.score < settings.critic_pass_threshold]
        history = state.get("errors_history", [])
        history.append(
            {
                "iteration": state["iterations"],
                "score": report.overall_score,
                "feedback": feedback,
            }
        )
        for p in report.principles:
            if p.score < settings.critic_pass_threshold:
                await memory.save_error(
                    document_type=state["document_type"],
                    error_type=p.principle + "_below_threshold",
                    principle=p.principle,
                    context={"feedback": p.feedback, "extracted": state["extracted"]},
                )
        return {"last_feedback": feedback, "errors_history": history}

    async def synthesize_node(state: PipelineState) -> dict[str, Any]:
        final = synthesizer.run(
            extracted=state["extracted"],
            critic_report=state["critic"],
            iterations=state["iterations"],
            errors_history=state.get("errors_history", []),
        )
        return {"final": final}

    def route_after_critic(state: PipelineState) -> str:
        if state["critic"].passes:
            return "synthesize"
        if state["iterations"] >= settings.max_reflection_iterations:
            return "synthesize"
        return "reflect"

    graph = StateGraph(PipelineState)
    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reflect", reflection_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"reflect": "reflect", "synthesize": "synthesize"})
    graph.add_edge("reflect", "extract")
    graph.add_edge("synthesize", END)

    return graph.compile()


async def run_pipeline(
    *,
    graph,
    document_type: str,
    content: str,
    metadata: dict[str, Any],
) -> ProcessResponse:
    initial: PipelineState = {
        "document_type": document_type,
        "content": content,
        "metadata": metadata,
        "iterations": 0,
        "errors_history": [],
    }
    state = await graph.ainvoke(initial)

    return ProcessResponse(
        success=state["critic"].passes,
        iterations=state["iterations"],
        final_score=state["critic"].overall_score,
        extracted_data=state["final"]["data"],
        critic_report=state["critic"],
        errors_history=state.get("errors_history", []),
    )


async def stream_pipeline(
    *,
    graph,
    document_type: str,
    content: str,
    metadata: dict[str, Any],
) -> AsyncIterator[PipelineEvent]:
    initial: PipelineState = {
        "document_type": document_type,
        "content": content,
        "metadata": metadata,
        "iterations": 0,
        "errors_history": [],
    }
    yield PipelineEvent(type="run_started", payload={"document_type": document_type})

    async for chunk in graph.astream(initial):
        for node_name, node_state in chunk.items():
            yield PipelineEvent(
                type="agent_finished",
                agent=node_name,
                iteration=node_state.get("iterations", 0),
                payload={k: _serializable(v) for k, v in node_state.items()},
            )

    yield PipelineEvent(type="run_completed")


def _serializable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _doc_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]
