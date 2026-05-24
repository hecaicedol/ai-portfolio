import type {
  CriticReport,
  DocumentType,
  EpisodicError,
  PipelineEvent,
  PrincipleScore,
} from "./types";

/**
 * Pre-baked demo scenario: an invoice that fails the critic on the first
 * extraction (the LLM picks the wrong line as `total`) and self-heals on
 * the second iteration once the reflection node feeds the feedback back
 * into the extractor.
 */
export const DEMO_SCENARIO = {
  document_type: "invoice" as DocumentType,
  content: `Globex Corporation
----------------
Invoice: GLX-2026-554
Issued on: March 22nd, 2026

Professional services:
  Architecture review                $3,500
  Implementation support              $7,200

Total due: $10,700.00
Net 30
`,
};

const EXTRACTED_BAD = {
  invoice_number: "GLX-2026-554",
  vendor: "Globex Corporation",
  total: 7200.0,
  issue_date: "2026-03-22",
};

const EXTRACTED_GOOD = {
  invoice_number: "GLX-2026-554",
  vendor: "Globex Corporation",
  total: 10700.0,
  issue_date: "2026-03-22",
};

const CRITIC_FAIL_PRINCIPLES: PrincipleScore[] = [
  {
    principle: "completeness",
    score: 0.95,
    feedback: "All four required fields present.",
  },
  {
    principle: "accuracy",
    score: 0.4,
    feedback:
      "`total: 7200.00` matches the second line item, not the document total ($10,700.00 stated under 'Total due').",
  },
  {
    principle: "consistency",
    score: 0.55,
    feedback:
      "Line items sum to $10,700 but extracted total is $7,200 — internal inconsistency.",
  },
  {
    principle: "format",
    score: 0.85,
    feedback: "Numeric / ISO-8601 date format correct.",
  },
];

const CRITIC_PASS_PRINCIPLES: PrincipleScore[] = [
  {
    principle: "completeness",
    score: 0.95,
    feedback: "All four required fields present.",
  },
  {
    principle: "accuracy",
    score: 0.95,
    feedback: "Every value verifiable from the source text.",
  },
  {
    principle: "consistency",
    score: 0.92,
    feedback: "Line items now sum to extracted total.",
  },
  {
    principle: "format",
    score: 0.9,
    feedback: "Numeric / ISO-8601 date format correct.",
  },
];

function avg(scores: PrincipleScore[]): number {
  return scores.reduce((s, p) => s + p.score, 0) / scores.length;
}

const CRITIC_FAIL: CriticReport = {
  overall_score: avg(CRITIC_FAIL_PRINCIPLES),
  principles: CRITIC_FAIL_PRINCIPLES,
  passes: false,
  similar_past_errors: [
    "accuracy_below_threshold",
    "consistency_below_threshold",
  ],
};

const CRITIC_PASS: CriticReport = {
  overall_score: avg(CRITIC_PASS_PRINCIPLES),
  principles: CRITIC_PASS_PRINCIPLES,
  passes: true,
  similar_past_errors: ["accuracy_below_threshold"],
};

const STRUCTURAL_PASS = {
  structural_pass: true,
  missing_fields: [],
  expected_fields: ["invoice_number", "vendor", "total", "issue_date"],
};

/** A pre-recorded SSE stream with realistic per-event delays in ms. */
export const DEMO_FRAMES: Array<{ delayMs: number; event: PipelineEvent }> = [
  {
    delayMs: 120,
    event: {
      type: "run_started",
      agent: null,
      iteration: 0,
      payload: { document_type: "invoice" },
      at: new Date().toISOString(),
    },
  },
  {
    delayMs: 900,
    event: {
      type: "agent_finished",
      agent: "extract",
      iteration: 1,
      payload: { extracted: EXTRACTED_BAD, iterations: 1 },
      at: new Date().toISOString(),
    },
  },
  {
    delayMs: 350,
    event: {
      type: "agent_finished",
      agent: "validate",
      iteration: 1,
      payload: { structural: STRUCTURAL_PASS },
      at: new Date().toISOString(),
    },
  },
  {
    delayMs: 1700,
    event: {
      type: "agent_finished",
      agent: "critique",
      iteration: 1,
      payload: { critic: CRITIC_FAIL },
      at: new Date().toISOString(),
    },
  },
  {
    delayMs: 700,
    event: {
      type: "agent_finished",
      agent: "reflect",
      iteration: 1,
      payload: {
        last_feedback: [
          CRITIC_FAIL_PRINCIPLES[1].feedback,
          CRITIC_FAIL_PRINCIPLES[2].feedback,
        ],
        errors_history: [
          {
            iteration: 1,
            score: CRITIC_FAIL.overall_score,
            feedback: [
              CRITIC_FAIL_PRINCIPLES[1].feedback,
              CRITIC_FAIL_PRINCIPLES[2].feedback,
            ],
          },
        ],
      },
      at: new Date().toISOString(),
    },
  },
  {
    delayMs: 1300,
    event: {
      type: "agent_finished",
      agent: "extract",
      iteration: 2,
      payload: { extracted: EXTRACTED_GOOD, iterations: 2 },
      at: new Date().toISOString(),
    },
  },
  {
    delayMs: 350,
    event: {
      type: "agent_finished",
      agent: "validate",
      iteration: 2,
      payload: { structural: STRUCTURAL_PASS },
      at: new Date().toISOString(),
    },
  },
  {
    delayMs: 1500,
    event: {
      type: "agent_finished",
      agent: "critique",
      iteration: 2,
      payload: { critic: CRITIC_PASS },
      at: new Date().toISOString(),
    },
  },
  {
    delayMs: 450,
    event: {
      type: "agent_finished",
      agent: "synthesize",
      iteration: 2,
      payload: {
        final: {
          data: EXTRACTED_GOOD,
          audit: {
            score: CRITIC_PASS.overall_score,
            principles: CRITIC_PASS_PRINCIPLES,
            iterations: 2,
            self_healed: true,
            errors_history: [
              {
                iteration: 1,
                score: CRITIC_FAIL.overall_score,
                feedback: [
                  CRITIC_FAIL_PRINCIPLES[1].feedback,
                  CRITIC_FAIL_PRINCIPLES[2].feedback,
                ],
              },
            ],
          },
        },
      },
      at: new Date().toISOString(),
    },
  },
  {
    delayMs: 200,
    event: {
      type: "run_completed",
      agent: null,
      iteration: 2,
      payload: {},
      at: new Date().toISOString(),
    },
  },
];

/** Async generator that yields the scripted SSE frames with their delays. */
export async function* streamDemoPipeline(
  signal?: AbortSignal,
): AsyncGenerator<PipelineEvent> {
  for (const frame of DEMO_FRAMES) {
    if (signal?.aborted) return;
    await new Promise<void>((resolve, reject) => {
      const t = setTimeout(resolve, frame.delayMs);
      signal?.addEventListener(
        "abort",
        () => {
          clearTimeout(t);
          reject(new DOMException("aborted", "AbortError"));
        },
        { once: true },
      );
    });
    yield frame.event;
  }
}

/** Plausible-looking errors for the MemoryPanel when there's no real backend. */
export const DEMO_RECENT_ERRORS: EpisodicError[] = [
  {
    id: 1031,
    created_at: new Date(Date.now() - 1000 * 60 * 23).toISOString(),
    document_type: "invoice",
    error_type: "accuracy_below_threshold",
    principle: "accuracy",
    context: {
      feedback: "total field doesn't match document",
      extracted_total: 7200,
    },
    resolution: null,
    similarity: 0.92,
  },
  {
    id: 1029,
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 4).toISOString(),
    document_type: "invoice",
    error_type: "consistency_below_threshold",
    principle: "consistency",
    context: { line_items_sum: 10700, extracted_total: 7200 },
    resolution: null,
    similarity: 0.88,
  },
  {
    id: 1024,
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 26).toISOString(),
    document_type: "receipt",
    error_type: "completeness_below_threshold",
    principle: "completeness",
    context: { missing: ["date"] },
    resolution: null,
  },
  {
    id: 1018,
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(),
    document_type: "invoice",
    error_type: "format_below_threshold",
    principle: "format",
    context: { issue: "date returned as 'March 22nd, 2026' instead of ISO" },
    resolution: null,
  },
  {
    id: 1003,
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 5).toISOString(),
    document_type: "contract",
    error_type: "completeness_below_threshold",
    principle: "completeness",
    context: { missing: ["governing_law"] },
    resolution: "Added explicit governing-law extraction prompt",
  },
];
