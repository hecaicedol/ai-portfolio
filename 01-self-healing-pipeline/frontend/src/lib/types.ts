export type DocumentType =
  | "invoice"
  | "contract"
  | "purchase_order"
  | "receipt"
  | "generic";

export type Principle = "completeness" | "accuracy" | "consistency" | "format";

export interface ProcessRequest {
  document_type: DocumentType;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface PrincipleScore {
  principle: Principle;
  score: number;
  feedback: string;
}

export interface CriticReport {
  overall_score: number;
  principles: PrincipleScore[];
  passes: boolean;
  similar_past_errors: string[];
}

export type PipelineEventType =
  | "run_started"
  | "agent_started"
  | "agent_finished"
  | "critic_report"
  | "reflection_triggered"
  | "run_completed"
  | "run_failed";

export interface PipelineEvent {
  type: PipelineEventType;
  agent: string | null;
  iteration: number;
  payload: Record<string, unknown>;
  at: string;
}

export interface ProcessResponse {
  success: boolean;
  iterations: number;
  final_score: number;
  extracted_data: Record<string, unknown>;
  critic_report: CriticReport;
  errors_history: Array<Record<string, unknown>>;
}

export interface EpisodicError {
  id: number;
  created_at: string;
  document_type: string;
  error_type: string;
  principle: string;
  context: Record<string, unknown>;
  resolution: string | null;
  similarity?: number;
}
