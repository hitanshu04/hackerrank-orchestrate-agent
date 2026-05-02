from enum import Enum
from pydantic import BaseModel, Field

# --- Enums (Frozen) ---

class Status(str, Enum):
    REPLIED = "replied"
    ESCALATED = "escalated"

class RequestType(str, Enum):
    PRODUCT_ISSUE = "product_issue"
    FEATURE_REQUEST = "feature_request"
    BUG = "bug"
    INVALID = "invalid"

REQUEST_TYPE_PRIORITY = {
    RequestType.INVALID: 4, 
    RequestType.BUG: 3,
    RequestType.FEATURE_REQUEST: 2, 
    RequestType.PRODUCT_ISSUE: 1,
}

class Domain(str, Enum):
    HACKERRANK = "hackerrank"
    CLAUDE = "claude"
    VISA = "visa"
    AMBIGUOUS = "ambiguous"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    WEAK_SUPPORT = "weak_support"
    UNSUPPORTED = "unsupported"


# --- L1 Models ---

class RawTicket(BaseModel):
    issue: str
    subject: str = ""
    company: str = ""
    row_index: int

class CanonicalTicket(BaseModel):
    issue_norm: str
    subject_norm: str
    company_norm: str  # "hackerrank"|"claude"|"visa"|"none"
    combined_text: str
    row_index: int
    fast_fail: bool = False
    has_sensitive_keyword: bool = False
    detected_language: str = "en"
    needs_translation: bool = False
    input_flags: list[str] = Field(default_factory=list)


# --- L2 Models ---

class SubIntent(BaseModel):
    text: str
    confidence: float = 1.0
    risk_hints: list[str] = Field(default_factory=list)

class SplitResult(BaseModel):
    sub_intents: list[SubIntent]
    was_split: bool = False
    merge_applied: bool = False


# --- L3 Models ---

class RoutingDecision(BaseModel):
    domain: str  # Domain enum value or "ambiguous"
    confidence: float
    scores: dict[str, float] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    contradiction_detected: bool = False


# --- L4 Models ---

class TypeDecision(BaseModel):
    request_type: str  # RequestType enum value
    confidence: float
    reason_codes: list[str] = Field(default_factory=list)


# --- L5 Models ---

class RiskDecision(BaseModel):
    risk_level: str  # RiskLevel enum value
    risk_tags: list[str] = Field(default_factory=list)
    force_escalate: bool = False  # ONLY for critical (9-10)
    risk_flag: bool = False       # For high (5-8), L6 decides
    risk_score: int = 0
    reason_codes: list[str] = Field(default_factory=list)


# --- L6 Models ---

class EvidenceChunk(BaseModel):
    text: str
    source_file: str
    score: float

class RetrievalEvidence(BaseModel):
    chunks: list[EvidenceChunk] = Field(default_factory=list)
    support_status: str  # SupportStatus enum value
    best_score: float = 0.0
    domain_searched: str = ""
    reason_codes: list[str] = Field(default_factory=list)


# --- L7 Models ---

class ComposedOutput(BaseModel):
    response: str
    justification: str
    product_area: str = ""
    status: str  # Status enum value
    request_type: str  # RequestType enum value
    verbatim_quotes: list[str] = Field(default_factory=list)
    found_in_context: bool = True
    composer_confidence: float = 0.0
    unsupported_claims: list[str] = Field(default_factory=list)
    grounding_coverage: float = 0.0
    reason_codes: list[str] = Field(default_factory=list)


# --- L8 Models ---

class ValidatedOutput(BaseModel):
    """Final output row — exactly matches output.csv schema."""
    issue: str
    subject: str
    company: str
    response: str
    product_area: str = ""
    status: str
    request_type: str
    justification: str
    row_index: int
    validation_passed: bool = True
    fallback_applied: bool = False


# --- Pipeline State ---

class PipelineState(BaseModel):
    """Accumulates all layer decisions for a single row."""
    raw: RawTicket
    canonical: CanonicalTicket | None = None
    split: SplitResult | None = None
    routing: RoutingDecision | None = None
    type_decision: TypeDecision | None = None
    risk: RiskDecision | None = None
    evidence: RetrievalEvidence | None = None
    composed: ComposedOutput | None = None
    validated: ValidatedOutput | None = None
    error: str | None = None
    layer_timings: dict[str, float] = Field(default_factory=dict)
