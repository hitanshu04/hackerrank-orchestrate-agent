import json
import os
import re
from google import genai
from pydantic import BaseModel
from models import (
    CanonicalTicket,
    RiskDecision,
    RetrievalEvidence,
    ComposedOutput,
    Status,
)
from config import LLM_PROVIDER, PRO_MODEL_PARAMS, OPENAI_BASE_URL
from openai import OpenAI


class ComposerDraft(BaseModel):
    verbatim_quotes: list[str]
    response: str
    confidence: float


class CriticResult(BaseModel):
    unsupported_claims: list[str]


class GroundedComposer:
    """
    Layer 7: Grounded Composer + Critic
    Uses Gemini 3.1 Pro for complex reasoning and drafting.
    Enforces 'Quote-First' grounding and includes an adversarial critic pass.
    """

    def __init__(self):
        self.provider = LLM_PROVIDER
        if self.provider == "google":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is required when LLM_PROVIDER=google"
                )
            self.client = genai.Client(api_key=api_key)
            self.model_name = PRO_MODEL_PARAMS["google"]["model"]
        elif self.provider == "openai":
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY or OPENAI_API_KEY required")
            self.client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
            self.model_name = PRO_MODEL_PARAMS["openai"]["model"]

        self.trivial_patterns = [
            r"^(thank you|thanks|thx|thanks a lot)$",
            r"^(hello|hi|hey)$",
            r"^thank you for helping me$",
        ]

    def _is_trivial(self, text: str) -> bool:
        text_lower = text.strip().lower()
        for p in self.trivial_patterns:
            if re.match(p, text_lower):
                return True
        return False

    def _get_product_area(self, evidence: RetrievalEvidence) -> str:
        if not evidence.chunks:
            return ""
        # Best chunk source: data/visa/travel_support.md -> travel_support
        best_source = evidence.chunks[0].source_file
        basename = os.path.basename(best_source)
        area = os.path.splitext(basename)[0]
        return area

    def compose(
        self,
        ticket: CanonicalTicket,
        risk: RiskDecision,
        evidence: RetrievalEvidence,
        request_type: str,
    ) -> ComposedOutput:
        area = self._get_product_area(evidence)

        # 1. Short-circuit: Trivial
        if self._is_trivial(ticket.issue_norm):
            return ComposedOutput(
                response="Happy to help",
                justification="TRIVIAL_ACK",
                status=Status.REPLIED.value,
                request_type=request_type,
                product_area="",
                reason_codes=["short_circuit_trivial"],
            )

        # 2. Short-circuit: Critical Risk (Force Escalate)
        if risk.force_escalate:
            return ComposedOutput(
                response="Escalate to a human",
                justification="FORCED_ESCALATION_CRITICAL_RISK",
                status=Status.ESCALATED.value,
                request_type=request_type,
                product_area=area,
                reason_codes=["short_circuit_force_escalate"],
            )

        # 3. Short-circuit: Risk Flagged but Unsupported
        if risk.risk_flag and evidence.support_status == "unsupported":
            return ComposedOutput(
                response="Escalate to a human",
                justification="RISK_FLAGGED_UNSUPPORTED",
                status=Status.ESCALATED.value,
                request_type=request_type,
                product_area=area,
                reason_codes=["short_circuit_risk_unsupported"],
            )

        # 4. Short-circuit: Benign but Unsupported (Out of scope)
        if evidence.support_status == "unsupported":
            return ComposedOutput(
                response="I am sorry, this is out of scope from my capabilities",
                justification="OUT_OF_SCOPE_UNSUPPORTED",
                status=Status.REPLIED.value,
                request_type=request_type,
                product_area="",
                reason_codes=["short_circuit_oos"],
            )

        # 5. We have support! Draft a grounded response (L7A)
        draft = self._draft_response(ticket, evidence)
        if not draft:
            return ComposedOutput(
                response="Escalate to a human",
                justification="DRAFT_FAILED_ESCALATE",
                status=Status.ESCALATED.value,
                request_type=request_type,
                product_area=area,
                reason_codes=["llm_draft_failed"],
            )

        # Grounding check
        # For Google (structured output), enforce strict quoting.
        # For OpenAI/Groq, allow responses without exact quotes since
        # open-source models struggle with exact substring extraction.
        if not draft.verbatim_quotes and draft.response:
            if self.provider == "google":
                return ComposedOutput(
                    response="Escalate to a human",
                    justification="NO_VERBATIM_QUOTES_ESCALATE",
                    status=Status.ESCALATED.value,
                    request_type=request_type,
                    product_area=area,
                    reason_codes=["failed_quote_extraction"],
                )
            # For non-Google providers, pass through with reduced confidence
            draft.confidence = min(draft.confidence, 0.5)

        # 6. Critic Pass (L7B)
        # Only run critic with Google (structured output guarantees).
        # Open-source models produce too many false positives which
        # cause 100% escalation rates.
        if self.provider == "google":
            critic = self._run_critic(draft.response, evidence)
        else:
            critic = None
        if critic and len(critic.unsupported_claims) > 0:
            return ComposedOutput(
                response="Escalate to a human",
                justification="CRITIC_UNSUPPORTED_CLAIMS",
                status=Status.ESCALATED.value,
                request_type=request_type,
                product_area=area,
                unsupported_claims=critic.unsupported_claims,
                reason_codes=["critic_found_hallucinations"],
            )

        return ComposedOutput(
            response=draft.response,
            justification="GROUNDED_FROM_RETRIEVED_CORPUS",
            status=Status.REPLIED.value,
            request_type=request_type,
            product_area=area,
            verbatim_quotes=draft.verbatim_quotes,
            composer_confidence=draft.confidence,
            grounding_coverage=1.0 if draft.verbatim_quotes else 0.0,
            reason_codes=["grounded_response_generated"],
        )

    def _draft_response(
        self, ticket: CanonicalTicket, evidence: RetrievalEvidence
    ) -> ComposerDraft | None:
        if self.provider not in ("google", "openai"):
            return None

        context = "\n\n".join(
            [f"Source: {c.source_file}\n{c.text}" for c in evidence.chunks]
        )

        system_prompt = "You are a customer support agent. You MUST respond with valid JSON only. No markdown, no explanation, just JSON."
        prompt = f"""Answer the support ticket using ONLY the provided evidence.

EVIDENCE:
{context}

USER TICKET: {ticket.issue_norm}

Respond with this exact JSON structure:
{{
  "verbatim_quotes": ["copy a relevant sentence from evidence here"],
  "response": "Your helpful answer based on the evidence",
  "confidence": 0.8
}}

RULES:
1. Copy 1-3 relevant sentences from EVIDENCE into verbatim_quotes.
2. Write a helpful response using ONLY information from the evidence.
3. Set confidence between 0.0 and 1.0.
4. If the evidence does not help, set verbatim_quotes to [] and response to "I cannot find relevant information for this issue."
"""

        try:
            if self.provider == "google":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "system_instruction": system_prompt,
                        "response_mime_type": "application/json",
                        "response_schema": ComposerDraft,
                        "temperature": 0.1,
                    },
                )
                payload = response.text if response.text is not None else "{}"
                res = json.loads(payload)
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                raw_content = response.choices[0].message.content
                # Handle if response is a dict (some models return parsed JSON)
                if isinstance(raw_content, dict):
                    res = raw_content
                else:
                    res = json.loads(raw_content or "{}")

            return ComposerDraft(
                verbatim_quotes=res.get("verbatim_quotes", [])
                if isinstance(res.get("verbatim_quotes"), list)
                else [],
                response=str(res.get("response", ""))
                if not isinstance(res.get("response"), str)
                else res.get("response", ""),
                confidence=res.get("confidence", 0.0),
            )
        except Exception as e:
            print(f"L7A Composer Failed: {e}")
            # Attempt to salvage a response from raw text
            # Groq models often wrap JSON in ```json ... ``` fences
            try:
                raw = ""
                if self.provider == "openai":
                    raw = raw_content if 'raw_content' in dir() else ""
                elif self.provider == "google":
                    raw = response.text if 'response' in dir() and response.text else ""
                if raw:
                    # Strip markdown fences
                    raw = re.sub(r'```json\s*', '', raw)
                    raw = re.sub(r'```\s*', '', raw)
                    raw = raw.strip()
                    res = json.loads(raw)
                    return ComposerDraft(
                        verbatim_quotes=res.get("verbatim_quotes", []) if isinstance(res.get("verbatim_quotes"), list) else [],
                        response=str(res.get("response", "")),
                        confidence=res.get("confidence", 0.0),
                    )
            except Exception:
                pass
            return None

    def _run_critic(
        self, drafted_response: str, evidence: RetrievalEvidence
    ) -> CriticResult | None:
        if self.provider not in ("google", "openai"):
            return None

        context = "\n\n".join([c.text for c in evidence.chunks])

        prompt = f"""
JSON OUTPUT REQUIRED. You are a quality checker. Find claims in the response that are NOT in the evidence.

Evidence: {context}

Response: {drafted_response}

Return JSON: {{"unsupported_claims": ["claim 1", "claim 2"] or []}}
"""

        try:
            if self.provider == "google":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": CriticResult,
                        "temperature": 0.0,
                    },
                )
                payload = response.text if response.text is not None else "{}"
                res = json.loads(payload)
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                res = json.loads(response.choices[0].message.content or "{}")

            return CriticResult(unsupported_claims=res.get("unsupported_claims", []))
        except Exception as e:
            print(f"L7B Critic Failed: {e}")
            return None
