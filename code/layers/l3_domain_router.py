import json
import os
from google import genai
from pydantic import BaseModel
from config import COMPANY_WEIGHT, LEXICAL_WEIGHT, RETRIEVAL_PROBE_WEIGHT, CONTRADICTION_PENALTY, ABSTAIN_MARGIN, LLM_PROVIDER, FAST_MODEL_PARAMS
from models import CanonicalTicket, RoutingDecision, Domain
from retrieval.retriever import HybridRetriever

class DomainRouterLLMResponse(BaseModel):
    domain: str
    confidence: float

class DomainRouter:
    """
    Layer 3: Domain Router
    Determines if the ticket is for hackerrank, claude, or visa.
    Uses Python heuristics (Company metadata, Lexical keywords, BM25 probes).
    Falls back to a fast LLM if the result is ambiguous.
    """
    def __init__(self, retriever: HybridRetriever = None):
        self.retriever = retriever if retriever else HybridRetriever()
        self.domains = [Domain.HACKERRANK, Domain.CLAUDE, Domain.VISA]
        
        self.lexical_keywords = {
            Domain.HACKERRANK: ["hackerrank", "test", "candidate", "plagiarism", "ide", "compiler", "test case"],
            Domain.CLAUDE: ["claude", "anthropic", "opus", "sonnet", "haiku", "message", "conversation", "ai"],
            Domain.VISA: ["visa", "card", "stolen", "lost", "credit", "debit", "transaction", "dispute", "merchant"]
        }
        
        self.provider = LLM_PROVIDER
        if self.provider == "google":
            api_key = os.getenv("GEMINI_API_KEY", "dummy_key_for_testing")
            self.client = genai.Client(api_key=api_key)
            self.model_name = FAST_MODEL_PARAMS["google"]["model"]

    def _get_lexical_score(self, text: str, domain: Domain) -> float:
        text_lower = text.lower()
        score = 0.0
        for keyword in self.lexical_keywords[domain]:
            if keyword in text_lower:
                score += 1.0
        return min(score / 3.0, 1.0)  # Cap at 1.0 if 3+ keywords match

    def _get_retrieval_score(self, text: str, domain: Domain) -> float:
        # Probe the BM25 index. If we get good results, it's a strong signal.
        chunks = self.retriever.retrieve(text, domain=domain.value)
        if not chunks:
            return 0.0
        # Average score of top 3 chunks
        top_scores = [c.score for c in chunks[:3]]
        return sum(top_scores) / len(top_scores)

    def route(self, ticket: CanonicalTicket) -> RoutingDecision:
        scores = {d: 0.0 for d in self.domains}
        reason_codes = []
        
        company_norm = ticket.company_norm.lower()
        
        for domain in self.domains:
            # 1. Company Signal
            c_score = 1.0 if company_norm == domain.value else 0.0
            
            # 2. Lexical Signal
            l_score = self._get_lexical_score(ticket.combined_text, domain)
            
            # 3. Retrieval Probe
            r_score = self._get_retrieval_score(ticket.issue_norm, domain)
            
            total_score = (c_score * COMPANY_WEIGHT) + (l_score * LEXICAL_WEIGHT) + (r_score * RETRIEVAL_PROBE_WEIGHT)
            scores[domain] = total_score
            
        # 4. Contradiction Penalty
        best_domain = max(scores, key=scores.get)
        contradiction = False
        if company_norm in [d.value for d in self.domains] and company_norm != best_domain.value:
            # The stated company contradicts the content!
            contradiction = True
            reason_codes.append("contradiction_detected")
            # Penalize the company signal
            scores[Domain(company_norm)] -= (COMPANY_WEIGHT * CONTRADICTION_PENALTY)
            # Re-evaluate
            best_domain = max(scores, key=scores.get)
            
        # 5. Margin Check
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1]
        
        if sorted_scores[0] < 0.2 or margin < ABSTAIN_MARGIN:
            reason_codes.append("margin_too_low_ambiguous")
            return self._llm_fallback(ticket, scores, reason_codes)
            
        return RoutingDecision(
            domain=best_domain.value,
            confidence=sorted_scores[0],
            scores={k.value: v for k, v in scores.items()},
            reason_codes=reason_codes,
            contradiction_detected=contradiction
        )

    def _llm_fallback(self, ticket: CanonicalTicket, current_scores: dict, reason_codes: list) -> RoutingDecision:
        if self.provider not in ("google", "openai"):
            return RoutingDecision(
                domain=Domain.AMBIGUOUS.value,
                confidence=0.0,
                scores={k.value: v for k, v in current_scores.items()},
                reason_codes=reason_codes + ["llm_fallback_skipped"]
            )
            
        prompt = f"""
        You must route this support ticket to exactly one of the following domains: hackerrank, claude, or visa.
        If it's impossible to tell, route to "ambiguous".
        
        Ticket Company Meta: {ticket.company_norm}
        Ticket Subject: {ticket.subject_norm}
        Ticket Issue: {ticket.issue_norm}
        
        Return JSON matching the schema with domain and confidence (0.0 to 1.0).
        """
        
        try:
            if self.provider == "google":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": DomainRouterLLMResponse,
                        "temperature": 0.1,
                    },
                )
                res = json.loads(response.text)
            elif self.provider == "openai":
                from openai import OpenAI
                from config import OPENAI_BASE_URL, FAST_MODEL_PARAMS
                api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
                client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
                model = FAST_MODEL_PARAMS["openai"]["model"]
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                res = json.loads(response.choices[0].message.content or "{}")

            domain_val = res.get("domain", "ambiguous").lower()
            if domain_val not in [d.value for d in self.domains]:
                domain_val = "ambiguous"
                
            return RoutingDecision(
                domain=domain_val,
                confidence=res.get("confidence", 0.5),
                scores={k.value: v for k, v in current_scores.items()},
                reason_codes=reason_codes + ["llm_fallback_used"],
                contradiction_detected=False
            )
        except Exception as e:
            print(f"L3 LLM Fallback Failed: {e}")
            return RoutingDecision(
                domain=Domain.AMBIGUOUS.value,
                confidence=0.0,
                scores={k.value: v for k, v in current_scores.items()},
                reason_codes=reason_codes + ["llm_fallback_failed"]
            )
