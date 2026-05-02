import json
import os
import time
from google import genai
from pydantic import BaseModel
from models import TypeDecision, RequestType, REQUEST_TYPE_PRIORITY
from config import LLM_PROVIDER, FAST_MODEL_PARAMS, FAST_CALL_SLEEP_SEC, OPENAI_BASE_URL
from openai import OpenAI


class RequestTypeLLMResponse(BaseModel):
    request_type: str
    confidence: float
    reason: str


class RequestTypeClassifier:
    """
    Layer 4: Request Type Classifier
    Uses a fast LLM to categorize the intent into one of 4 strict buckets:
    product_issue, feature_request, bug, invalid.
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
            self.model_name = FAST_MODEL_PARAMS["google"]["model"]
        elif self.provider == "openai":
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY or OPENAI_API_KEY required")
            self.client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
            self.model_name = FAST_MODEL_PARAMS["openai"]["model"]

        self.valid_types = [t.value for t in RequestType]

    def classify(self, text: str) -> TypeDecision:
        if self.provider not in ("google", "openai"):
            return TypeDecision(
                request_type=RequestType.PRODUCT_ISSUE.value,
                confidence=0.5,
                reason_codes=["default_fallback"],
            )

        prompt = f"""
        Classify the following support ticket text into EXACTLY ONE of these 4 categories:
        1. "product_issue" (User needs help, how-to, account issues, general support)
        2. "feature_request" (User is asking for a new feature, improvement, or integration)
        3. "bug" (User is reporting something broken, error messages, system down)
        4. "invalid" (Spam, gibberish, completely irrelevant)
        
        Ticket Text:
        {text}
        
        Return JSON matching the schema. The request_type must be one of the 4 strings exactly.
        """

        try:
            if self.provider == "google":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": RequestTypeLLMResponse,
                        "temperature": 0.1,
                    },
                )
                if FAST_CALL_SLEEP_SEC > 0:
                    time.sleep(FAST_CALL_SLEEP_SEC)
                payload = response.text if response.text is not None else "{}"
                res = json.loads(payload)
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                if FAST_CALL_SLEEP_SEC > 0:
                    time.sleep(FAST_CALL_SLEEP_SEC)
                res = json.loads(response.choices[0].message.content or "{}")

            req_type = res.get("request_type", "").lower()
            if req_type not in self.valid_types:
                req_type = RequestType.PRODUCT_ISSUE.value  # Safe default

            return TypeDecision(
                request_type=req_type,
                confidence=res.get("confidence", 0.8),
                reason_codes=[res.get("reason", "")],
            )
        except Exception as e:
            print(f"L4 Classification Failed: {e}")
            return TypeDecision(
                request_type=RequestType.PRODUCT_ISSUE.value,
                confidence=0.0,
                reason_codes=["llm_failure"],
            )

    def aggregate_multi_intent(self, decisions: list[TypeDecision]) -> TypeDecision:
        """
        If a ticket was split into multiple intents, we classify each and then
        take the highest priority one using the deterministic tie-breaker:
        invalid > bug > feature_request > product_issue
        """
        if not decisions:
            return TypeDecision(
                request_type=RequestType.PRODUCT_ISSUE.value, confidence=0.0
            )

        best_decision = decisions[0]
        best_priority = REQUEST_TYPE_PRIORITY.get(
            RequestType(best_decision.request_type), 0
        )

        for d in decisions[1:]:
            prio = REQUEST_TYPE_PRIORITY.get(RequestType(d.request_type), 0)
            if prio > best_priority:
                best_decision = d
                best_priority = prio

        return best_decision
