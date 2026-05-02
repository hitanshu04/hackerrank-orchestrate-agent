import json
import time
from google import genai
from pydantic import BaseModel
from models import CanonicalTicket, SplitResult, SubIntent
from config import (
    LLM_PROVIDER,
    FAST_MODEL_PARAMS,
    MAX_INTENTS,
    FAST_CALL_SLEEP_SEC,
    OPENAI_BASE_URL,
)
import os
from openai import OpenAI


class SplitResultSchema(BaseModel):
    sub_intents: list[str]


class IntentSplitter:
    """
    Layer 2: Intent Splitter
    Uses a fast LLM (Gemini Flash) to split multi-intent tickets.
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
                raise RuntimeError(
                    "GROQ_API_KEY or OPENAI_API_KEY required for openai provider"
                )
            self.client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
            self.model_name = FAST_MODEL_PARAMS["openai"]["model"]
        else:
            raise NotImplementedError(
                f"Provider {self.provider} not fully implemented for L2 yet."
            )

    def split_intents(self, ticket: CanonicalTicket) -> SplitResult:
        if ticket.fast_fail:
            return SplitResult(
                sub_intents=[SubIntent(text=ticket.combined_text)], was_split=False
            )

        prompt = f"""
        You are a highly analytical support ticket parser.
        Your job is to read the following support ticket and split it into distinct, non-overlapping intents IF AND ONLY IF the user is asking about completely different problems (e.g. "My Visa card is blocked AND I need help logging into HackerRank").
        If the ticket is just one problem with multiple details, DO NOT split it. Keep it as one intent.
        Maximum allowed intents: {MAX_INTENTS}.
        
        Ticket Text:
        {ticket.combined_text}
        
        Return a JSON object matching the requested schema. The `sub_intents` list should contain the individual string texts of each distinct problem.
        """

        try:
            if self.provider == "google":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": SplitResultSchema,
                        "temperature": 0.1,  # Extremely low temperature for deterministic splitting
                    },
                )
                if FAST_CALL_SLEEP_SEC > 0:
                    time.sleep(FAST_CALL_SLEEP_SEC)

                payload = response.text if response.text is not None else "{}"
                result_dict = json.loads(payload)
                intents_list = result_dict.get("sub_intents", [ticket.combined_text])

                # Process results (same as openai path)
                if not isinstance(intents_list, list) or len(intents_list) == 0:
                    intents_list = [ticket.combined_text]
                intents_list = intents_list[:MAX_INTENTS]
                sub_intents = [
                    SubIntent(text=i.strip()) for i in intents_list if isinstance(i, str) and i.strip()
                ]
                if not sub_intents:
                    sub_intents = [SubIntent(text=ticket.combined_text)]
                return SplitResult(
                    sub_intents=sub_intents, was_split=len(sub_intents) > 1
                )

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                if FAST_CALL_SLEEP_SEC > 0:
                    time.sleep(FAST_CALL_SLEEP_SEC)
                result_dict = json.loads(response.choices[0].message.content or "{}")
                intents_list = result_dict.get("sub_intents", [ticket.combined_text])

                # If LLM failed to return a list, fallback to single intent
                if not isinstance(intents_list, list) or len(intents_list) == 0:
                    intents_list = [ticket.combined_text]

                # Cap the number of intents
                intents_list = intents_list[:MAX_INTENTS]

                sub_intents = [
                    SubIntent(text=i.strip()) for i in intents_list if i.strip()
                ]

                # If empty after stripping, fallback
                if not sub_intents:
                    sub_intents = [SubIntent(text=ticket.combined_text)]

                return SplitResult(
                    sub_intents=sub_intents, was_split=len(sub_intents) > 1
                )
        except Exception as e:
            print(f"L2 Splitting Failed: {e}. Falling back to single intent.")
            return SplitResult(
                sub_intents=[SubIntent(text=ticket.combined_text)], was_split=False
            )

        return SplitResult(
            sub_intents=[SubIntent(text=ticket.combined_text)], was_split=False
        )
