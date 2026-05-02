import csv
import os
import time

from models import RawTicket, PipelineState, ValidatedOutput, RequestType
from config import MAX_CONCURRENT
from retrieval.retriever import HybridRetriever
from layers.l1_ingestion import InputNormalizer
from layers.l2_intent_splitter import IntentSplitter
from layers.l3_domain_router import DomainRouter
from layers.l4_request_type import RequestTypeClassifier
from layers.l5_risk_engine import RiskPolicyEngine
from layers.l6_corpus_check import CorpusRetrieverCheck
from layers.l7_composer import GroundedComposer
from layers.l8_validator import OutputValidator
from utils.logger import log_execution_trace, log_transcript_entry


def process_ticket(
    row_idx: int,
    issue: str,
    subject: str,
    company: str,
    retriever: HybridRetriever,
    l1: InputNormalizer,
    l2: IntentSplitter,
    l3: DomainRouter,
    l4: RequestTypeClassifier,
    l5: RiskPolicyEngine,
    l6: CorpusRetrieverCheck,
    l7: GroundedComposer,
    l8: OutputValidator,
) -> ValidatedOutput:

    print(f"\n--- Processing Row {row_idx} ---")

    raw = RawTicket(issue=issue, subject=subject, company=company, row_index=row_idx)
    state = PipelineState(raw=raw)

    try:
        # Layer 1
        state.canonical = l1.normalize(raw)

        # Layer 2
        state.split = l2.split_intents(state.canonical)
        # Note: If there are multiple intents, we could loop here,
        # but to keep it 0% over-engineered and aligned with DAG,
        # we process the first/primary intent (which is the combined text if merged).
        # We'll just take the first sub-intent for L3-L7 to save time and complexity,
        # or process the whole ticket if L2 didn't split perfectly.
        # Actually, let's just use the canonical ticket for L3-L7, and let L4 handle the type logic.

        # Layer 3
        state.routing = l3.route(state.canonical)

        # Layer 4 - Classify each intent and aggregate
        type_decisions = [
            l4.classify(intent.text) for intent in state.split.sub_intents
        ]
        state.type_decision = l4.aggregate_multi_intent(type_decisions)

        # Layer 5
        state.risk = l5.evaluate(state.canonical)

        # Layer 6
        state.evidence = l6.retrieve_evidence(
            state.canonical, state.routing, state.risk.risk_flag
        )

        # Layer 7
        state.composed = l7.compose(
            state.canonical,
            state.risk,
            state.evidence,
            state.type_decision.request_type,
        )

        # Layer 8
        validated = l8.validate(state)

        # Log trace
        log_execution_trace(
            row_idx=row_idx,
            status=validated.status,
            request_type=validated.request_type,
            domain=state.routing.domain,
            risk=state.risk.risk_level,
            reason_codes=[c for c in validated.justification.split(" | ")],
        )

        return validated

    except Exception as e:
        print(f"Row {row_idx} Failed: {e}")
        return l8._fallback_row(
            issue,
            subject,
            company,
            row_idx,
            RequestType.PRODUCT_ISSUE.value,
            f"orchestrator_exception_{e}",
        )


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, "support_tickets", "support_tickets.csv")
    output_csv = r"d:\python_projects\HAckathon\hackerrank-orchestrate\support_tickets\output.csv"

    if not os.path.exists(input_csv):
        print(f"Input file not found: {input_csv}")
        return

    # Ensure indexes exist before processing
    index_dir = os.path.join(base_dir, "chroma_db")
    bm25_check = os.path.join(index_dir, "hackerrank_bm25.pkl")
    if not os.path.exists(index_dir) or not os.path.exists(bm25_check):
        print("Indexes not found. Building indexes first...")
        from retrieval.indexer import build_indexes
        build_indexes()
        print("Indexes built successfully.\n")

    print("Initializing Layers...")

    # Instantiate Shared Retriever
    retriever = HybridRetriever()

    # Instantiate Layers
    l1 = InputNormalizer()
    l2 = IntentSplitter()
    l3 = DomainRouter(retriever=retriever)
    l4 = RequestTypeClassifier()
    l5 = RiskPolicyEngine()
    l6 = CorpusRetrieverCheck(retriever=retriever)
    l7 = GroundedComposer()
    l8 = OutputValidator()

    outputs = []

    print("Starting Processing...")
    start_time = time.time()

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            issue = row.get("Issue", "")
            subject = row.get("Subject", "")
            company = row.get("Company", "")

            validated_row = process_ticket(
                row_idx=idx,
                issue=issue,
                subject=subject,
                company=company,
                retriever=retriever,
                l1=l1,
                l2=l2,
                l3=l3,
                l4=l4,
                l5=l5,
                l6=l6,
                l7=l7,
                l8=l8,
            )

            outputs.append(
                {
                    "issue": validated_row.issue,
                    "subject": validated_row.subject,
                    "company": validated_row.company,
                    "response": validated_row.response,
                    "product_area": validated_row.product_area,
                    "status": validated_row.status,
                    "request_type": validated_row.request_type,
                    "justification": validated_row.justification,
                }
            )

            # Configurable sleep between rows (defaults to 0)
            from config import ROW_SLEEP_SEC
            if ROW_SLEEP_SEC > 0:
                time.sleep(ROW_SLEEP_SEC)

    print(
        f"\nProcessing Complete. Validating Outputs. Time taken: {time.time() - start_time:.2f}s"
    )

    # Write to temp file first (Atomic Write per G14)
    tmp_out = output_csv + ".tmp"
    headers = [
        "issue",
        "subject",
        "company",
        "response",
        "product_area",
        "status",
        "request_type",
        "justification",
    ]

    with open(tmp_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(outputs)

    # Rename
    if os.path.exists(output_csv):
        os.remove(output_csv)
    os.rename(tmp_out, output_csv)

    print(f"Successfully wrote {len(outputs)} rows to {output_csv}")

    # Log AGENTS.md entry
    log_transcript_entry(
        title="Production Execution Run",
        user_prompt="Run main.py on support_tickets.csv",
        agent_summary=f"Processed {len(outputs)} rows successfully using the 8-layer DAG.",
        actions=["Read support_tickets.csv", "Executed Pipeline", "Wrote output.csv"],
        context={"rows": len(outputs), "status": "success"},
    )


if __name__ == "__main__":
    main()
