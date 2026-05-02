from models import CanonicalTicket, RoutingDecision, RetrievalEvidence, SupportStatus
from retrieval.retriever import HybridRetriever
from config import SIMILARITY_FLOOR

class CorpusRetrieverCheck:
    """
    Layer 6: Corpus Retriever
    Retrieves evidence from the designated domain's index.
    Determines if the ticket is "supported" based on a distance floor threshold.
    """
    def __init__(self, retriever: HybridRetriever = None):
        self.retriever = retriever if retriever else HybridRetriever()

    def retrieve_evidence(self, ticket: CanonicalTicket, routing: RoutingDecision, risk_flag: bool = False) -> RetrievalEvidence:
        domain = routing.domain
        
        if domain == "ambiguous":
            # If domain is ambiguous, we can't reliably scope the search.
            # We could search all, but per specs we fail-closed if ambiguous.
            return RetrievalEvidence(
                support_status=SupportStatus.UNSUPPORTED.value,
                domain_searched=domain,
                reason_codes=["domain_ambiguous_unsupported"]
            )
            
        chunks = self.retriever.retrieve(ticket.issue_norm, domain=domain)
        
        if not chunks:
            return RetrievalEvidence(
                support_status=SupportStatus.UNSUPPORTED.value,
                domain_searched=domain,
                reason_codes=["no_chunks_found"]
            )
            
        best_score = chunks[0].score
        
        # Count chunks above floor
        chunks_above_floor = sum(1 for c in chunks if c.score >= SIMILARITY_FLOOR)
        
        from config import STRONG_SUPPORT_MIN_CHUNKS # Usually 2
        
        # Check against SIMILARITY_FLOOR and risk_flag
        if risk_flag and chunks_above_floor < STRONG_SUPPORT_MIN_CHUNKS:
            status = SupportStatus.UNSUPPORTED.value
            reason_code = f"risk_flagged_but_only_{chunks_above_floor}_strong_chunks"
        elif best_score >= SIMILARITY_FLOOR:
            status = SupportStatus.SUPPORTED.value
            reason_code = "above_similarity_floor"
        elif best_score >= (SIMILARITY_FLOOR - 0.1): # e.g. 0.25 if floor is 0.35
            status = SupportStatus.WEAK_SUPPORT.value
            reason_code = "weak_similarity_floor"
        else:
            status = SupportStatus.UNSUPPORTED.value
            reason_code = "below_similarity_floor"
            
        return RetrievalEvidence(
            chunks=chunks,
            support_status=status,
            best_score=best_score,
            domain_searched=domain,
            reason_codes=[reason_code]
        )
