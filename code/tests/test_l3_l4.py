import pytest
from models import CanonicalTicket, Domain, RequestType, TypeDecision
from layers.l3_domain_router import DomainRouter
from layers.l4_request_type import RequestTypeClassifier

@pytest.fixture
def router():
    return DomainRouter()

@pytest.fixture
def classifier():
    return RequestTypeClassifier()

def test_l3_domain_routing_hackerrank(router):
    # Lexical keyword 'hackerrank' should boost it
    ticket = CanonicalTicket(
        issue_norm="How do I add extra time for a candidate?",
        subject_norm="",
        company_norm="hackerrank",
        combined_text="How do I add extra time for a candidate on hackerrank?",
        row_index=0
    )
    decision = router.route(ticket)
    # The company meta + lexical should strongly favor hackerrank
    assert decision.domain == Domain.HACKERRANK.value

def test_l4_request_type_aggregation(classifier):
    # Test that the priority aggregation works: invalid(4) > bug(3) > feature(2) > product(1)
    d1 = TypeDecision(request_type=RequestType.PRODUCT_ISSUE.value, confidence=0.9)
    d2 = TypeDecision(request_type=RequestType.BUG.value, confidence=0.8)
    
    agg = classifier.aggregate_multi_intent([d1, d2])
    assert agg.request_type == RequestType.BUG.value
