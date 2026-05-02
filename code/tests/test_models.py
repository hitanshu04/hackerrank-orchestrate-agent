import pytest
from models import RawTicket, Status, RequestType

def test_raw_ticket_instantiation():
    ticket = RawTicket(issue="Test issue", subject="Test subject", company="HackerRank", row_index=1)
    assert ticket.issue == "Test issue"
    assert ticket.subject == "Test subject"
    assert ticket.company == "HackerRank"
    assert ticket.row_index == 1

def test_enum_values():
    assert Status.REPLIED.value == "replied"
    assert Status.ESCALATED.value == "escalated"
    assert RequestType.PRODUCT_ISSUE.value == "product_issue"
    assert RequestType.BUG.value == "bug"
    assert RequestType.FEATURE_REQUEST.value == "feature_request"
    assert RequestType.INVALID.value == "invalid"
