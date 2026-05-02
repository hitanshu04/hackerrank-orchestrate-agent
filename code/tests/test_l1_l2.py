import pytest
from layers.l1_ingestion import InputNormalizer
from layers.l2_intent_splitter import IntentSplitter
from models import RawTicket

def test_l1_normalizer_fast_fail():
    normalizer = InputNormalizer()
    ticket = RawTicket(issue="hi", row_index=0)
    result = normalizer.normalize(ticket)
    
    assert result.fast_fail is True
    assert "hi" in result.combined_text

def test_l1_normalizer_unicode():
    normalizer = InputNormalizer()
    ticket = RawTicket(issue="This is a test “quote”.", row_index=1)
    result = normalizer.normalize(ticket)
    
    assert result.fast_fail is False
    assert "quote" in result.combined_text

def test_l2_intent_splitter():
    # Only test if GEMINI_API_KEY is available (in a real scenario we'd mock this)
    import os
    if os.getenv("GEMINI_API_KEY") == "dummy_key_for_testing" or not os.getenv("GEMINI_API_KEY"):
        pytest.skip("Skipping L2 test because no GEMINI_API_KEY is set.")
        
    splitter = IntentSplitter()
    ticket = RawTicket(issue="My visa card is lost AND I can't login to hackerrank", row_index=2)
    normalizer = InputNormalizer()
    canonical = normalizer.normalize(ticket)
    
    result = splitter.split_intents(canonical)
    
    # It should split into 2 intents
    assert result.was_split is True
    assert len(result.sub_intents) == 2
