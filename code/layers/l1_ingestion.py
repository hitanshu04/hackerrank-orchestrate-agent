import re
import ftfy
from config import MIN_ISSUE_LENGTH, MAX_TOKEN_WINDOW, HEAD_KEEP, TAIL_KEEP
from models import RawTicket, CanonicalTicket

class InputNormalizer:
    """
    Layer 1: Deterministic Input Normalization
    Fast-fails junk tickets, truncates massive context to prevent token exhaustion,
    and sanitizes unicode to protect downstream LLM layers from prompt injection or noise.
    """
    def __init__(self):
        # Rough token approximation (1 token ~= 4 chars)
        self.char_limit = MAX_TOKEN_WINDOW * 4
        self.head_chars = HEAD_KEEP * 4
        self.tail_chars = TAIL_KEEP * 4

    def normalize(self, ticket: RawTicket) -> CanonicalTicket:
        # 1. Combine subject and issue
        subject = str(ticket.subject).strip() if ticket.subject else ""
        issue = str(ticket.issue).strip() if ticket.issue else ""
        company = str(ticket.company).strip() if ticket.company else ""
        
        full_text = f"Subject: {subject}\n\nIssue: {issue}".strip()
        
        # 2. Fast Fail for empty or ridiculously short tickets
        if len(issue) < MIN_ISSUE_LENGTH:
            return CanonicalTicket(
                issue_norm=issue,
                subject_norm=subject,
                company_norm=company,
                combined_text=full_text,
                row_index=ticket.row_index,
                fast_fail=True
            )
            
        # 3. Unicode Fixes (fixes broken encoding like 'â€œ' -> '"')
        cleaned_text = ftfy.fix_text(full_text)
        
        # 4. Basic Sanitation (Remove excessive repeating characters or newlines)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'(.)\1{20,}', r'\1\1\1', cleaned_text) # truncate repeated 'aaaaa...'
        
        # 5. Length Truncation (Head + Tail strategy)
        if len(cleaned_text) > self.char_limit:
            head = cleaned_text[:self.head_chars]
            tail = cleaned_text[-self.tail_chars:]
            cleaned_text = f"{head}\n\n...[TRUNCATED TO SAVE TOKENS]...\n\n{tail}"
            
        return CanonicalTicket(
            issue_norm=ftfy.fix_text(issue),
            subject_norm=ftfy.fix_text(subject),
            company_norm=company,
            combined_text=cleaned_text,
            row_index=ticket.row_index,
            fast_fail=False
        )
