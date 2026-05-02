import re
from models import CanonicalTicket, RiskDecision, RiskLevel

class RiskPolicyEngine:
    """
    Layer 5: Risk Policy Engine (ZERO LLM)
    Uses strict Python regex to identify sensitive topics.
    Does NOT auto-escalate unless the score is CRITICAL (9-10).
    Otherwise, it flags the ticket for L6 to determine if we have corpus evidence.
    """
    def __init__(self):
        # (patterns, base_risk_score)
        self.risk_patterns: dict[str, tuple[list[str], int]] = {
            "stolen_card": ([
                r"\bstolen\s+(card|cheque|money)\b",
                r"\blost\s+(or\s+)?stolen\b",
                r"\blost\s+(my\s+)?(visa\s+)?card\b",
            ], 5),
            
            "identity_theft": ([
                r"\bidentity\s+(theft|stolen)\b",
                r"\bidentity\s+has\s+been\s+stolen\b",
            ], 6),
            
            "billing_dispute": ([
                r"\bdispute\s+(a\s+)?charge\b",
                r"\brefund\b",
                r"\bchargeback\b",
            ], 5),
            
            "subscription_change": ([
                r"\bpause\s+(our\s+)?subscription\b",
                r"\bcancel\s+(my\s+)?(subscription|plan|account)\b",
            ], 8),
            
            "score_manipulation": ([
                r"\bincrease\s+my\s+score\b",
                r"\breview\s+my\s+(answers|score)\b",
                r"\bgraded\s+(me\s+)?unfairly\b",
            ], 9),
            
            "security_vuln": ([
                r"\bvulnerability\b",
                r"\bbug\s+bounty\b",
                r"\bsecurity\s+(breach|flaw|issue)\b",
            ], 6),
            
            "account_access": ([
                r"\blost\s+access\b",
                r"\brestore\s+(my\s+)?access\b",
                r"\blocked\s+out\b",
            ], 5),
            
            "critical_outage": ([
                r"\bsite\s+is\s+down\b",
                r"\ball\s+requests?\s+(are\s+)?failing\b",
                r"\bcompletely\s+(broken|down|not\s+working)\b",
            ], 7),
            
            "prompt_injection": ([
                r"\b(display|show|reveal)\s+(all|the)\s+(internal|system|hidden)\b",
                r"\bignore\s+(previous|prior)\s+instructions?\b",
                r"\bdelete\s+all\s+files\b",
            ], 9),
            
            "payment_order": ([
                r"\border\s+id\b",
                r"\bpayment\s+.*order\b",
            ], 7),
        }

    def evaluate(self, ticket: CanonicalTicket) -> RiskDecision:
        text = ticket.combined_text.lower()
        max_score = 0
        tags = []
        reason_codes = []
        
        for tag, (patterns, score) in self.risk_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    tags.append(tag)
                    if score > max_score:
                        max_score = score
                    break # Stop checking patterns for this tag if one matches
                    
        # Determine risk level
        if max_score >= 9:
            level = RiskLevel.CRITICAL
        elif max_score >= 6:
            level = RiskLevel.HIGH
        elif max_score >= 3:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW
            
        force_escalate = (level == RiskLevel.CRITICAL)
        risk_flag = (max_score >= 5)
        
        if force_escalate:
            reason_codes.append("RISK_CRITICAL_ESCALATE")
            
        return RiskDecision(
            risk_level=level.value,
            risk_tags=tags,
            force_escalate=force_escalate,
            risk_flag=risk_flag,
            risk_score=max_score,
            reason_codes=reason_codes
        )
