import os
import datetime

# Determine the log file path based on OS per AGENTS.md §2
if os.name == 'nt':
    LOG_PATH = os.path.join(os.environ.get('USERPROFILE', ''), 'hackerrank_orchestrate', 'log.txt')
else:
    LOG_PATH = os.path.join(os.environ.get('HOME', ''), 'hackerrank_orchestrate', 'log.txt')

def log_transcript_entry(title: str, user_prompt: str, agent_summary: str, actions: list[str], context: dict):
    """
    Appends a turn to the log.txt transcript in the exact format required by AGENTS.md §5.2.
    NOTE: In an actual agent implementation, the agent itself writes this. This is a helper for programmatic logging if needed.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    
    timestamp = datetime.datetime.now().astimezone().isoformat()
    
    entry = f"## [{timestamp}] {title[:80]}\n\n"
    entry += "User Prompt (verbatim, secrets redacted):\n"
    entry += f"{user_prompt}\n\n"
    entry += "Agent Response Summary:\n"
    entry += f"{agent_summary}\n\n"
    entry += "Actions:\n"
    for action in actions:
        entry += f"* {action}\n"
    
    entry += "\nContext:\n"
    for k, v in context.items():
        entry += f"{k}={v}\n"
    entry += "\n"
    
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(entry)


def log_execution_trace(row_idx: int, status: str, request_type: str, domain: str, risk: str, reason_codes: list[str]):
    """
    System-level structured trace for our own debugging and evaluation. 
    This is separate from the AI chat transcript. We'll write this to a local execution.log
    """
    trace_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "execution.log")
    
    codes_str = ", ".join(reason_codes)
    entry = f"[ROW {row_idx:02d}] status={status} type={request_type} domain={domain} risk={risk} codes=[{codes_str}]\n"
    
    with open(trace_path, 'a', encoding='utf-8') as f:
        f.write(entry)

