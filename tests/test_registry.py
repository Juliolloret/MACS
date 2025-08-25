from utils import set_status_callback
from agents.registry import load_agents

def test_load_agents_uses_log_status():
    messages = []
    set_status_callback(messages.append)
    try:
        load_agents()
    finally:
        set_status_callback(print)
    assert any("Starting Agent Loading" in msg for msg in messages)
    assert any("Agent Loading Complete" in msg for msg in messages)
