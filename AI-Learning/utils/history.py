def estimate_token_count(text):
    # Rough estimate: 1 token ≈ 4 characters (for English)
    return len(text) // 4

def trim_chat_history(history_list, max_tokens=2048):
    trimmed = []
    total_tokens = 0
    for msg in reversed(history_list):
        msg_text = f"User: {msg['human']}\nBenji: {msg['ai']}" if 'human' in msg and 'ai' in msg else f"{msg['role']}: {msg['content']}"
        msg_tokens = estimate_token_count(msg_text)
        if total_tokens + msg_tokens > max_tokens:
            break
        trimmed.insert(0, msg)
        total_tokens += msg_tokens
    return trimmed

def get_history_text(history_list, max_tokens=2048):
    trimmed_history = trim_chat_history(history_list, max_tokens)
    return "\n".join([
        f"User: {msg['human']}\nBenji: {msg['ai']}" if 'human' in msg and 'ai' in msg else f"{msg['role']}: {msg['content']}" for msg in trimmed_history
    ])
