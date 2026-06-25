import uuid

def generate_session_id(session_id: str | None = None) -> str:
    """Return the provided session id, or generate a new one."""
    if session_id and session_id.strip():
        return session_id.strip()
    return str(uuid.uuid4())
