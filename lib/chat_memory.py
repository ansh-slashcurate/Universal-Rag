from llama_index.core.memory import ChatMemoryBuffer
from db.redis import upStash_chatStore
from lib.session import generate_session_id

def get_chat_memory(session_id: str | None = None) -> ChatMemoryBuffer:
    """
    Create a chat memory buffer for having a context for rag
    """
    reddis_key = f"sessions:{session_id}"
    return ChatMemoryBuffer.from_defaults(
        chat_store = upStash_chatStore,
        token_limit= 2000,
        chat_store_key = reddis_key
    )
