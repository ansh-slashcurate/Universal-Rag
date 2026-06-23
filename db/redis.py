from llama_index.storage.chat_store.upstash import UpstashChatStore
from llama_index.core.memory import ChatMemoryBuffer
import os
import uuid


upStash_chatStore = UpstashChatStore(
    redis_url = os.getenv('UPSTASH_REDIS_REST_URL'),
    redis_token =os.getenv('UPSTASH_REDIS_REST_TOKEN'),
    ttl = 7200
)

chat_memory =ChatMemoryBuffer.from_defaults(
    chat_store = upStash_chatStore,
    token_limit = 2000,
    chat_store_key =str(id(uuid))
)