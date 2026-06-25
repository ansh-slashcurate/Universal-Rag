from llama_index.storage.chat_store.upstash import UpstashChatStore
import os


upStash_chatStore = UpstashChatStore(
    redis_url = os.getenv('UPSTASH_REDIS_REST_URL'),
    redis_token =os.getenv('UPSTASH_REDIS_REST_TOKEN'),
    ttl = 7200
)
