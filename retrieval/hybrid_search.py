import hashlib
import math
import re
from collections import Counter

from qdrant_client.models import SparseVector


SPARSE_VECTOR_NAME = "text_sparse"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _token_to_index(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def text_to_sparse_vector(text: str) -> SparseVector:
    tokens = TOKEN_PATTERN.findall((text or "").lower())
    if not tokens:
        return SparseVector(indices=[], values=[])

    token_counts = Counter(tokens)
    norm = math.sqrt(sum(count * count for count in token_counts.values())) or 1.0

    index_weights = {}
    for token, count in token_counts.items():
        index = _token_to_index(token)
        index_weights[index] = index_weights.get(index, 0.0) + (count / norm)

    sorted_items = sorted(index_weights.items())
    return SparseVector(
        indices=[index for index, _ in sorted_items],
        values=[weight for _, weight in sorted_items],
    )
