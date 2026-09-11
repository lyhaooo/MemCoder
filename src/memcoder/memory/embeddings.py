"""Pluggable local and HTTP embedding providers."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx


class Embedder(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]: ...


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return vector if norm == 0 else [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


class HashingEmbedder:
    """Deterministic, dependency-free feature hashing for local development.

    It is intentionally modest: useful for tests, demos, and lexical-semantic
    baselines, but not presented as a replacement for a trained embedding model.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 32:
            raise ValueError("dimensions must be at least 32")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return normalize(vector)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        lowered = text.lower()
        words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered)
        char_ngrams = [lowered[i : i + 3] for i in range(max(0, len(lowered) - 2))]
        return words + char_ngrams


class OpenAICompatibleEmbedder:
    """Embedding client for OpenAI-compatible `/embeddings` endpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        dimensions: int,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimensions = dimensions
        self._client = client or httpx.Client(timeout=timeout)

    def embed(self, text: str) -> list[float]:
        payload: dict[str, object] = {"model": self.model, "input": text}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        response = self._client.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
        )
        response.raise_for_status()
        vector = response.json()["data"][0]["embedding"]
        if not isinstance(vector, list):
            raise TypeError("Embedding endpoint returned an invalid vector")
        self.dimensions = len(vector)
        return normalize([float(value) for value in vector])

