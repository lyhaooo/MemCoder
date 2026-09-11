"""Model provider interfaces and HTTP implementations."""

from memcoder.models.base import StructuredModel
from memcoder.models.openai_compatible import OpenAICompatibleModel

__all__ = ["OpenAICompatibleModel", "StructuredModel"]

