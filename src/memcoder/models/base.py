"""Protocol for structured model calls."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from memcoder.domain import ModelResult

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredModel(Protocol):
    """A model that returns JSON validated against a Pydantic schema."""

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
    ) -> tuple[SchemaT, ModelResult]: ...

