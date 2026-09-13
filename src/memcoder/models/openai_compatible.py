"""OpenAI Responses API and OpenAI-compatible Chat Completions adapter."""

from __future__ import annotations

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from memcoder.domain import ModelResult, TokenUsage

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ModelAPIError(RuntimeError):
    """Raised when the configured model endpoint returns an invalid response."""


class OpenAICompatibleModel:
    """Small HTTP client supporting Responses and Chat Completions APIs.

    Responses mode is intended for OpenAI. Chat Completions mode is the
    compatibility path for providers such as DeepSeek or local gateways.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_mode: str = "responses",
        timeout: float = 90.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for a live model")
        if api_mode not in {"responses", "chat_completions"}:
            raise ValueError("api_mode must be 'responses' or 'chat_completions'")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_mode = api_mode
        self._client = client or httpx.Client(timeout=timeout)

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
    ) -> tuple[SchemaT, ModelResult]:
        payload = (
            self._responses_payload(system, prompt, schema)
            if self.api_mode == "responses"
            else self._chat_payload(system, prompt, schema)
        )
        endpoint = "responses" if self.api_mode == "responses" else "chat/completions"
        try:
            response = self._client.post(
                f"{self.base_url}/{endpoint}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelAPIError(f"Model request failed: {exc}") from exc

        raw_text = (
            self._responses_text(body)
            if self.api_mode == "responses"
            else self._chat_text(body)
        )
        try:
            parsed = schema.model_validate_json(raw_text)
        except Exception as exc:
            raise ModelAPIError(
                f"Model returned invalid structured output: {raw_text[:500]}"
            ) from exc

        usage = self._usage(body)
        return parsed, ModelResult(
            data=parsed.model_dump(mode="json"), usage=usage, raw_text=raw_text
        )

    def _responses_payload(
        self, system: str, prompt: str, schema: type[BaseModel]
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "instructions": system,
            "input": prompt,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__.lower(),
                    "schema": schema.model_json_schema(),
                }
            },
        }

    def _chat_payload(self, system: str, prompt: str, schema: type[BaseModel]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            },
        }

    @staticmethod
    def _responses_text(body: dict[str, Any]) -> str:
        if isinstance(body.get("output_text"), str):
            return body["output_text"]
        for item in body.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return content["text"]
        raise ModelAPIError("Responses API body did not contain output text")

    @staticmethod
    def _chat_text(body: dict[str, Any]) -> str:
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelAPIError("Chat Completions body did not contain message content") from exc
        if not isinstance(content, str):
            raise ModelAPIError("Chat Completions message content is not text")
        return content

    @staticmethod
    def _usage(body: dict[str, Any]) -> TokenUsage:
        usage = body.get("usage") or {}
        return TokenUsage(
            input_tokens=int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
            output_tokens=int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
        )
