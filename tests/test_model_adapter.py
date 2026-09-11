import json

import httpx

from memcoder.domain import PlanOutput
from memcoder.models.openai_compatible import OpenAICompatibleModel


def test_responses_adapter_uses_json_schema_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["store"] is False
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "summary": "plan",
                                        "steps": ["step"],
                                        "target_language": "python",
                                    }
                                ),
                            }
                        ]
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenAICompatibleModel(
        api_key="test",
        model="test-model",
        client=client,
        api_mode="responses",
    )
    parsed, result = model.generate(system="system", prompt="prompt", schema=PlanOutput)

    assert parsed.summary == "plan"
    assert result.usage.total_tokens == 15


def test_chat_completions_adapter() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"summary": "plan", "steps": ["step"], "target_language": "python"}
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenAICompatibleModel(
        api_key="test",
        model="compatible-model",
        base_url="https://example.test/v1",
        client=client,
        api_mode="chat_completions",
    )
    _, result = model.generate(system="system", prompt="prompt", schema=PlanOutput)
    assert result.usage.total_tokens == 10
