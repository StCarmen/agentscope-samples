# -*- coding: utf-8 -*-
import asyncio
from typing import Any, Dict, Literal, Type
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import BaseModel

from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter

from alias.agent.utils.constants import MODEL_MAX_RETRIES


@retry(
    stop=stop_after_attempt(MODEL_MAX_RETRIES),
    wait=wait_fixed(5),
    reraise=True,
    # before_sleep=_print_exc_on_retry
)
async def model_call_with_retry(
    model: DashScopeChatModel = None,
    formatter: DashScopeChatFormatter = None,
    messages: list[dict[str, Any]] = None,
    tool_json_schemas: list[dict] | None = None,
    tool_choice: Literal["auto", "none", "required"] | str | None = None,
    structured_model: Type[BaseModel] | None = None,
    msg_name: str = "model_call",
    **kwargs: Any,
) -> Msg:
    format_messages = await formatter.format(msgs=messages)

    res = await model(
        messages=format_messages,
        tools=tool_json_schemas,
        tool_choice=tool_choice,
        structured_model=structured_model,
        kwargs=kwargs,
    )
    if model.stream:
        msg = Msg(msg_name, [], "assistant")
        async for content_chunk in res:
            msg.content = content_chunk.content
        # Add a tiny sleep to yield the last message object in the
        # message queue
        await asyncio.sleep(0.001)
    else:
        msg = Msg(msg_name, list(res.content), "assistant")
    return msg


class LLMCallManager:
    def __init__(
        self,
        base_model_name: str,
        vl_model_name: str,
        model_formatter_mapping: Dict[str, Any],
    ):
        self.base_model_name = base_model_name
        self.vl_model_name = vl_model_name
        self.model_formatter_mapping = model_formatter_mapping

    def get_base_model_name(self) -> str:
        return self.base_model_name

    def get_vl_model_name(self) -> str:
        return self.vl_model_name

    async def __call__(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "required"] | str | None = None,
        structured_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> str:
        model, formatter = self.model_formatter_mapping[model_name]
        raw_response = await model_call_with_retry(
            model=model,
            formatter=formatter,
            messages=messages,
            tool_json_schemas=tools,
            tool_choice=tool_choice,
            structured_model=structured_model,
            msg_name="model_call",
            kwargs=kwargs,
        )
        response = raw_response.content[0]["text"]
        return response
