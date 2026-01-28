# -*- coding: utf-8 -*-
import asyncio
from typing import Any, Dict, Literal
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter

from tenacity import retry, stop_after_attempt, wait_fixed


@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(5),
    reraise=True,
    # before_sleep=_print_exc_on_retry
)
async def _model_call_with_retry(
    model: DashScopeChatModel = None,
    formatter: DashScopeChatFormatter = None,
    sys_content: Any = None,
    user_content: Any = None,
    tool_json_schemas: list[dict] | None = None,
    tool_choice: Literal["auto", "none", "required"] | str | None = None,
    msg_name: str = "model_call",
    structured_model=None,
) -> Msg:
    msgs = [
        Msg("system", sys_content, "system"),
        Msg("user", user_content, "user"),
    ]

    format_msgs = await formatter.format(msgs=msgs)

    res = await model(
        format_msgs,
        tools=tool_json_schemas,
        tool_choice=tool_choice,
        structured_model=structured_model,
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


class UnifiedModelCallInterface:
    def __init__(
        self,
        base_model_name: str,
        vl_model_name: str,
        model_formatter_mapping: Dict[str, Any],
    ):
        self.base_model_name = base_model_name
        self.vl_model_name = vl_model_name
        self.model_formatter_mapping = model_formatter_mapping

    async def unified_model_call_interface(
        self,
        model_name: str = None,
        user_content: Any = None,
        sys_content: Any = None,
    ) -> Msg:
        model, formatter = self._load_model_and_formatter(
            model_name=model_name,
        )
        if sys_content is None:
            sys_content = (
                "You are a helpful AI assistant for database management."
            )

        raw_response = await _model_call_with_retry(
            model=model,
            formatter=formatter,
            sys_content=sys_content,
            user_content=user_content,
        )
        response = raw_response.content[0]["text"]
        return response

    def _load_model_and_formatter(self, model_name: str):
        model, formatter = self.model_formatter_mapping[model_name]
        return model, formatter

    def get_base_model_name(self) -> str:
        return self.base_model_name

    def get_vl_model_name(self) -> str:
        return self.vl_model_name
