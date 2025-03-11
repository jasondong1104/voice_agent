from typing import AsyncIterable, List, Optional, Callable, Any, Dict
from livekit.agents import llm, utils
from livekit.agents.llm import LLMStream, ChatMessage, ChatChunk, Choice, ChoiceDelta
from llm_agent import MyMessagesState, WorkFlow
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
import uuid
from api_config import config, update_config
from langgraph.graph import StateGraph

class LangchainLLMStream(LLMStream):
    def __init__(self, llm: 'LangchainLLM', **kwargs) -> None:
        super().__init__(llm, **kwargs)
        self._request_id = str(uuid.uuid4())
        
    async def _run(self) -> None:
        try:
            # Convert LiveKit chat format to Langchain format
            print('运用agent 前 的最后三条消息:', self.chat_ctx.messages[-3:])
            messages = []
            for msg in self.chat_ctx.messages[1:]:
                message = {
                    "role": msg.role,
                    "content": msg.content or ""  # Changed from text to content
                }
                messages.append(message)

            # Changed to use astream() for async operation
            result = await llm_workflow.ainvoke({"messages": messages})
            print('agent 内部 的最后3条消息:', [ele.content + '\n' for ele in result['messages'][-3:]])

            
                # Process the response
            if isinstance(result, str):
                response_text = result
            elif isinstance(result, dict) and 'messages' in result:
                response_text = result['messages'][-1].content
            else:
                response_text = "抱歉，我现在无法正确回答，请稍后再试。"

            # Create and send the chat chunk using our generated request_id
            chunk = ChatChunk(
                request_id=self._request_id,
                choices=[
                        Choice(
                        delta=ChoiceDelta(
                            role="assistant",
                            content=response_text
                                        )
                        )
                    ]
                )
            await self._event_ch.send(chunk)

        except Exception as e:
            print(f"LangchainLLMStream error: {e}")
            # Send error response using our generated request_id
            error_chunk = ChatChunk(
                request_id=self._request_id,
                choices=[
                    Choice(
                        delta=ChoiceDelta(
                            role="assistant",
                            content="抱歉，我现在无法正确回答，请稍后再试。"
                        )
                    )
                ]
            )
            await self._event_ch.send(error_chunk)

class LangchainLLM(llm.LLM):
    def __init__(self):
        super().__init__()
        self._metrics_handlers = []
        self._config_handlers = []
        self._reinitialize_llm()

    def update_config(self, new_config: dict):
        update_config(new_config)
        # 通知所有配置处理器
        for handler in self._config_handlers:
            handler(new_config)
        # 重新初始化LLM
        self._reinitialize_llm()

    def _reinitialize_llm(self):
        # 重新初始化LLM和工作流
        global llm_workflow
        llm_workflow = WorkFlow().get_workflow()

    def on_config_update(self, callback: Callable[[Dict], None]):
        """注册配置更新回调"""
        self._config_handlers.append(callback)

    def on(self, event: str, callback: Callable[[Any], None] | None = None):
        if event == "metrics_collected":
            if callback:
                self._metrics_handlers.append(callback)
        return super().on(event, callback)

    def chat(
        self,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: Optional[llm.FunctionContext] = None,
        temperature: float | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Any = None,
    ) -> LLMStream:
        return LangchainLLMStream(
            llm=self,  # Changed from llm_instance to llm
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options
        )
