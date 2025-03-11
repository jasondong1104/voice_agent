import logging
import os
import asyncio
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local")
# python agent.py --server-url wss://your-livekit-server --api-key your-api-key --api-secret your-api-secret

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, openai, deepgram, silero, turn_detector, elevenlabs

from backend.fastapi_server import run_fastapi
from workflow.langchain_llm import LangchainLLM

logger = logging.getLogger("voice-agent")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # 启动FastAPI服务器
    asyncio.create_task(run_fastapi())
    
    global llm_instance
    llm_instance = LangchainLLM()

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            """个人资料：你是一位新入职山西太原市杨帆高中的物理老师；
            最近几个月在负责学校的招生工作，正在与学生家长进行电话联系。"""
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(language="zh"),
        llm=llm_instance,
        tts=elevenlabs.TTS(language='zh',
        ),
        turn_detector=turn_detector.EOUModel(),
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
        chat_ctx=initial_ctx,
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    agent.start(ctx.room, participant)

    await agent.say("你好，我是太原市杨帆高中的物理老师。", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
