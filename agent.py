import logging
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_config import update_config, get_default_config
from langchain_llm import LangchainLLM
# python agent.py --server-url wss://your-livekit-server --api-key your-api-key --api-secret your-api-secret
#todo: 1. 前端添加模型 baseurl 文本框 2.agent profile content 文本框

# Load environment variables from .env file
load_dotenv(dotenv_path=".env.local")

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

logger = logging.getLogger("voice-agent")

# 创建FastAPI应用
app = FastAPI()
llm_instance = None  # 将在entrypoint中初始化

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/config")
async def update_model_config(config: dict):
    if llm_instance:
        llm_instance.update_config(config)
    return {"status": "success"}

@app.get("/default-config")
async def get_default_configuration():
    return get_default_config()

async def run_fastapi():
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

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
