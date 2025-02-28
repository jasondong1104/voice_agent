import logging
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_config import update_config
from langchain_llm import LangchainLLM
# python agent.py --server-url wss://your-livekit-server --api-key your-api-key --api-secret your-api-secret
#todo: 1. 前端添加模型 baseurl 文本框 2....

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
from livekit.plugins import cartesia, openai, deepgram, silero, turn_detector

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

async def run_fastapi():
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # 启动FastAPI服务器
    global llm_instance
    llm_instance = LangchainLLM()
    asyncio.create_task(run_fastapi())

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            """个人资料：你是一位新入职山西太原市杨帆高中的物理老师；最近几个月在负责学校的招生工作，正在与学生家长进行电话联系。
谈话策略：在通话话开始阶段首先介绍自己。然后主动了解对方的家庭情况，孩子的升学情况。通话过程中要始终记得你的目的是为杨帆高中招生。在谈话过程中保持主动！每次对话都要注意跟通话目的和策略保持一致。
谈话的目的是了解对方的家庭情况，特别是对方的孩子的升学情况。如果对方的家庭中有孩子准备升学进入高中阶段，就向其推荐杨帆中学，并邀请其加入群聊；对方表达意愿后可逐渐结束对话。如果对方没有准备升学的孩子，逐渐结束对话。
约束：
回答使用口语化的语言，并保持简洁。
讲话的声音不要太慢，回答不要超过50个字。
回答具体问题时精炼一些，少用感叹句。
对于你不懂的问题，回答你自己不了解；并且邀请对方加入群聊向骨干老师咨询。
杨帆高中资料：
对学生的激励措施：高中期间的奖金和竞赛辅导。
免费的公益活动：
对有升学意向的初三学生提供免费的体育加试辅导和培训。
免费参观学校的活动。"""
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
        llm=llm_instance,  # 使用全局llm实例
        tts=cartesia.TTS(language="zh",speed='normal'),
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
